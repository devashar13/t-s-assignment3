import os
import csv
import re
from typing import List, Optional
from urllib.parse import urlparse

from atproto import Client, models
from atproto.exceptions import AtProtocolError, RequestException

T_AND_S_LABEL = "t-and-s"

class AutomatedLabeler:
    """Automated labeler implementation for T&S keyword/domain matching and news citation."""

    def __init__(self, client: Client, input_dir: str):
        """
        Initializes the AutomatedLabeler.

        Args:
            client: An authenticated atproto.Client instance.
            input_dir: The directory containing the T&S word/domain lists and news domains.
        """
        self.client = client
        self._load_t_and_s_lists(input_dir)
        self._load_news_domains(input_dir)
        self._did_cache = {}

    def _load_t_and_s_lists(self, d: str):
        """Loads T&S words and domains from CSV files."""
        self.words = set()
        self.domains = set()
        words_path = os.path.join(d, 't-and-s-words.csv')
        domains_path = os.path.join(d, 't-and-s-domains.csv')

        try:
            with open(words_path, newline='', encoding='utf-8') as f:
                reader = csv.reader(f)
                self.words = {row[0].strip().lower() for row in reader if row and row[0].strip()}
        except FileNotFoundError:
            print(f"Warning: T&S words file not found at {words_path}.")
        except Exception as e:
            print(f"Error loading T&S words from {words_path}: {e}")

        try:
            with open(domains_path, newline='', encoding='utf-8') as f:
                reader = csv.reader(f)
                self.domains = {row[0].strip().lower() for row in reader if row and row[0].strip()}
        except FileNotFoundError:
            print(f"Warning: T&S domains file not found at {domains_path}.")
        except Exception as e:
            print(f"Error loading T&S domains from {domains_path}: {e}")

    def _load_news_domains(self, d: str):
        """Loads news domains and their corresponding labels from CSV file."""
        self.news_domain_map = {}
        news_path = os.path.join(d, 'news-domains.csv')
        try:
            with open(news_path, newline='', encoding='utf-8') as f:
                reader = csv.reader(f)
                for row in reader:
                    if row and row[0].strip():
                        domain = row[0].strip().lower()
                        label = row[1].strip() if len(row) > 1 else domain
                        self.news_domain_map[domain] = label
        except FileNotFoundError:
            print(f"Warning: News domains file not found at {news_path}.")
        except Exception as e:
            print(f"Error loading news domains from {news_path}: {e}")

    def _resolve_handle_to_did(self, handle: str) -> Optional[str]:
        """Resolves a handle to a DID using the client, with caching."""
        if handle in self._did_cache:
            return self._did_cache[handle]
        try:
            response = self.client.resolve_handle(handle=handle)
            did = response.did
            self._did_cache[handle] = did
            return did
        except (RequestException, AtProtocolError, Exception) as e:
            print(f"Warning: Could not resolve handle '{handle}' to DID: {type(e).__name__} - {e}")
            self._did_cache[handle] = None
            return None

    def _convert_web_url_to_at_uri(self, url: str) -> Optional[str]:
        """
        Tries to convert a bsky.app web URL to an AT URI.
        Returns the AT URI string if successful, None otherwise.
        """
        if not url.startswith("https://bsky.app/profile/"):
            print(f"Warning: URL '{url}' is not a recognizable bsky.app profile URL.")
            return None

        try:
            parsed_url = urlparse(url)
            path_parts = parsed_url.path.strip('/').split('/')
            if len(path_parts) == 4 and path_parts[0] == 'profile' and path_parts[2] == 'post':
                handle = path_parts[1]
                rkey = path_parts[3]
                did = self._resolve_handle_to_did(handle)
                if did:
                    return f"at://{did}/app.bsky.feed.post/{rkey}"
            print(f"Warning: Unexpected URL path format: {parsed_url.path}")
        except Exception as e:
            print(f"Warning: Error parsing URL '{url}' or resolving handle: {e}")
        return None

    def moderate_post(self, url: str) -> List[str]:
        """
        Applies moderation to the post specified by the given URL (Web or AT URI).
        Labels it "t-and-s" if it contains matching words or domains, and applies
        news publication labels for any linked articles.

        Args:
            url: The Web URL (bsky.app) or AT URI of the post.

        Returns:
            A list of labels if matches are found, otherwise an empty list.
        """
        labels = set()
        at_uri = None

        # Determine AT URI
        if url.startswith("at://"):
            at_uri = url
        elif url.startswith("https://bsky.app/"):
            at_uri = self._convert_web_url_to_at_uri(url)
        else:
            print(f"Error: Input URL '{url}' is neither an AT URI nor a bsky.app URL.")
            return []

        if not at_uri:
            print(f"Error: Failed to obtain a valid AT URI for '{url}'. Skipping moderation.")
            return []

        try:
            # Fetch the post
            response = self.client.app.bsky.feed.get_post_thread({'uri': at_uri, 'depth': 0})
            if not response or not response.thread or not isinstance(response.thread, models.AppBskyFeedDefs.ThreadViewPost):
                return []

            post_view = response.thread.post
            record = getattr(post_view, 'record', None)
            text = getattr(record, 'text', '') if record else ''
            if not text:
                return []

            text_lower = text.lower()

            # T&S keyword matching
            for word in self.words:
                try:
                    if re.search(rf"\b{re.escape(word)}\b", text_lower):
                        labels.add(T_AND_S_LABEL)
                        break
                except re.error as re_err:
                    print(f"Debug: Regex error for word '{word}': {re_err}")

            # T&S domain matching (if no keyword found)
            if T_AND_S_LABEL not in labels:
                for domain in self.domains:
                    if domain in text_lower:
                        labels.add(T_AND_S_LABEL)
                        break

            # News citation labeling
            urls_in_text = re.findall(r'https?://\S+', text)
            for u in urls_in_text:
                try:
                    parsed = urlparse(u)
                    host = parsed.netloc.lower().split(':')[0]
                    if host.startswith('www.'):
                        host = host[4:]
                    for domain, label in self.news_domain_map.items():
                        if host == domain or host.endswith(f'.{domain}'):
                            labels.add(label)
                except Exception:
                    continue

        except (AtProtocolError, RequestException) as api_error:
            print(f"API Error processing post AT URI {at_uri}: {type(api_error).__name__} - {api_error}")
            return []
        except Exception as e:
            print(f"Unexpected Error processing post AT URI {at_uri}: {type(e).__name__} - {e}")
            return []

        return list(labels)
