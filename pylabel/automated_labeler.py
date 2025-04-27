# pylabel/automated_labeler.py

import os
import csv
import re
from typing import List, Optional
from urllib.parse import urlparse

from atproto import Client, models
from atproto.exceptions import AtProtocolError, RequestException

T_AND_S_LABEL = "t-and-s"

class AutomatedLabeler:
    """Automated labeler implementation for T&S keyword/domain matching."""

    def __init__(self, client: Client, input_dir: str):
        """
        Initializes the AutomatedLabeler.

        Args:
            client: An authenticated atproto.Client instance.
            input_dir: The directory containing the T&S word/domain lists.
        """
        self.client = client
        self._load_t_and_s_lists(input_dir)
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
                    at_uri = f"at://{did}/app.bsky.feed.post/{rkey}"
                    return at_uri
                else:
                    return None
            else:
                print(f"Warning: Unexpected URL path format: {parsed_url.path}")
                return None
        except Exception as e:
            print(f"Warning: Error parsing URL '{url}' or resolving handle: {e}")
            return None

    def moderate_post(self, url: str) -> List[str]:
        """
        Applies moderation to the post specified by the given URL (Web or AT URI).
        Labels it "t-and-s" if it contains matching words or domains.

        Args:
            url: The Web URL (bsky.app) or AT URI of the post.

        Returns:
            A list containing the 't-and-s' label if a match is found,
            otherwise an empty list. Returns empty list on error.
        """
        labels_to_apply = []
        found_match = False
        at_uri = None

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
            response = self.client.app.bsky.feed.get_post_thread({'uri': at_uri, 'depth': 0})

            if not response or not response.thread or not isinstance(response.thread, models.AppBskyFeedDefs.ThreadViewPost):
                 return []

            post_view = response.thread.post
            record = getattr(post_view, 'record', None)
            text = getattr(record, 'text', '') if record else ''

            if not text:
                return []

            text_lower = text.lower()

            for word in self.words:
                try:
                    if re.search(rf'\b{re.escape(word)}\b', text_lower):
                        found_match = True
                        break
                except re.error as re_err:
                     print(f"Debug: Regex error for word '{word}': {re_err}")
                     continue

            if not found_match:
                for domain in self.domains:
                    if domain in text_lower:
                        found_match = True
                        break

            if found_match:
                labels_to_apply.append(T_AND_S_LABEL)

        except (AtProtocolError, RequestException) as api_error:
             print(f"API Error processing post AT URI {at_uri}: {type(api_error).__name__} - {api_error}")
             return []
        except Exception as e:
            print(f"Unexpected Error processing post AT URI {at_uri}: {type(e).__name__} - {e}")
            return []

        return labels_to_apply