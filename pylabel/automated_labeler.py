import os
import csv
import re
from typing import List, Optional, Dict, Set
from urllib.parse import urlparse
import requests
from PIL import Image
import imagehash
import io
import numpy as np

from atproto import Client, models
from atproto.exceptions import AtProtocolError, RequestException

T_AND_S_LABEL = "t-and-s"
DOG_LABEL = "dog"
# Conservative threshold based on observed minimum distance
THRESH = 18

class AutomatedLabeler:
    """Automated labeler implementation for T&S keyword/domain matching, news citation, and image matching."""

    def __init__(self, client: Client, input_dir: str):
        """
        Initializes the AutomatedLabeler.

        Args:
            client: An authenticated atproto.Client instance.
            input_dir: The directory containing the T&S word/domain lists, news domains, and dog-list images.
        """
        self.client = client
        self._load_t_and_s_lists(input_dir)
        self._load_news_domains(input_dir)
        self._load_dog_list_images(input_dir)
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

    def _load_dog_list_images(self, d: str):
        """Loads and hashes dog-list images from the specified directory."""
        self.dog_hashes = []
        dog_images_dir = os.path.join(d, 'dog-list-images')
        
        try:
            for filename in os.listdir(dog_images_dir):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_path = os.path.join(dog_images_dir, filename)
                    try:
                        with Image.open(image_path) as img:
                            if img.mode != 'RGB':
                                img = img.convert('RGB')
                            phash = imagehash.phash(img)
                            self.dog_hashes.append(phash)
                    except Exception as e:
                        print(f"Error processing dog-list image {filename}: {e}")
        except Exception as e:
            print(f"Error loading dog-list images: {e}")

    def _get_post_images(self, post_view: models.AppBskyFeedDefs.PostView) -> List[str]:
        """Extracts image URLs from a post."""
        images = []
        try:
            if hasattr(post_view, 'embed') and post_view.embed:
                if isinstance(post_view.embed, models.AppBskyEmbedImages.View):
                    for image in post_view.embed.images:
                        if hasattr(image, 'fullsize'):
                            images.append(image.fullsize)
                elif isinstance(post_view.embed, models.AppBskyEmbedRecord.View):
                    if hasattr(post_view.embed.record, 'embeds'):
                        for embed in post_view.embed.record.embeds:
                            if isinstance(embed, models.AppBskyEmbedImages.View):
                                for image in embed.images:
                                    if hasattr(image, 'fullsize'):
                                        images.append(image.fullsize)
        except Exception as e:
            print(f"Error extracting images from post: {e}")
        return images

    def _hash_image_from_url(self, url: str) -> Optional[imagehash.ImageHash]:
        """Downloads and hashes an image from a URL."""
        try:
            response = requests.get(url)
            response.raise_for_status()
            img = Image.open(io.BytesIO(response.content))
            if img.mode != 'RGB':
                img = img.convert('RGB')
            phash = imagehash.phash(img)
            return phash
        except Exception as e:
            print(f"Error hashing image from {url}: {e}")
            return None

    def _check_image_match(self, img_hash: imagehash.ImageHash) -> bool:
        """Checks if an image hash matches any dog-list image within the threshold."""
        distances = []
        
        for dog_hash in self.dog_hashes:
            distance = img_hash - dog_hash
            distances.append(distance)
        
        distances = np.array(distances)
        min_distance = np.min(distances)
        mean_distance = np.mean(distances)
        std_distance = np.std(distances)
        sorted_distances = sorted(distances)
        
        if (min_distance == 0 or
            min_distance <= 8 or
            (min_distance <= 16 and 
             (sorted_distances[1] - min_distance) >= 3 and
             (mean_distance - min_distance) > 1.5 * std_distance)):
            return True
        
        return False

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
        Labels it "t-and-s" if it contains matching words or domains, applies
        news publication labels for any linked articles, and applies "dog" label
        if it contains matching images.

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

            # Image matching
            image_urls = self._get_post_images(post_view)
            for img_url in image_urls:
                img_hash = self._hash_image_from_url(img_url)
                if img_hash and self._check_image_match(img_hash):
                    labels.add(DOG_LABEL)
                    break

            # Only proceed with text-based checks if no dog image was found
            if DOG_LABEL not in labels and text:
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
