import csv
import time
from atproto import Client

# ==== CONFIGURATION ====
# Login information
USERNAME = "team10.bsky.social"
PASSWORD = "trustandsafety"

# Initialize client
client = Client()
client.login(USERNAME, PASSWORD)

# Single query for normal crypto-related posts
crypto_query = ["crypto OR bitcoin OR ethereum"]

scam_query = [
    "crypto AND giveaway",
    "crypto AND promo",
    "crypto AND airdrop",
    # "crypto AND code",
    # "crypto AND returns",
    # "crypto AND guarantees"
]

# ========================

# ==== TESTING QUERIES ====
# # Define test queries
# test_queries = [
#     "crypto",  # simple, should work
#     "bitcoin",  # simple, should work
#     "ethereum",  # simple, should work
#     "crypto OR bitcoin",  # simple OR
#     "(crypto OR bitcoin OR ethereum)",  # parenthesis
#     "crypto AND giveaway", #success
#     "crypto AND (giveaway OR promo)" #failed
# ]

# def try_query(q):
#     try:
#         print(f"\n=== Trying query: {q} ===")
#         results = client.app.bsky.feed.search_posts({'q': q, 'limit': 5, 'sort': 'latest', 'lang': 'en'})
#         posts = results.posts or []
#         print(f"✅ Success. Found {len(posts)} posts.")
#         if posts:
#             print(f"Example post text: {posts[0].record.text[:100]}...")
#     except Exception as e:
#         print(f"❌ Failed with error: {e}")

# # Run tests
# for query in test_queries:
#     try_query(query)
#     time.sleep(1)  # small sleep between calls

# ========================


def search_posts(keywords, max_posts):
    collected_posts = []

    for keyword in keywords:
        cursor = None
        keyword_posts = []

        while len(keyword_posts) < max_posts:
            results = client.app.bsky.feed.search_posts(
                {
                    "q": keyword,
                    "limit": 25,
                    "cursor": cursor,
                    "sort": "latest",
                    "lang": "en",
                }
            )

            posts = results.posts or []

            if not posts:
                print(f"No more posts found for keyword: {keyword}")
                break  # Stop if no posts are returned

            new_posts_found = 0
            for post in posts:
                record = post.record
                if not record or not hasattr(record, "text"):
                    continue
                text = record.text
                creator = post.author.handle
                likes = getattr(post, "like_count", 0)
                reposts = getattr(post, "repost_count", 0)
                replies = getattr(post, "reply_count", 0)

                # --- Extract links ---
                links = []
                if hasattr(record, "facets") and record.facets:
                    for facet in record.facets:
                        for feature in facet.features:
                            if hasattr(feature, "uri"):
                                links.append(feature.uri)

                # --- Extract embedded images ---
                image_urls = []
                if post.embed and hasattr(post.embed, "images"):
                    for image in post.embed.images:
                        if hasattr(image, "fullsize"):
                            image_urls.append(image.fullsize)

                # --- Extract hashtags (tags) ---
                tags = []
                if hasattr(record, "tags") and record.tags:
                    tags.extend(record.tags)

                keyword_posts.append(
                    {
                        "text": text,
                        "creator": creator,
                        "likes": likes,
                        "reposts": reposts,
                        "responses": replies,
                        "links": ", ".join(links),
                        "images": ", ".join(image_urls),
                        "tags": ", ".join(tags),
                    }
                )
                new_posts_found += 1

                if len(keyword_posts) >= max_posts:
                    break

            cursor = getattr(results, "cursor", None)
            if not cursor:
                print(f"No next page for keyword: {keyword}")
                break  # Stop if no next page
            if new_posts_found == 0:
                print(f"No new posts added for keyword: {keyword}")
                break  # Stop if no new posts were added

            time.sleep(1)  # Gentle rate limiting

        collected_posts.extend(keyword_posts)

    return collected_posts


# Collect normal crypto posts
print("Collecting normal crypto posts...")
normal_posts = search_posts(crypto_query, 50)

# Collect scam-like crypto posts
print("Collecting scam-like giveaway posts...")
scam_posts = search_posts(scam_query, 20)

# Combine datasets
all_posts = normal_posts + scam_posts

print(
    f"✅ Collected {len(normal_posts)} normal posts and {len(scam_posts)} scam posts."
)

# Save to CSV
csv_filename = "crypto_posts_dataset.csv"
with open(csv_filename, mode="w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(
        file,
        fieldnames=[
            "text",
            "creator",
            "likes",
            "reposts",
            "responses",
            "links",
            "images",
            "tags",
            "label",
        ],
    )
    writer.writeheader()

    for post in normal_posts:
        post["label"] = "normal"
        writer.writerow(post)

    for post in scam_posts:
        post["label"] = "scam"
        writer.writerow(post)

print(f"Saved {len(all_posts)} posts to {csv_filename}")
