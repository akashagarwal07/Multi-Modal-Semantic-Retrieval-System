import requests
import json
import time
import random
import csv
import os

API_URL = "https://api.semanticscholar.org/graph/v1/paper/search"

TOPICS = [
    "artificial intelligence",
    "machine learning",
    "deep learning",
    "neural networks",
    "computer vision",
    "natural language processing",
    "information retrieval",
    "blockchain",
    "parallel processing"
]

PER_TOPIC_LIMIT = 600      # <<< LIMIT EACH TOPIC TO 600 PAPERS
PAGE_SIZE = 100
FIELDS = "title,year,authors,abstract,externalIds,url,openAccessPdf"

OUTPUT_JSONL = "semantic_scholar_results.jsonl"
OUTPUT_CSV = "semantic_scholar_results.csv"

seen_ids = set()
results = []

# Load previous session
if os.path.exists(OUTPUT_JSONL):
    with open(OUTPUT_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            p = json.loads(line)
            seen_ids.add(p["paperId"])
            results.append(p)
else:
    open(OUTPUT_JSONL, "w").close()


def fetch_page(query, offset, attempt=1):
    params = {
        "query": query,
        "offset": offset,
        "limit": PAGE_SIZE,
        "fields": FIELDS
    }

    try:
        r = requests.get(API_URL, params=params, timeout=30)

        # 429 Too Many Requests
        if r.status_code == 429:
            # exponential backoff with jitter
            base_wait = 10 * attempt
            jitter = random.uniform(0.5, 1.5)
            wait = base_wait * jitter

            # HARD CAP at 90 seconds
            wait = min(wait, 90)

            print(f"⏳ 429 throttled. Waiting {wait:.1f}s before retry...")
            time.sleep(wait)
            return fetch_page(query, offset, attempt + 1)

        r.raise_for_status()
        return r.json()

    except Exception as e:
        # network/server error
        base_wait = 5 * attempt
        jitter = random.uniform(0.5, 1.5)
        wait = base_wait * jitter

        # HARD CAP at 90 seconds
        wait = min(wait, 90)

        print(f"⚠ Error at offset {offset}: {e}")
        print(f"Retrying in {wait:.1f}s...")
        time.sleep(wait)
        return fetch_page(query, offset, attempt + 1)



print("\n==============================")
print("     FETCHING PAPERS")
print("==============================\n")

for topic in TOPICS:
    print(f"\n🔍 Topic: {topic}")

    offset = 0
    topic_count = 0

    while topic_count < PER_TOPIC_LIMIT:
        print(f"  → Fetching offset {offset}...")

        data = fetch_page(topic, offset)
        if not data or "data" not in data:
            print("  ⚠ No results returned")
            break

        batch = data["data"]
        if not batch:
            print("  ⚠ No more results for this topic")
            break

        added = 0

        for p in batch:
            pid = p.get("paperId")
            if not pid or pid in seen_ids:
                continue

            seen_ids.add(pid)

            record = {
                "paperId": pid,
                "title": p.get("title"),
                "year": p.get("year"),
                "abstract": p.get("abstract"),
                "authors": [a.get("name") for a in p.get("authors", [])],
                "url": p.get("url"),
                "doi": p.get("externalIds", {}).get("DOI"),
                "arxivId": p.get("externalIds", {}).get("ArXiv"),
                "openAccessPdf": p.get("openAccessPdf", {}).get("url")
            }

            results.append(record)
            added += 1
            topic_count += 1

            with open(OUTPUT_JSONL, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")

            if topic_count >= PER_TOPIC_LIMIT:
                break

        print(f"    +{added} new papers | Topic total: {topic_count}")

        offset += PAGE_SIZE
        time.sleep(random.uniform(0.3, 1.0))  # polite delay

print("\n==============================")
print("            DONE")
print("==============================")
print(f"Total papers collected: {len(results)}")

# Save CSV
with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)

print(f"\n📁 JSONL saved → {OUTPUT_JSONL}")
print(f"📁 CSV saved → {OUTPUT_CSV}")
