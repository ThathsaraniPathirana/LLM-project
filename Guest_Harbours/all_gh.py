import requests
import json
import time

BASE_URL = "https://data.visitsweden.com/store/search"
# Wikidata URI must be escaped with %5C%3A etc.
QUERY = "public:true+AND+rdfType:http%5C%3A%2F%2Fwww.wikidata.org%2Fentity%2FQ283202"
LIMIT = 100
OUT_FILE = "all_guestharbours_sweden.json"

all_results = []
offset = 0

print("⚓ Fetching all guest harbours from VisitSweden API...")

while True:
    url = f"{BASE_URL}?type=solr&query={QUERY}&limit={LIMIT}&offset={offset}&rdfFormat=application/ld+json"
    print(f"Fetching offset={offset} ...")

    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        data = r.json()

        results = data.get("results", 0)
        if results == 0:
            print("✅ Done — no more results.")
            break

        if isinstance(data, dict):
            if "resource" in data and "children" in data["resource"]:
                entries = data["resource"]["children"]
            elif isinstance(data.get("resource"), list):
                entries = data["resource"]
            else:
                entries = [data]
        elif isinstance(data, list):
            entries = data
        else:
            entries = []

        all_results.extend(entries)
        print(f"  → Added {len(entries)} entries, total so far: {len(all_results)}")

        if results < LIMIT:
            break

        offset += LIMIT
        time.sleep(1)

    except Exception as e:
        print(f"⚠️ Error at offset {offset}: {e}")
        break

with open(OUT_FILE, "w", encoding="utf-8") as f:
    json.dump(all_results, f, ensure_ascii=False, indent=2)

print(f"\n💾 Saved {len(all_results)} total guest harbour entries to {OUT_FILE}")
