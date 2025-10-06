import requests
import json
import time

# --- Configuration ---
BASE_URL = "https://data.visitsweden.com/store/search"
QUERY = "public:true+AND+rdfType:http%5C%3A%2F%2Fschema.org%2FTrip"
LIMIT = 100  # max per request
OUT_FILE = "all_trip_sweden.json"

all_results = []
offset = 0

print("🔎 Fetching all places from VisitSweden API...")

while True:
    # Construct paginated URL
    url = f"{BASE_URL}?type=solr&query={QUERY}&limit={LIMIT}&offset={offset}&rdfFormat=application/ld+json"
    print(f"Fetching offset={offset} ...")

    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        data = response.json()

        # Some responses might return {"results": 0, ...}
        results = data.get("results", 0)
        if results == 0:
            print("✅ Done — no more results.")
            break

        # Extract the actual entries (depends on structure)
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

        # Stop if fewer than limit (last page)
        if results < LIMIT:
            break

        offset += LIMIT
        time.sleep(1)  # be kind to API

    except Exception as e:
        print(f"⚠️ Error at offset {offset}: {e}")
        break

# --- Save to JSON file ---
with open(OUT_FILE, "w", encoding="utf-8") as f:
    json.dump(all_results, f, ensure_ascii=False, indent=2)

print(f"\n💾 Saved {len(all_results)} total entries to {OUT_FILE}")
