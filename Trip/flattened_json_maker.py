import json
import os

# Input & output files
INPUT_FILE = r"C:\Users\wmdna\Desktop\Scraping\Success - Copy\Trip\all_trip_sweden.json"
OUTPUT_FILE = r"C:\Users\wmdna\Desktop\Scraping\Success - Copy\Trip\all_trip_sweden_flat.json"


def extract_lang_value(values):
    """Return English text if possible; otherwise first available string."""
    if isinstance(values, list):
        for v in values:
            if isinstance(v, dict) and any(k in json.dumps(v).lower() for k in ["english", "eng", "en"]):
                return v.get("@value", "").strip()
        for v in values:
            if isinstance(v, dict) and "@value" in v:
                return v["@value"].strip()
    elif isinstance(values, dict):
        return values.get("@value", "").strip()
    elif isinstance(values, str):
        return values.strip()
    return None


def extract_id(value):
    """Handle dict, list, or string for @id."""
    if isinstance(value, dict):
        return value.get("@id")
    elif isinstance(value, list):
        for v in value:
            if isinstance(v, dict) and "@id" in v:
                return v["@id"]
            elif isinstance(v, str):
                return v
    elif isinstance(value, str):
        return value
    return None


def extract_geo(graphs, geo_ref):
    """Extract latitude/longitude if linked geo node exists."""
    if not geo_ref:
        return None, None
    for g in graphs:
        if g.get("@id") == geo_ref:
            return g.get("schema:latitude"), g.get("schema:longitude")
    return None, None


def flatten_graph(graphs):
    """Flatten @graph from Trip metadata into a single record."""
    main = next((g for g in graphs if g.get("@type") == "schema:Trip"), None)
    if not main:
        return None

    # --- Extract references ---
    addr_ref = extract_id(main.get("schema:address"))
    geo_ref = extract_id(main.get("schema:geo"))
    provider_ref = extract_id(main.get("schema:provider"))

    addr = next((g for g in graphs if g.get("@id") == addr_ref), {})
    geo = next((g for g in graphs if g.get("@id") == geo_ref), {})
    provider = next((g for g in graphs if g.get("@id") == provider_ref), {})

    latitude, longitude = extract_geo(graphs, geo_ref)

    flat = {
        "name": extract_lang_value(main.get("schema:name")),
        "type": main.get("@type"),
        "additional_type": extract_id(main.get("schema:additionalType")),  # e.g. schema:BoatTrip
        "description": extract_lang_value(main.get("schema:description")),
        "facts_text": extract_lang_value(main.get("dcterms:abstract")),
        "url": extract_id(main.get("schema:url")),
        "region": extract_id(main.get("dcterms:spatial")),
        "provider": extract_lang_value(provider.get("schema:name")) if provider else None,
        "destination": extract_id(main.get("schema:itinerary")),
        "latitude": latitude,
        "longitude": longitude,
        "image": extract_id(main.get("schema:image")),
    }

    for k, v in flat.items():
        if v in ["", [], {}, None]:
            flat[k] = None

    return flat


def main():
    print("📂 Reading JSON file...")
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    flattened = []
    for entry in data:
        graphs = entry.get("metadata", {}).get("@graph", [])
        record = flatten_graph(graphs)
        if record:
            flattened.append(record)

    print(f"✅ Flattened {len(flattened)} trip entries.")
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(flattened, f, ensure_ascii=False, indent=2)

    print(f"💾 Saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
