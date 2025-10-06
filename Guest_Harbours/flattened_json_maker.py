import json
import os

INPUT_FILE = r"C:\Users\wmdna\Desktop\Scraping\Success\Guest_Harbours\all_guestharbours_sweden.json"
OUTPUT_FILE = r"c:\Users\wmdna\Desktop\Scraping\Success\Guest_Harbours\all_guestharbours_sweden_flat.json"


def extract_lang_value(values):
    """Return English text if possible; otherwise Swedish or first available string."""
    if isinstance(values, list):
        eng = next((v.get("@value") for v in values if isinstance(v, dict) and "eng" in json.dumps(v).lower()), None)
        swe = next((v.get("@value") for v in values if isinstance(v, dict) and "sv" in json.dumps(v).lower()), None)
        any_val = next((v.get("@value") for v in values if isinstance(v, dict) and "@value" in v), None)
        return eng or swe or any_val
    elif isinstance(values, dict):
        return values.get("@value")
    elif isinstance(values, str):
        return values.strip()
    return None


def extract_id(value):
    """Safely extract @id from dict, list, or string."""
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


def flatten_graph(graphs):
    """Flatten guest harbour @graph into single record."""
    main = next((g for g in graphs if g.get("@type") == "http://www.wikidata.org/entity/Q283202"), None)
    if not main:
        return None

    # Linked nodes
    addr_id = extract_id(main.get("schema:address"))
    geo_id = extract_id(main.get("schema:geo"))
    contained_id = extract_id(main.get("schema:containedInPlace"))
    region_id = extract_id(main.get("dcterms:spatial"))

    addr = next((g for g in graphs if g.get("@id") == addr_id), {})
    geo = next((g for g in graphs if g.get("@id") == geo_id), {})
    region = next((g for g in graphs if g.get("@id") == region_id), {})
    contained = next((g for g in graphs if g.get("@id") == contained_id), {})

    flat = {
        "name": extract_lang_value(main.get("schema:name")),
        "description": extract_lang_value(main.get("schema:description")),
        "facts": extract_lang_value(main.get("dcterms:abstract")),
        "image": extract_id(main.get("schema:image")),
        "homepage": extract_id(main.get("schema:url")),
        "price": extract_lang_value(main.get("schema:price")),
        "telephone": extract_lang_value(main.get("schema:telephone")),
        "street_address": extract_lang_value(addr.get("schema:streetAddress")),
        "city": extract_lang_value(addr.get("schema:addressLocality")),
        "country": addr.get("schema:addressCountry"),
        "latitude": geo.get("schema:latitude"),
        "longitude": geo.get("schema:longitude"),
        "opening_hours": extract_lang_value(main.get("schema:openingHoursSpecification")),
        "amenities": extract_lang_value(main.get("schema:amenityFeature")),
        "contained_in_place": extract_lang_value(contained.get("schema:name")),
        "associated_place": extract_lang_value(main.get("schema:containsPlace")),
        "member_of": extract_lang_value(main.get("schema:memberOf")),
        "region": extract_id(main.get("dcterms:spatial")),
    }

    # Clean nulls
    for k, v in flat.items():
        if v in ["", [], {}, None]:
            flat[k] = None

    return flat


def main():
    print("📂 Reading Guest Harbour JSON file...")
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    flattened = []
    for entry in data:
        graphs = entry.get("metadata", {}).get("@graph", [])
        record = flatten_graph(graphs)
        if record:
            flattened.append(record)

    print(f"✅ Flattened {len(flattened)} guest harbour entries.")
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(flattened, f, ensure_ascii=False, indent=2)

    print(f"💾 Saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
