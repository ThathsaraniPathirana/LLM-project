import json
import os

INPUT_FILE = r"C:\Users\wmdna\Desktop\Scraping\Success - Copy\Places\all_places_sweden.json"
OUTPUT_FILE = r"c:\Users\wmdna\Desktop\Scraping\Success - Copy\Places\all_places_sweden_flat.json"


def extract_lang_value(values):
    """Return English text if available; otherwise first available string."""
    if isinstance(values, list):
        for v in values:
            if isinstance(v, dict) and any(x in json.dumps(v).lower() for x in ["english", "eng"]):
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


def extract_opening_hours(value):
    """Parse schema:openingHoursSpecification into structured format."""
    if not value:
        return None
    if isinstance(value, dict):
        value = [value]
    entries = []
    for v in value:
        opens = v.get("schema:opens")
        closes = v.get("schema:closes")
        day = v.get("schema:dayOfWeek")
        valid_from = v.get("schema:validFrom")
        valid_to = v.get("schema:validThrough")
        desc = v.get("schema:description")
        entry = {
            "dayOfWeek": day,
            "opens": opens,
            "closes": closes,
            "validFrom": valid_from,
            "validThrough": valid_to,
            "description": desc
        }
        clean = {k: v for k, v in entry.items() if v}
        if clean:
            entries.append(clean)
    return entries or None


def flatten_graph(graphs):
    """Flatten schema:Place @graph entry into one record."""
    main = next((g for g in graphs if g.get("@type") == "schema:Place"), None)
    if not main:
        return None

    geo_id = extract_id(main.get("schema:geo"))
    addr_id = extract_id(main.get("schema:address"))

    geo = next((g for g in graphs if g.get("@id") == geo_id), {})
    addr = next((g for g in graphs if g.get("@id") == addr_id), {})

    flat = {
        # Core identity
        "name": extract_lang_value(main.get("schema:name")),
        "alternate_name": extract_lang_value(main.get("schema:alternateName")),
        "type": main.get("@type"),
        "additional_type": extract_id(main.get("schema:additionalType")),  # e.g., schema:Museum, schema:Park, visit:Adventure
        # Description
        "description": extract_lang_value(main.get("schema:description")),
        "facts_text": extract_lang_value(main.get("dcterms:abstract")),
        # URLs and media
        "url": extract_id(main.get("schema:url")),
        "main_image": extract_id(main.get("schema:image")),
        "photo": extract_id(main.get("schema:photo")),
        # Location and geography
        "region": extract_id(main.get("dcterms:spatial")),
        "city": addr.get("schema:addressLocality"),
        "country": addr.get("schema:addressCountry"),
        "street": addr.get("schema:streetAddress"),
        "latitude": geo.get("schema:latitude"),
        "longitude": geo.get("schema:longitude"),
        # Time / Hours
        "opening_hours": extract_opening_hours(main.get("schema:openingHoursSpecification")),
        # Prices, features, events
        "price": main.get("schema:price"),
        "certificate": extract_id(main.get("schema:hasCredential")),
        "member_of": extract_id(main.get("schema:memberOf")),
        "parent_org": extract_id(main.get("schema:parentOrganization")),
        "contained_in_place": extract_id(main.get("schema:containedInPlace")),
        "associated_place": extract_id(main.get("schema:containsPlace")),
        "trail": extract_id(main.get("schema:geoContains")),
        "event": extract_id(main.get("schema:event")),
        "amenities": extract_id(main.get("schema:amenityFeature")),
        # For extra mapping (from 5.3 Type of place)
        "place_category": extract_id(main.get("schema:additionalType")),  # category mapping placeholder
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

    print(f"✅ Flattened {len(flattened)} places.")
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(flattened, f, ensure_ascii=False, indent=2)

    print(f"💾 Saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
