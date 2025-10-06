import json

def safe_get(d, key, default=None):
    """Safely handle dict or list access."""
    if isinstance(d, dict):
        return d.get(key, default)
    elif isinstance(d, list) and d:
        # Return first element if list of dicts or strings
        first = d[0]
        return first.get(key, default) if isinstance(first, dict) else first
    return default

def extract_lang_value(obj):
    """Extract multilingual value — returns first @value if list."""
    if isinstance(obj, list):
        return obj[0].get("@value") if obj and isinstance(obj[0], dict) else obj[0]
    elif isinstance(obj, dict):
        return obj.get("@value")
    elif isinstance(obj, str):
        return obj
    return None

def flatten_event(graphs):
    """Flatten a single event entry."""
    event_node = next((g for g in graphs if g.get("@type") == "schema:Event"), None)
    if not event_node:
        return None

    # --- Basic fields ---
    record = {
        "id": event_node.get("dcterms:identifier"),
        "type": safe_get(event_node, "@type"),
        "event_type": safe_get(safe_get(event_node, "schema:additionalType"), "@id"),
        "name": extract_lang_value(event_node.get("schema:name")),
        "description": extract_lang_value(event_node.get("schema:description")),
        "facts_text": extract_lang_value(event_node.get("dcterms:abstract")),
        "url": safe_get(safe_get(event_node, "schema:url"), "@id"),
        "region": safe_get(safe_get(event_node, "dcterms:spatial"), "@id"),
        "capacity": safe_get(event_node, "schema:maximumAttendeeCapacity"),
        "price": safe_get(event_node, "schema:price"),
        "telephone": safe_get(event_node, "schema:telephone"),
        "organizer": safe_get(safe_get(event_node, "schema:organizer"), "@id"),
        "audience": safe_get(safe_get(event_node, "schema:audience"), "@id"),
        "event_schedule": safe_get(safe_get(event_node, "schema:eventSchedule"), "@id"),
        "duration": safe_get(event_node, "schema:duration"),
        "start_date": extract_lang_value(event_node.get("schema:startDate")),
        "end_date": extract_lang_value(event_node.get("schema:endDate")),
        "main_image": safe_get(safe_get(event_node, "schema:image"), "@id"),
        "photo": [safe_get(p, "@id") for p in event_node.get("schema:photo", []) if isinstance(p, dict)],
    }

    # --- Address ---
    address_id = safe_get(safe_get(event_node, "schema:address"), "@id")
    address_node = next((g for g in graphs if g.get("@id") == address_id), None)
    if address_node:
        record.update({
            "street": extract_lang_value(address_node.get("schema:streetAddress")),
            "city": extract_lang_value(address_node.get("schema:addressLocality")),
            "country": address_node.get("schema:addressCountry"),
        })
    else:
        record.update({"street": None, "city": None, "country": None})

    # --- Geo ---
    geo_id = safe_get(safe_get(event_node, "schema:geo"), "@id")
    geo_node = next((g for g in graphs if g.get("@id") == geo_id), None)
    if geo_node:
        record.update({
            "latitude": safe_get(geo_node, "schema:latitude"),
            "longitude": safe_get(geo_node, "schema:longitude"),
            "location_name": extract_lang_value(geo_node.get("schema:name")),
        })
    else:
        record.update({"latitude": None, "longitude": None, "location_name": None})

    return record


def main():
    input_file = r"C:\Users\wmdna\Desktop\Scraping\Success - Copy\Events\all_events_sweden.json"
    output_file = r"C:\Users\wmdna\Desktop\Scraping\Success - Copy\Events\all_events_sweden_flat.json"

    print(f"📂 Reading {input_file}...")
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    flattened = []
    for entry in data:
        graphs = entry.get("metadata", {}).get("@graph", [])
        rec = flatten_event(graphs)
        if rec:
            flattened.append(rec)

    print(f"✅ Flattened {len(flattened)} events.")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(flattened, f, ensure_ascii=False, indent=2)

    print(f"💾 Saved to {output_file}")


if __name__ == "__main__":
    main()
