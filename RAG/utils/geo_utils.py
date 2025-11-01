import math

def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

def find_nearby_places(dataset, base_lat, base_lon, max_distance_km):
    results = []
    for rec in dataset:
        lat, lon = rec.get("latitude"), rec.get("longitude")
        if lat and lon:
            try:
                dist = haversine(float(base_lat), float(base_lon), float(lat), float(lon))
                if dist <= max_distance_km:
                    rec["distance_km"] = round(dist, 1)
                    results.append(rec)
            except Exception:
                pass
    results.sort(key=lambda x: x.get("distance_km", 999))
    return results[:8]
