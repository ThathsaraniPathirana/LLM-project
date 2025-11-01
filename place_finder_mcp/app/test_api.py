import googlemaps, os
from dotenv import load_dotenv

load_dotenv()
gmaps = googlemaps.Client(key=os.getenv("GOOGLE_MAPS_API_KEY"))

result = gmaps.places_nearby(location=(59.325, 18.07), radius=1000, type="restaurant")
print(result["results"][0]["name"])
