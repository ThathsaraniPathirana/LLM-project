import googlemaps
import os
from dotenv import load_dotenv
from .schemas import PlaceInfo, PlaceResponse
from loguru import logger

load_dotenv()
API_KEY = os.getenv("GOOGLE_MAPS_API_KEY")
gmaps = googlemaps.Client(key=API_KEY)

def _get_price_level_text(price_level):
    """Convert price level to readable text."""
    price_map = {
        1: "Budget-friendly",
        2: "Moderate",
        3: "Upscale",
        4: "Luxury"
    }
    return price_map.get(price_level, "Price not available")

def _is_valid_place(place, category):
    """
    Check if a place is actually what we're looking for.
    Filters out misclassified results.
    """
    types = place.get("types", [])
    name = place.get("name", "").lower()
    
    if category == "restaurant":
        # Must be food-related, NOT lodging
        food_types = ["restaurant", "cafe", "food", "bar", "meal_takeaway", "meal_delivery"]
        lodging_types = ["lodging", "hotel", "travel_agency"]
        
        has_food_type = any(t in types for t in food_types)
        has_lodging_type = any(t in types for t in lodging_types)
        
        # Exclude if it's clearly lodging
        if has_lodging_type and not has_food_type:
            return False
        # Exclude if name contains hotel/hostel but no food indicators
        if any(word in name for word in ["hotel", "hotell", "hostel"]) and not has_food_type:
            return False
            
        return True
        
    elif category == "hotel":
        # Must have lodging-related types
        lodging_types = ["lodging", "hotel", "hostel", "guest_house"]
        return any(t in types for t in lodging_types)
    
    return True

def _score_place(place, category):
    """
    Score a place based on quality indicators.
    Higher score = better recommendation.
    """
    score = 0
    
    # 1. Rating (0-50 points) - Most important
    rating = place.get("rating", 0)
    score += rating * 10  # 5.0 rating = 50 points
    
    # 2. Popularity based on review count (0-30 points)
    total_ratings = place.get("user_ratings_total", 0)
    if total_ratings >= 1000:
        score += 30
    elif total_ratings >= 500:
        score += 25
    elif total_ratings >= 200:
        score += 20
    elif total_ratings >= 100:
        score += 15
    elif total_ratings >= 50:
        score += 10
    elif total_ratings >= 20:
        score += 5
    
    # 3. Price level preference (0-20 points)
    price_level = place.get("price_level")
    if price_level:
        if category == "restaurant":
            # For restaurants: prefer affordable to moderate (1-2)
            if price_level in [1, 2]:
                score += 20
            elif price_level == 3:
                score += 15
            elif price_level == 4:
                score += 10
        else:  # hotel
            # For hotels: prefer moderate to upscale (2-3)
            if price_level in [2, 3]:
                score += 20
            elif price_level == 1:
                score += 15
            elif price_level == 4:
                score += 15
    
    # 4. Bonus for high rating + many reviews (quality + popularity)
    if rating >= 4.5 and total_ratings >= 200:
        score += 10
    
    return score

def find_nearby_places(
    location: str, 
    category: str, 
    radius: int = 2000, 
    max_results: int = 5, 
    min_rating: float = 4.0
) -> PlaceResponse:
    """
    Find nearby places with smart filtering and ranking.
    Returns the best quality places based on rating, popularity, and relevance.
    
    Args:
        location: City or place name
        category: 'restaurant' or 'hotel'
        radius: Search radius in meters
        max_results: Maximum number of results to return
        min_rating: Minimum rating threshold
    """
    logger.info(f"Searching for {category}s near {location} (radius={radius}m, max={max_results}, min_rating={min_rating})")
    
    # Geocode the location
    geocode = gmaps.geocode(location)
    if not geocode:
        raise ValueError(f"Invalid location: {location}")
    latlng = geocode[0]["geometry"]["location"]

    # Search for nearby places
    # Get more results initially to have better selection after filtering
    results = gmaps.places_nearby(
        location=(latlng["lat"], latlng["lng"]),
        radius=radius,
        type=category
    )
    
    # Process and score places
    candidates = []
    for r in results.get("results", []):
        # Basic rating filter
        if r.get("rating", 0) < min_rating:
            continue
        
        # Validate it's actually the right type of place
        if not _is_valid_place(r, category):
            logger.debug(f"Filtered out: {r.get('name')} (wrong type)")
            continue
        
        # Get additional details if available
        place_id = r.get("place_id")
        
        # Try to get more details for better scoring
        try:
            details = gmaps.place(place_id, fields=[
                "name", "rating", "user_ratings_total", 
                "price_level", "formatted_address", "types", "vicinity"
            ])
            place_data = details.get("result", r)
        except:
            place_data = r
        
        # Calculate quality score
        score = _score_place(place_data, category)
        
        candidates.append({
            "data": place_data,
            "score": score,
            "place_id": place_id
        })
    
    # Sort by score (highest first)
    candidates.sort(key=lambda x: x["score"], reverse=True)
    
    # Take top N results
    top_candidates = candidates[:max_results]
    
    logger.info(f"Filtered {len(candidates)} candidates, returning top {len(top_candidates)}")
    
    # Format results
    places = []
    for candidate in top_candidates:
        r = candidate["data"]
        
        place_info = PlaceInfo(
            name=r.get("name", "Unknown"),
            rating=r.get("rating", 0.0),
            address=r.get("formatted_address") or r.get("vicinity", "N/A"),
            maps_url=f"https://www.google.com/maps/place/?q=place_id:{candidate['place_id']}",
            price_level=_get_price_level_text(r.get("price_level")),
            total_ratings=r.get("user_ratings_total", 0)
        )
        places.append(place_info)
    
    logger.success(f"Returning {len(places)} high-quality {category}s")
    return PlaceResponse(places=places)