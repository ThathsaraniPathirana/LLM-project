"""
Helper functions for MCP integration with the RAG orchestrator.
"""
import httpx
import json
from typing import Optional, Dict, List

MCP_URL = "http://localhost:9000/tools/find_nearby_places"

async def fetch_places(
    location: str, 
    category: str, 
    radius: int = 2000, 
    max_results: int = 5,
    min_rating: float = 4.0
) -> Optional[Dict]:
    """
    Fetch places from MCP server.
    
    Args:
        location: Location to search near
        category: 'restaurant' or 'hotel'
        radius: Search radius in meters
        max_results: Maximum number of results
        min_rating: Minimum rating threshold
    
    Returns:
        Dict with places data or None if failed
    """
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            response = await client.post(MCP_URL, json={
                "location": location,
                "category": category,
                "radius": radius,
                "max_results": max_results,
                "min_rating": min_rating
            })
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as e:
            print(f"HTTP error fetching places: {e}")
            return None
        except Exception as e:
            print(f"Error fetching places: {e}")
            return None

def format_places_markdown(places_data: Dict, category: str) -> str:
    """
    Format places data as markdown for display.
    
    Args:
        places_data: Response from MCP server
        category: 'restaurant' or 'hotel'
    
    Returns:
        Formatted markdown string
    """
    if not places_data or not places_data.get("places"):
        return f"_No {category}s found matching your criteria._"
    
    places = places_data["places"]
    location = places_data.get("location", "this area")
    
    markdown = f"\n\n### 🎯 Top {category.title()}s near {location}\n\n"
    
    for i, place in enumerate(places, 1):
        rating_stars = "⭐" * int(place['rating'])
        reviews = place.get('total_ratings', 0)
        
        markdown += f"""**{i}. {place['name']}**  
{rating_stars} {place['rating']}/5 ({reviews:,} reviews)  
💰 {place.get('price_level', 'Price not available')}  
📍 {place['address']}  
[View on Google Maps]({place['maps_url']})

"""
    
    return markdown

def format_places_for_llm_context(places_data: Dict, category: str) -> str:
    """
    Format places data as context for LLM prompt.
    More detailed than markdown for better LLM understanding.
    
    Args:
        places_data: Response from MCP server
        category: 'restaurant' or 'hotel'
    
    Returns:
        Formatted context string
    """
    if not places_data or not places_data.get("places"):
        return ""
    
    places = places_data["places"]
    location = places_data.get("location", "this area")
    
    context = f"\n\n### Live {category.title()} Data from Google Places API:\n"
    context += f"Location: {location}\n"
    context += f"Total results: {places_data.get('total_found', len(places))}\n\n"
    
    for i, place in enumerate(places, 1):
        context += f"""Place {i}:
- Name: {place['name']}
- Rating: {place['rating']}/5.0
- Total Reviews: {place.get('total_ratings', 0)}
- Price Level: {place.get('price_level', 'Not available')}
- Address: {place['address']}
- Google Maps: {place['maps_url']}

"""
    
    return context

def create_places_summary(places_data: Dict, category: str) -> str:
    """
    Create a brief summary of the places for quick reference.
    
    Args:
        places_data: Response from MCP server
        category: 'restaurant' or 'hotel'
    
    Returns:
        Brief summary string
    """
    if not places_data or not places_data.get("places"):
        return f"No {category}s found."
    
    places = places_data["places"]
    count = len(places)
    avg_rating = sum(p['rating'] for p in places) / count if count > 0 else 0
    
    price_levels = [p.get('price_level', '') for p in places]
    has_budget = any('Budget' in p for p in price_levels)
    has_luxury = any('Luxury' in p for p in price_levels)
    
    summary = f"Found {count} highly-rated {category}s (avg {avg_rating:.1f}★)"
    
    if has_budget and has_luxury:
        summary += " ranging from budget-friendly to luxury options"
    elif has_budget:
        summary += " with budget-friendly options"
    elif has_luxury:
        summary += " including upscale options"
    
    return summary + "."

def extract_location_from_query(query: str) -> Optional[str]:
    """
    Simple extraction of Swedish locations from query.
    This is a fallback if LLM intent detection fails.
    
    Args:
        query: User query string
    
    Returns:
        Extracted location or None
    """
    # Common Swedish cities and areas
    locations = [
        "Stockholm", "Gamla Stan", "Södermalm", "Östermalm", "Vasastan",
        "Gothenburg", "Göteborg", "Malmö", "Uppsala", "Västerås",
        "Örebro", "Linköping", "Helsingborg", "Jönköping", "Norrköping",
        "Lund", "Umeå", "Gävle", "Borås", "Eskilstuna", "Kiruna",
        "Visby", "Dalarna", "Lapland", "Lappland"
    ]
    
    query_lower = query.lower()
    for loc in locations:
        if loc.lower() in query_lower:
            return loc
    
    return None