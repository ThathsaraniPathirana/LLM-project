"""
Test the HTTP wrapper - simulates RAG orchestrator calls.
"""
import httpx
import asyncio
import json

async def test_restaurants():
    """Test restaurant search with quality filtering."""
    url = "http://localhost:9000/tools/find_nearby_places"
    payload = {
        "location": "Kungsträdgården, Stockholm",
        "category": "restaurant",
        "radius": 1500,
        "max_results": 5,
        "min_rating": 4.0
    }
    
    print("=" * 70)
    print("Test 1: Restaurant Search (Quality Filtered)")
    print("=" * 70)
    await make_request(url, payload)

async def test_hotels():
    """Test hotel search."""
    url = "http://localhost:9000/tools/find_nearby_places"
    payload = {
        "location": "Stockholm Central Station",
        "category": "hotel",
        "radius": 2000,
        "max_results": 5,
        "min_rating": 4.0
    }
    
    print("\n" + "=" * 70)
    print("Test 2: Hotel Search")
    print("=" * 70)
    await make_request(url, payload)

async def test_budget_restaurants():
    """Test with different parameters."""
    url = "http://localhost:9000/tools/find_nearby_places"
    payload = {
        "location": "Gamla Stan, Stockholm",
        "category": "restaurant",
        "radius": 1000,
        "max_results": 3,
        "min_rating": 4.3
    }
    
    print("\n" + "=" * 70)
    print("Test 3: High-Rated Restaurants (min 4.3★)")
    print("=" * 70)
    await make_request(url, payload)

async def make_request(url, payload):
    """Make HTTP request and display results."""
    print(f"\nRequest:")
    print(f"   Location: {payload['location']}")
    print(f"   Category: {payload['category']}")
    print(f"   Radius: {payload['radius']}m")
    print(f"   Max results: {payload['max_results']}")
    print(f"   Min rating: {payload['min_rating']}★")
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            response = await client.post(url, json=payload)
            response.raise_for_status()
            
            result = response.json()
            
            print(f"\nStatus: {response.status_code}")
            print(f"Found: {result.get('total_found', 0)} places")
            
            if result.get("places"):
                print(f"\n{'#':<3} {'Name':<35} {'Rating':<8} {'Reviews':<10} {'Price'}")
                print("-" * 70)
                
                for i, place in enumerate(result["places"], 1):
                    name = place['name'][:32] + "..." if len(place['name']) > 32 else place['name']
                    rating = f"{place['rating']}★"
                    reviews = f"({place.get('total_ratings', 0)})"
                    price = place.get('price_level', 'N/A')
                    
                    print(f"{i:<3} {name:<35} {rating:<8} {reviews:<10} {price}")
                
                print("\nSample addresses:")
                for i, place in enumerate(result["places"][:3], 1):
                    print(f"   {i}. {place['address']}")
                    
            else:
                print("\nNo places found matching criteria")
            
        except httpx.HTTPError as e:
            print(f"\nHTTP Error: {e}")
            if hasattr(e, 'response'):
                print(f"Response: {e.response.text}")
        except Exception as e:
            print(f"\nError: {e}")
            import traceback
            traceback.print_exc()

async def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("Places Finder Quality Tests")
    print("=" * 70)
    print("\nThese tests show high-quality, well-filtered results ready for RAG.\n")
    
    await test_restaurants()
    await test_hotels()
    await test_budget_restaurants()
    
    print("\n" + "=" * 70)
    print(" All tests completed!")
    print("=" * 70)
    print("\nFor RAG integration, use:")
    print("   POST http://localhost:9000/tools/find_nearby_places")
    print("   Your RAG orchestrator can now use these high-quality results!\n")

if __name__ == "__main__":
    asyncio.run(main())