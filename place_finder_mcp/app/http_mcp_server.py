"""
Simple HTTP wrapper for the places finder.
Returns high-quality, well-filtered results for RAG orchestrator.
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import httpx
import os
from dotenv import load_dotenv

load_dotenv()
API_URL = os.getenv("MCP_SERVER_URL", "http://localhost:8080/find_nearby")

app = FastAPI(title="Places Finder for RAG")

class ToolRequest(BaseModel):
    location: str
    category: str
    radius: int = 2000
    max_results: int = 5
    min_rating: float = 4.0

class PlaceInfo(BaseModel):
    name: str
    rating: float
    address: str
    maps_url: str
    price_level: str
    total_ratings: int

class ToolResponse(BaseModel):
    places: list[PlaceInfo]
    location: str
    category: str
    total_found: int

@app.post("/tools/find_nearby_places", response_model=ToolResponse)
async def find_nearby_places(req: ToolRequest):
    """
    Find nearby places with smart filtering.
    Returns top-quality results ready for RAG processing.
    """
    print(f"[HTTP] {req.category.title()}s near {req.location} (radius={req.radius}m, max={req.max_results})")
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            resp = await client.post(API_URL, json={
                "location": req.location,
                "category": req.category,
                "radius": req.radius,
                "max_results": req.max_results,
                "min_rating": req.min_rating
            })
            resp.raise_for_status()
            result = resp.json()
            
            places = result.get("places", [])
            print(f"[HTTP] Returning {len(places)} high-quality {req.category}s")
            
            return ToolResponse(
                places=places,
                location=req.location,
                category=req.category,
                total_found=len(places)
            )
            
        except httpx.HTTPError as e:
            print(f"[HTTP] Error: {e}")
            raise HTTPException(status_code=500, detail=str(e))
        except Exception as e:
            print(f"[HTTP] Error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok", "service": "Places Finder for RAG"}

if __name__ == "__main__":
    import uvicorn
    print("Starting Places Finder HTTP service...")
    print("Listening on http://0.0.0.0:9000")
    print("Endpoint: POST /tools/find_nearby_places")
    print("\nReady for RAG orchestrator integration!\n")
    uvicorn.run(app, host="0.0.0.0", port=9000)