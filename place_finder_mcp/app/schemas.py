from typing import List, Optional
from pydantic import BaseModel, Field

class PlaceRequest(BaseModel):
    location: str = Field(..., description="City or place name to search nearby from.")
    category: str = Field(..., pattern="^(restaurant|hotel)$", description="Type of place to find.")
    radius: int = Field(2000, description="Search radius in meters (default 2000).")
    max_results: int = Field(5, ge=1, le=20, description="Maximum number of results (default 5).")
    min_rating: float = Field(4.0, ge=0, le=5, description="Minimum rating threshold (default 4.0).")

class PlaceInfo(BaseModel):
    name: str
    rating: float
    address: str
    maps_url: str
    price_level: str = "Price not available"
    total_ratings: int = 0

class PlaceResponse(BaseModel):
    places: List[PlaceInfo]
    
    class Config:
        json_schema_extra = {
            "example": {
                "places": [
                    {
                        "name": "Restaurang Tradition",
                        "rating": 4.5,
                        "address": "Östermalm 12, Stockholm",
                        "maps_url": "https://...",
                        "price_level": "Moderate (€€)",
                        "total_ratings": 523
                    }
                ]
            }
        }