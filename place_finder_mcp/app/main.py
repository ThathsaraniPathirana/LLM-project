from fastapi import FastAPI, HTTPException
from .schemas import PlaceRequest, PlaceResponse
from .google_places import find_nearby_places
from loguru import logger
import os
from dotenv import load_dotenv

load_dotenv()
app = FastAPI(title="Places Finder MCP Server", version="1.0")

@app.post("/find_nearby", response_model=PlaceResponse)
def find_nearby(req: PlaceRequest):
    try:
        logger.info(f"Request received: {req.location}, {req.category}")
        return find_nearby_places(req.location, req.category, req.radius)
    except Exception as e:
        logger.error(f"Error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
