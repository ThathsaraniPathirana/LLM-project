"""
Simple MCP server using the official MCP SDK - no FastMCP complications.
This is the most straightforward way to create an MCP server.
"""
import asyncio
import httpx
import os
from dotenv import load_dotenv
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

load_dotenv()
API_URL = os.getenv("MCP_SERVER_URL", "http://localhost:8080/find_nearby")

# Create MCP server
app = Server("places-finder-mcp")

@app.list_tools()
async def list_tools() -> list[Tool]:
    """List available tools."""
    return [
        Tool(
            name="find_nearby_places",
            description="Find nearby places using Google Places API",
            inputSchema={
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City or place name to search nearby from (e.g., 'Gamla Stan, Stockholm')"
                    },
                    "category": {
                        "type": "string",
                        "description": "Type of place to find (restaurant or hotel)",
                        "enum": ["restaurant", "hotel"]
                    },
                    "radius": {
                        "type": "integer",
                        "description": "Search radius in meters (default 2000)",
                        "default": 2000
                    }
                },
                "required": ["location", "category"]
            }
        )
    ]

@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Handle tool calls."""
    if name != "find_nearby_places":
        raise ValueError(f"Unknown tool: {name}")
    
    location = arguments.get("location")
    category = arguments.get("category")
    radius = arguments.get("radius", 2000)
    
    print(f"[MCP TOOL] find_nearby_places({location}, {category}, {radius})")
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            resp = await client.post(API_URL, json={
                "location": location,
                "category": category,
                "radius": radius
            })
            resp.raise_for_status()
            result = resp.json()
            
            print(f"Got {len(result.get('places', []))} places from backend")
            
            # Return as TextContent
            import json
            return [TextContent(
                type="text",
                text=json.dumps(result, ensure_ascii=False, indent=2)
            )]
            
        except httpx.HTTPError as e:
            print(f"HTTP error calling backend: {e}")
            import json
            return [TextContent(
                type="text",
                text=json.dumps({
                    "error": f"Backend error: {str(e)}",
                    "places": []
                })
            )]
        except Exception as e:
            print(f"Error: {e}")
            import json
            return [TextContent(
                type="text",
                text=json.dumps({
                    "error": str(e),
                    "places": []
                })
            )]

async def main():
    """Run the server using stdio transport."""
    print("Starting MCP server with stdio transport...")
    async with stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            app.create_initialization_options()
        )

if __name__ == "__main__":
    asyncio.run(main())