"""
Startup script for the MCP server.
This uses FastMCP's built-in server which handles all SSE transport automatically.
"""
from app.mcp_wrapper import mcp
import os

if __name__ == "__main__":
    # Set environment variables for FastMCP
    os.environ.setdefault("MCP_HOST", "0.0.0.0")
    os.environ.setdefault("MCP_PORT", "8000")
    
    print("Starting Places Finder MCP server...")
    print("Listening on http://0.0.0.0:8000")
    print("SSE endpoint: http://0.0.0.0:8000/sse")
    print("\nPress CTRL+C to stop\n")
    
    # FastMCP.run() handles everything automatically
    mcp.run(transport="sse")