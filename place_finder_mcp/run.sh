#!/bin/bash

echo "Starting services..."

# Start the FastAPI backend (Google Places API wrapper)
echo "1. Starting FastAPI backend (port 8080)..."
python -m uvicorn app.main:app --host 0.0.0.0 --port 8080 &
BACKEND_PID=$!

# Wait for backend to be ready
sleep 2

# Start the HTTP wrapper for RAG orchestrator
echo "2. Starting HTTP wrapper for RAG (port 9000)..."
python app/http_mcp_server.py &
HTTP_PID=$!

echo ""
echo "All services started!"
echo "   - Backend API: http://localhost:8080"
echo "   - RAG Endpoint: http://localhost:9000/tools/find_nearby_places"
echo ""
echo "Test with: python test_http_wrapper.py"
echo "Press CTRL+C to stop all services"
echo ""

# Cleanup function
cleanup() {
    echo ""
    echo "Shutting down services..."
    kill $BACKEND_PID 2>/dev/null
    kill $HTTP_PID 2>/dev/null
    exit
}

# Trap signals
trap cleanup SIGINT SIGTERM

# Wait for processes
wait