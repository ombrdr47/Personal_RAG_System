# main.py - Entry point for your FastAPI application

from src.api import app

# This allows uvicorn to find the FastAPI app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)