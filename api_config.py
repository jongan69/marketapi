from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
from typing import Dict, Any

# API Metadata
API_TITLE = "Market Data API"
API_VERSION = "1.0.0"
API_DESCRIPTION = """
A comprehensive API for accessing market data, economic calendar, and stock information.
Provides real-time and historical data from various financial sources.
"""

# Rate Limiting Settings
RATE_LIMIT_REQUESTS = 100  # requests per minute
RATE_LIMIT_WINDOW = 60  # seconds

def custom_openapi(app: FastAPI) -> Dict[str, Any]:
    """Generate custom OpenAPI schema for RapidAPI"""
    if app.openapi_schema:
        return app.openapi_schema

    openapi_schema = get_openapi(
        title=API_TITLE,
        version=API_VERSION,
        description=API_DESCRIPTION,
        routes=app.routes,
    )

    # Add RapidAPI security scheme
    openapi_schema["components"]["securitySchemes"] = {
        "RapidAPIKey": {
            "type": "apiKey",
            "in": "header",
            "name": "X-RapidAPI-Key"
        }
    }

    # Add security requirement to all endpoints
    openapi_schema["security"] = [{"RapidAPIKey": []}]

    app.openapi_schema = openapi_schema
    return app.openapi_schema

def setup_middleware(app: FastAPI) -> None:
    """Setup CORS and other middleware"""
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    ) 