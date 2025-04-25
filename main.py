import os
from fastapi import FastAPI
import uvicorn
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.middleware.cors import CORSMiddleware
from api_config import custom_openapi, setup_middleware, RATE_LIMIT_REQUESTS
from rate_limiter import RateLimiter
from logger import LoggingMiddleware, log_endpoint_access
import time

# Import routers
from routes.stocks import router as stocks_router
from routes.screener import router as screener_router
from routes.calendar import router as calendar_router
from routes.maxpain import router as maxpain_router
from routes.health import router as health_router
from routes.insider import router as insider_router
from routes.news import router as news_router
from routes.futures import router as futures_router

app = FastAPI(
    title="Market Data API",
    description="A comprehensive API for accessing market data, economic calendar, and stock information",
    version="1.0.0"
)

# Setup middleware
setup_middleware(app)
app.add_middleware(RateLimiter, requests_per_minute=RATE_LIMIT_REQUESTS)
app.add_middleware(LoggingMiddleware)  # Add logging middleware

# Custom OpenAPI schema
app.openapi = lambda: custom_openapi(app)

# Add GZip compression
app.add_middleware(GZipMiddleware, minimum_size=1000)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    log_endpoint_access("/")
    return {"message": "Welcome to Market Data API"}

# Add routers
app.include_router(stocks_router)
app.include_router(screener_router)
app.include_router(calendar_router)
app.include_router(maxpain_router)
app.include_router(health_router)
app.include_router(insider_router)
app.include_router(news_router)
app.include_router(futures_router)

# Store start time for uptime calculation
app.state.start_time = time.time()

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))  # Default to 8000 if PORT is not set
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True) 