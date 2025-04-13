import os
from fastapi import FastAPI, Query, Request, HTTPException, Path
from typing import Optional, Dict, List, Any
import uvicorn
from finvizfinance.quote import finvizfinance
from finvizfinance.screener.overview import Overview
from finvizfinance.screener.valuation import Valuation
from finvizfinance.screener.financial import Financial
from finvizfinance.news import News
from finvizfinance.insider import Insider
from finvizfinance.future import Future
from custom_calendar import CustomCalendar
from pydantic import BaseModel, Field
import pandas as pd
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import time
import asyncio
from cachetools import TTLCache, cached
from functools import wraps
from api_config import custom_openapi, setup_middleware, RATE_LIMIT_REQUESTS
from rate_limiter import RateLimiter
from datetime import datetime

# Initialize caches with TTL (Time To Live)
stock_cache = TTLCache(maxsize=100, ttl=300)  # 5 minutes cache for stock data
screener_cache = TTLCache(maxsize=50, ttl=600)  # 10 minutes cache for screener data
calendar_cache = TTLCache(maxsize=10, ttl=1800)  # 30 minutes cache for calendar data

app = FastAPI(
    title="Market Data API",
    description="A comprehensive API for accessing market data, economic calendar, and stock information",
    version="1.0.0"
)

# Setup middleware
setup_middleware(app)
app.add_middleware(RateLimiter, requests_per_minute=RATE_LIMIT_REQUESTS)

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

# Response Models
class StockResponse(BaseModel):
    """Response model for stock information"""
    fundamentals: Dict[str, Any]
    description: Dict[str, Any]
    ratings: Dict[str, Any]
    news: Dict[str, Any]
    insider_trading: Dict[str, Any]
    signal: Dict[str, Any]
    full_info: Dict[str, Any]

class ScreenerFilters(BaseModel):
    """Filters for stock screener"""
    Exchange: Optional[str] = None
    Sector: Optional[str] = None
    Industry: Optional[str] = None
    Country: Optional[str] = None

class CalendarFilter(BaseModel):
    """Filters for economic calendar"""
    date: Optional[str] = None
    impact: Optional[str] = None
    release: Optional[str] = None

class CalendarEvent(BaseModel):
    Date: str
    Time: str
    Datetime: str
    Release: str
    Impact: str
    For: str
    Actual: Optional[str]
    Expected: Optional[str]
    Prior: Optional[str]

class CalendarResponse(BaseModel):
    events: List[CalendarEvent]
    total_events: int
    available_dates: List[str]

class CalendarSummary(BaseModel):
    overall_summary: Dict[str, int]
    today_summary: Dict[str, int]
    total_events: int
    today_events: int

def async_cached(cache):
    """Decorator for caching async function results"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            key = str(args) + str(kwargs)
            if key in cache:
                return cache[key]
            result = await func(*args, **kwargs)
            cache[key] = result
            return result
        return wrapper
    return decorator

@app.get("/")
async def root():
    return {"message": "Welcome to Market Data API"}

@app.get("/stock/{symbol}", response_model=StockResponse, tags=["Stocks"])
async def get_stock_info(
    symbol: str = Path(..., description="Stock symbol (e.g., AAPL, MSFT)"),
    screener: Optional[str] = Query(None, description="Screener to use (e.g., 'overview', 'all')")
):
    """
    Get comprehensive stock information including fundamentals, description, ratings, news, and more.
    
    Parameters:
    - symbol: Stock symbol (e.g., AAPL, MSFT)
    - screener: Optional screener to use (e.g., 'overview', 'all')
    """
    try:
        # Initialize stock object
        stock = finvizfinance(symbol)
        
        async def gather_data():
            tasks = [
                asyncio.to_thread(lambda: stock.TickerFundament()),
                asyncio.to_thread(lambda: stock.TickerDescription()),
                asyncio.to_thread(lambda: stock.TickerRatings()),
                asyncio.to_thread(lambda: stock.TickerNews()),
                asyncio.to_thread(lambda: stock.TickerInsider()),
                asyncio.to_thread(lambda: stock.TickerSignal()),
                asyncio.to_thread(lambda: stock.TickerFullInfo())
            ]
            return await asyncio.gather(*tasks)
        
        # Gather all data concurrently
        fundamentals, description, ratings, news, insider, signal, full_info = await gather_data()
        
        # Convert DataFrames to dictionaries
        def safe_df_to_dict(df):
            if df is None or df.empty:
                return {}
            return df.to_dict('records')
        
        return StockResponse(
            fundamentals=safe_df_to_dict(fundamentals),
            description=safe_df_to_dict(description),
            ratings=safe_df_to_dict(ratings),
            news=safe_df_to_dict(news),
            insider_trading=safe_df_to_dict(insider),
            signal=safe_df_to_dict(signal),
            full_info=safe_df_to_dict(full_info)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/calendar", response_model=CalendarResponse, tags=["Calendar"])
async def get_calendar():
    """
    Get the economic calendar data.
    
    Returns a list of economic events with their details including date, time, release name, and impact.
    """
    try:
        calendar = CustomCalendar()
        df = await calendar.calendar()
        
        if df.empty:
            return JSONResponse(
                status_code=404,
                content={"message": "No calendar data available"}
            )
        
        events = df.to_dict('records')
        available_dates = sorted(df['Date'].unique().tolist())
        
        return {
            "events": events,
            "total_events": len(events),
            "available_dates": available_dates
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/calendar/filter", response_model=CalendarResponse, tags=["Calendar"])
async def filter_calendar(filter: CalendarFilter):
    """
    Filter economic calendar events by date, impact, or release name.
    
    Parameters:
    - date: Filter by date (e.g., "Mon Apr 07")
    - impact: Filter by impact level (1-3)
    - release: Filter by release name (e.g., "CPI")
    """
    try:
        calendar = CustomCalendar()
        df = await calendar.calendar()
        
        if df.empty:
            return JSONResponse(
                status_code=404,
                content={"message": "No calendar data available"}
            )
        
        # Apply filters
        if filter.date:
            df = df[df['Date'] == filter.date]
        if filter.impact:
            df = df[df['Impact'] == filter.impact]
        if filter.release:
            df = df[df['Release'].str.contains(filter.release, case=False, na=False)]
        
        events = df.to_dict('records')
        available_dates = sorted(df['Date'].unique().tolist())
        
        return {
            "events": events,
            "total_events": len(events),
            "available_dates": available_dates
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/calendar/summary", response_model=CalendarSummary, tags=["Calendar"])
async def get_calendar_summary():
    """
    Get a summary of economic calendar events grouped by impact level.
    
    Returns summary statistics including:
    - Overall summary of events by impact level
    - Today's events summary
    - Total number of events
    - Number of events today
    """
    try:
        calendar = CustomCalendar()
        df = await calendar.calendar()
        
        if df.empty:
            return JSONResponse(
                status_code=404,
                content={"message": "No calendar data available"}
            )
        
        # Get today's date in the format used in the calendar
        today = datetime.now().strftime("%a %b %d")
        
        # Calculate summaries
        overall_summary = df['Impact'].value_counts().to_dict()
        today_df = df[df['Date'] == today]
        today_summary = today_df['Impact'].value_counts().to_dict()
        
        return {
            "overall_summary": overall_summary,
            "today_summary": today_summary,
            "total_events": len(df),
            "today_events": len(today_df)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))  # Default to 8000 if PORT is not set
    uvicorn.run("api:app", host="0.0.0.0", port=port, reload=True) 