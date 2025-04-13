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
from fomc_calendar import FOMCCalendar
import aiohttp
from bs4 import BeautifulSoup
import re

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

class HealthCheck(BaseModel):
    status: str
    version: str
    timestamp: str
    uptime: float
    services: Dict[str, str]

# FOMC Response Models
class FOMCMeeting(BaseModel):
    """Model for FOMC meeting data"""
    Date: str
    Is_Projection: bool
    Has_Press_Conference: bool
    Statement_Link: Optional[str] = None
    Minutes_Link: Optional[str] = None
    Minutes_Text: Optional[str] = None
    Minutes_Summary: Optional[str] = None

class FOMCLatestResponse(BaseModel):
    """Response model for the latest FOMC meeting endpoint"""
    meeting: Optional[FOMCMeeting] = None
    next_meeting: Optional[FOMCMeeting] = None
    status: str
    error: Optional[str] = None

# Add this near the start of your file, after the imports
start_time = time.time()

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

@app.post("/screener/overview", tags=["Screener"])
async def get_screener_overview(filters: ScreenerFilters):
    """Get stock screener overview with specified filters"""
    try:
        # Create a cache key from the filters
        cache_key = f"overview_{hash(str(filters.dict(exclude_none=True)))}"
        
        # Check cache first
        if cache_key in screener_cache:
            return screener_cache[cache_key]
            
        overview = Overview()
        if filters:
            overview.set_filter(filters_dict=filters.dict(exclude_none=True))
        df = overview.screener_view()
        # Replace NaN values with None before converting to JSON
        df = df.replace({pd.NA: None, pd.NaT: None, float('nan'): None})
        result = df.to_dict('records')
        
        # Store in cache
        screener_cache[cache_key] = result
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/screener/valuation", tags=["Screener"])
async def get_screener_valuation(filters: ScreenerFilters):
    """Get stock screener valuation metrics with specified filters"""
    try:
        # Create a cache key from the filters
        cache_key = f"valuation_{hash(str(filters.dict(exclude_none=True)))}"
        
        # Check cache first
        if cache_key in screener_cache:
            return screener_cache[cache_key]
            
        valuation = Valuation()
        if filters:
            valuation.set_filter(filters_dict=filters.dict(exclude_none=True))
        df = valuation.screener_view()
        # Replace NaN values with None before converting to JSON
        df = df.replace({pd.NA: None, pd.NaT: None, float('nan'): None})
        result = df.to_dict('records')
        
        # Store in cache
        screener_cache[cache_key] = result
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/screener/financial", tags=["Screener"])
async def get_screener_financial(filters: ScreenerFilters):
    """Get stock screener financial metrics with specified filters"""
    try:
        # Create a cache key from the filters
        cache_key = f"financial_{hash(str(filters.dict(exclude_none=True)))}"
        
        # Check cache first
        if cache_key in screener_cache:
            return screener_cache[cache_key]
            
        financial = Financial()
        if filters:
            financial.set_filter(filters_dict=filters.dict(exclude_none=True))
        df = financial.screener_view()
        # Replace NaN values with None before converting to JSON
        df = df.replace({pd.NA: None, pd.NaT: None, float('nan'): None})
        result = df.to_dict('records')
        
        # Store in cache
        screener_cache[cache_key] = result
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/news", tags=["News"])
async def get_news():
    """Get latest financial news and blog posts"""
    try:
        fnews = News()
        all_news = fnews.get_news()
        return {
            "news": all_news['news'].to_dict('records'),
            "blogs": all_news['blogs'].to_dict('records')
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/insider", tags=["Insider Trading"])
async def get_insider_trading(option: str = Query("top owner trade", description="Type of insider trading data")):
    """Get insider trading information"""
    try:
        finsider = Insider(option=option)
        df = finsider.get_insider()
        # Replace NaN values with None before converting to JSON
        df = df.replace({pd.NA: None, pd.NaT: None, float('nan'): None})
        return df.to_dict('records')
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/futures", tags=["Futures"])
async def get_futures():
    """Get futures market performance"""
    try:
        future = Future()
        return future.performance().to_dict('records')
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

@app.get("/health", tags=["System"], response_model=HealthCheck)
async def health_check():
    """
    Health check endpoint for monitoring API status.
    Returns the current status of the API and its services.
    """
    try:
        # Check if we can access FinViz data
        stock = finvizfinance("AAPL")
        stock_status = "healthy"
    except Exception as e:
        stock_status = f"unhealthy: {str(e)}"

    try:
        # Check if we can access calendar data
        calendar = CustomCalendar()
        calendar_status = "healthy"
    except Exception as e:
        calendar_status = f"unhealthy: {str(e)}"

    return HealthCheck(
        status="healthy",
        version="1.0.0",
        timestamp=datetime.now().isoformat(),
        uptime=time.time() - start_time,
        services={
            "stock_data": stock_status,
            "calendar_data": calendar_status
        }
    )

@app.get("/fomc/calendar", tags=["FOMC"])
async def get_fomc_calendar():
    """Get FOMC meeting calendar data.
    
    Returns:
        dict: A dictionary containing:
            - past_meetings: List of past FOMC meetings
            - future_meetings: List of future FOMC meetings
            - total_meetings: Total number of meetings
            - years: List of unique years
    """
    try:
        fomc = FOMCCalendar()
        past_meetings_df, future_meetings_df = await fomc.calendar()
        
        # Convert DataFrames to dictionaries
        past_meetings = past_meetings_df.to_dict(orient='records')
        future_meetings = future_meetings_df.to_dict(orient='records')
        
        # Get unique years
        all_years = sorted(list(set([meeting['Year'] for meeting in past_meetings + future_meetings])))
        
        return {
            "past_meetings": past_meetings,
            "future_meetings": future_meetings,
            "total_meetings": len(past_meetings) + len(future_meetings),
            "years": all_years
        }
    except Exception as e:
        print(f"Error in FOMC calendar endpoint: {e}")
        return {
            "past_meetings": [],
            "future_meetings": [],
            "total_meetings": 0,
            "years": []
        }

def parse_date(date_str: str, year: int) -> pd.Timestamp:
    try:
        # Split on hyphen for multi-day meetings and take first day
        first_day = date_str.split('-')[0].strip()
        # Convert month name to number
        month_map = {
            'January': 1, 'February': 2, 'March': 3, 'April': 4,
            'May': 5, 'June': 6, 'July': 7, 'August': 8,
            'September': 9, 'October': 10, 'November': 11, 'December': 12
        }
        # Extract month and day
        parts = first_day.split()
        if len(parts) != 2:
            print(f"Invalid date format: {first_day}")
            return None
        month, day = parts[0], int(parts[1])
        month_num = month_map.get(month)
        if not month_num:
            print(f"Invalid month: {month}")
            return None
        
        # Create datetime object - use noon (12:00) for all past dates to ensure consistent ordering
        dt = datetime(year=year, month=month_num, day=day, hour=12, minute=0, second=0)
        now = datetime.now()
        
        # Only adjust future dates
        if dt.date() > now.date():
            dt = datetime(year=year, month=month_num, day=day, hour=0, minute=0, second=0)
            
        return pd.Timestamp(dt)
    except Exception as e:
        print(f"Error parsing date {date_str}: {e}")
        return None

@app.get("/fomc/latest", response_model=FOMCLatestResponse, tags=["FOMC"])
async def get_latest_fomc_meeting():
    """
    Get detailed information about the latest and next FOMC meetings.
    """
    try:
        calendar = FOMCCalendar(openai_api_key=os.environ.get("OPENAI_API_KEY"))
        past_meetings, future_meetings = await calendar.calendar()
        
        # Get the latest non-projection meeting
        latest_meeting = None
        for _, meeting in past_meetings.iterrows():
            if not meeting['Is_Projection']:
                latest_meeting = meeting.copy()
                # Convert Timestamp to string
                latest_meeting['Date'] = latest_meeting['Date'].strftime('%Y-%m-%d')
                break
        
        # Get the next meeting
        next_meeting = None
        if not future_meetings.empty:
            next_meeting = future_meetings.iloc[0].copy()
            # Convert Timestamp to string
            next_meeting['Date'] = next_meeting['Date'].strftime('%Y-%m-%d')
            # For future meetings, set links to None
            next_meeting['Statement_Link'] = None
            next_meeting['Minutes_Link'] = None
            next_meeting['Minutes_Text'] = None
            next_meeting['Minutes_Summary'] = None
        
        return FOMCLatestResponse(
            meeting=FOMCMeeting(**latest_meeting.to_dict()) if latest_meeting is not None else None,
            next_meeting=FOMCMeeting(**next_meeting.to_dict()) if next_meeting is not None else None,
            status="success"
        )
    except Exception as e:
        print(f"Error in get_latest_fomc_meeting: {e}")
        return FOMCLatestResponse(
            meeting=None,
            next_meeting=None,
            status="error",
            error=str(e)
        )

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))  # Default to 8000 if PORT is not set
    uvicorn.run("api:app", host="0.0.0.0", port=port, reload=True) 