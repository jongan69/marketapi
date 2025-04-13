from fastapi import FastAPI, Query, Request
from typing import Optional, Dict, List
import uvicorn
from finvizfinance.quote import finvizfinance
from finvizfinance.screener.overview import Overview
from finvizfinance.screener.valuation import Valuation
from finvizfinance.screener.financial import Financial
from finvizfinance.news import News
from finvizfinance.insider import Insider
from finvizfinance.future import Future
from custom_calendar import CustomCalendar
from pydantic import BaseModel
import pandas as pd
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import time
import asyncio
from cachetools import TTLCache, cached

# Initialize caches with TTL (Time To Live)
stock_cache = TTLCache(maxsize=100, ttl=300)  # 5 minutes cache for stock data
screener_cache = TTLCache(maxsize=50, ttl=600)  # 10 minutes cache for screener data
calendar_cache = TTLCache(maxsize=10, ttl=1800)  # 30 minutes cache for calendar data

app = FastAPI(
    title="FinViz API",
    description="API for accessing FinViz financial data",
    version="1.0.0"
)

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

# Rate limiting middleware
class RateLimiter:
    def __init__(self, requests_per_minute=60):
        self.requests_per_minute = requests_per_minute
        self.requests = {}

    async def __call__(self, request: Request, call_next):
        client_ip = request.client.host
        now = time.time()
        
        # Clean old requests
        self.requests = {ip: reqs for ip, reqs in self.requests.items() 
                        if now - reqs[-1] < 60}
        
        # Check rate limit
        if client_ip in self.requests:
            if len(self.requests[client_ip]) >= self.requests_per_minute:
                return JSONResponse(
                    status_code=429,
                    content={"error": "Too many requests. Please try again later."}
                )
            self.requests[client_ip].append(now)
        else:
            self.requests[client_ip] = [now]
        
        return await call_next(request)

app.middleware("http")(RateLimiter())

class StockResponse(BaseModel):
    fundamentals: Dict
    description: str
    ratings: List[Dict]
    news: List[Dict]
    insider_trading: List[Dict]
    signal: Dict
    full_info: Dict

class ScreenerFilters(BaseModel):
    Exchange: Optional[str] = None
    Sector: Optional[str] = None
    Industry: Optional[str] = None
    Country: Optional[str] = None
    # Add more filter options as needed

class CalendarFilter(BaseModel):
    date: Optional[str] = None
    impact: Optional[str] = None
    release: Optional[str] = None

@app.get("/")
async def root():
    return {"message": "Welcome to FinViz API"}

@app.get("/stock/{ticker}")
@cached(cache=stock_cache)
async def get_stock_info(ticker: str):
    """Get comprehensive information about a specific stock"""
    try:
        stock = finvizfinance(ticker)
        
        # Helper function to safely convert DataFrame to dict with optimized operations
        def safe_df_to_dict(df, single_row=False):
            if not isinstance(df, pd.DataFrame) or df.empty:
                return {} if single_row else []
            try:
                # Use to_dict with orient='records' for better performance
                return df.to_dict('records')[0] if single_row else df.to_dict('records')
            except Exception:
                return {} if single_row else []
        
        # Gather all data concurrently
        async def gather_data():
            tasks = [
                asyncio.create_task(asyncio.to_thread(stock.ticker_fundament)),
                asyncio.create_task(asyncio.to_thread(stock.ticker_description)),
                asyncio.create_task(asyncio.to_thread(stock.ticker_outer_ratings)),
                asyncio.create_task(asyncio.to_thread(stock.ticker_news)),
                asyncio.create_task(asyncio.to_thread(stock.ticker_inside_trader)),
                asyncio.create_task(asyncio.to_thread(stock.ticker_signal)),
                asyncio.create_task(asyncio.to_thread(stock.ticker_full_info))
            ]
            return await asyncio.gather(*tasks, return_exceptions=True)
        
        # Get all data concurrently
        fundamentals, description, ratings_df, news, insider, signal, full_info = await gather_data()
        
        # Process fundamentals
        if not isinstance(fundamentals, dict):
            fundamentals = {}
            
        # Process description
        if isinstance(description, pd.DataFrame):
            description = description.iloc[0, 0] if not description.empty else ""
        elif not isinstance(description, str):
            description = ""
            
        # Process ratings
        ratings = []
        if isinstance(ratings_df, pd.DataFrame) and not ratings_df.empty:
            ratings = ratings_df.to_dict('records')
        
        # Process signal data
        signal_dict = {}
        if signal and isinstance(signal, list) and len(signal) > 0 and any(signal):
            signal_dict = {
                'Technical': signal[0] if len(signal) > 0 else {},
                'Pivot': signal[1] if len(signal) > 1 else {},
                'Oscillator': signal[2] if len(signal) > 2 else {}
            }
        elif isinstance(fundamentals, dict):
            signal_dict = {
                'Technical': {
                    'RSI': fundamentals.get('RSI (14)', 'N/A'),
                    'MACD': fundamentals.get('MACD', 'N/A'),
                    'MA50': fundamentals.get('SMA50', 'N/A'),
                    'MA200': fundamentals.get('SMA200', 'N/A')
                },
                'Pivot': {
                    'Pivot': fundamentals.get('Pivot', 'N/A'),
                    'Support1': fundamentals.get('S1', 'N/A'),
                    'Support2': fundamentals.get('S2', 'N/A'),
                    'Resistance1': fundamentals.get('R1', 'N/A'),
                    'Resistance2': fundamentals.get('R2', 'N/A')
                },
                'Oscillator': {
                    'StochRSI': fundamentals.get('StochRSI', 'N/A'),
                    'StochRSI_K': fundamentals.get('StochRSI_K', 'N/A'),
                    'StochRSI_D': fundamentals.get('StochRSI_D', 'N/A')
                }
            }
        
        # Process full info
        if not full_info and isinstance(fundamentals, dict):
            full_info = {
                'Company': fundamentals.get('Company', 'N/A'),
                'Sector': fundamentals.get('Sector', 'N/A'),
                'Industry': fundamentals.get('Industry', 'N/A'),
                'Country': fundamentals.get('Country', 'N/A'),
                'Market Cap': fundamentals.get('Market Cap', 'N/A'),
                'P/E': fundamentals.get('P/E', 'N/A'),
                'EPS': fundamentals.get('EPS (ttm)', 'N/A'),
                'Dividend': fundamentals.get('Dividend TTM', 'N/A'),
                'Beta': fundamentals.get('Beta', 'N/A'),
                'RSI': fundamentals.get('RSI (14)', 'N/A'),
                'MACD': fundamentals.get('MACD', 'N/A'),
                'MA50': fundamentals.get('SMA50', 'N/A'),
                'MA200': fundamentals.get('SMA200', 'N/A'),
                'Volume': fundamentals.get('Volume', 'N/A'),
                'Avg Volume': fundamentals.get('Avg Volume', 'N/A'),
                'Price': fundamentals.get('Price', 'N/A'),
                'Change': fundamentals.get('Change', 'N/A')
            }
        
        if not any([fundamentals, description, ratings, news, insider, signal_dict, full_info]):
            return {"error": "No data available for ticker"}
            
        return StockResponse(
            fundamentals=fundamentals,
            description=description,
            ratings=ratings,
            news=news,
            insider_trading=insider,
            signal=signal_dict,
            full_info=full_info
        )
    except Exception as e:
        return {"error": str(e)}

@app.post("/screener/overview")
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
        return {"error": str(e)}

@app.post("/screener/valuation")
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
        return {"error": str(e)}

@app.post("/screener/financial")
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
        return {"error": str(e)}

@app.get("/news")
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
        return {"error": str(e)}

@app.get("/insider")
async def get_insider_trading(option: str = Query("top owner trade", description="Type of insider trading data")):
    """Get insider trading information"""
    try:
        finsider = Insider(option=option)
        df = finsider.get_insider()
        # Replace NaN values with None before converting to JSON
        df = df.replace({pd.NA: None, pd.NaT: None, float('nan'): None})
        return df.to_dict('records')
    except Exception as e:
        return {"error": str(e)}

@app.get("/futures")
async def get_futures():
    """Get futures market performance"""
    try:
        future = Future()
        return future.performance().to_dict('records')
    except Exception as e:
        return {"error": str(e)}

@app.get("/calendar")
@cached(cache=calendar_cache)
async def get_calendar():
    """Get economic calendar events with detailed information"""
    try:
        calendar = CustomCalendar()
        calendar_data = calendar.calendar()
        
        if calendar_data.empty:
            return {
                "message": "No calendar data available at this time",
                "calendar": [],
                "total_events": 0,
                "dates": []
            }
        
        # Optimize DataFrame operations
        calendar_records = calendar_data.to_dict('records')
        dates = calendar_data['Date'].unique().tolist() if 'Date' in calendar_data.columns else []
        
        return {
            "calendar": calendar_records,
            "total_events": len(calendar_records),
            "dates": dates
        }
    except Exception as e:
        return {"error": str(e)}

@app.post("/calendar/filter")
async def filter_calendar(filters: CalendarFilter):
    """Filter economic calendar events by date, impact level, or release name"""
    try:
        calendar = CustomCalendar()
        calendar_data = calendar.calendar()
        
        # Check if the calendar data is empty
        if calendar_data.empty:
            return {
                "message": "No calendar data available at this time",
                "calendar": [],
                "total_events": 0,
                "dates": []
            }
        
        # Apply filters if provided
        if filters.date and 'Date' in calendar_data.columns:
            calendar_data = calendar_data[calendar_data['Date'].str.contains(filters.date, case=False)]
        
        if filters.impact and 'Impact' in calendar_data.columns:
            calendar_data = calendar_data[calendar_data['Impact'].str.contains(filters.impact, case=False)]
            
        if filters.release and 'Release' in calendar_data.columns:
            calendar_data = calendar_data[calendar_data['Release'].str.contains(filters.release, case=False)]
        
        # Convert the filtered DataFrame to a list of dictionaries
        calendar_records = calendar_data.to_dict('records')
        
        return {
            "calendar": calendar_records,
            "total_events": len(calendar_records),
            "dates": list(calendar_data['Date'].unique()) if 'Date' in calendar_data.columns else []
        }
    except Exception as e:
        return {"error": str(e)}

@app.get("/calendar/summary")
async def get_calendar_summary():
    """Get a summary of economic calendar events grouped by impact level"""
    try:
        calendar = CustomCalendar()
        calendar_data = calendar.calendar()
        
        # Check if the calendar data is empty
        if calendar_data.empty:
            return {
                "message": "No calendar data available at this time",
                "overall_summary": {},
                "today_summary": {},
                "total_events": 0,
                "today_events": 0
            }
        
        # Group by impact level and count events
        impact_summary = {}
        if 'Impact' in calendar_data.columns:
            impact_summary = calendar_data.groupby('Impact').size().to_dict()
        
        # Get today's events
        today_summary = {}
        today_events_count = 0
        if 'Date' in calendar_data.columns:
            today = pd.Timestamp.now().strftime('%a %b %d')
            today_events = calendar_data[calendar_data['Date'].str.contains(today, case=False)]
            today_events_count = len(today_events)
            if 'Impact' in today_events.columns:
                today_summary = today_events.groupby('Impact').size().to_dict()
        
        return {
            "overall_summary": impact_summary,
            "today_summary": today_summary,
            "total_events": len(calendar_data),
            "today_events": today_events_count
        }
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True) 