import os
from fastapi import FastAPI, Query, HTTPException, Path
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
from pydantic import BaseModel
import pandas as pd
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import time
import asyncio
import traceback
from cachetools import TTLCache
from functools import wraps
from api_config import custom_openapi, setup_middleware, RATE_LIMIT_REQUESTS
from rate_limiter import RateLimiter
from datetime import datetime
from fomc_calendar import FOMCCalendar
from logger import LoggingMiddleware, log_endpoint_access, log_error, log_performance
from tools.combined_metrics import analyze_stock

# Initialize caches with TTL (Time To Live)
stock_cache = TTLCache(maxsize=100, ttl=300)  # 5 minutes cache for stock data
screener_cache = TTLCache(maxsize=50, ttl=600)  # 10 minutes cache for screener data
calendar_cache = TTLCache(maxsize=10, ttl=1800)  # 30 minutes cache for calendar data
volume_cache = TTLCache(maxsize=1, ttl=300)  # 5 minutes cache for volume data

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

# Response Models
class StockResponse(BaseModel):
    """Response model for stock information"""
    fundamentals: Dict[str, Any]
    description: str
    ratings: List[Dict[str, Any]]
    news: List[Dict[str, Any]]
    insider_trading: List[Dict[str, Any]]
    signal: List[str]
    full_info: Dict[str, Any]

class AnalystMetricsResponse(BaseModel):
    """Response model for analyst metrics"""
    symbol: str
    analyst_ratings: List[Dict[str, Any]]
    price_targets: Dict[str, Any]
    recommendation_summary: Dict[str, Any]
    earnings_estimates: Dict[str, Any]
    revenue_estimates: Dict[str, Any]
    eps_estimates: Dict[str, Any]

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

class CombinedMetricsResponse(BaseModel):
    """Response model for combined metrics analysis"""
    symbol: str
    analysis: Dict[str, Any]
    combined_score: float
    analyst_scores: Dict[str, float]
    details: Dict[str, Any]

class VolumeStock(BaseModel):
    """Response model for volume stock data"""
    ticker: str
    company: str
    volume: int
    price: float

class VolumeResponse(BaseModel):
    """Response model for volume endpoint"""
    stocks: List[VolumeStock]
    total_stocks: int

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
    log_endpoint_access("/")
    return {"message": "Welcome to Market Data API"}

@app.get("/stock/{symbol}", response_model=StockResponse, tags=["Stocks"])
async def get_stock_info(
    symbol: str = Path(..., description="Stock symbol (e.g., AAPL, MSFT)"),
    screener: Optional[str] = Query(None, description="Screener to use (e.g., 'overview', 'all')")
):
    start_time = time.time()
    log_endpoint_access("/stock/{symbol}", symbol=symbol, screener=screener)
    
    try:
        # Initialize stock object
        stock = finvizfinance(symbol)
        
        async def gather_data():
            try:
                tasks = [
                    asyncio.to_thread(lambda: stock.ticker_fundament()),
                    asyncio.to_thread(lambda: stock.ticker_description()),
                    asyncio.to_thread(lambda: stock.ticker_outer_ratings()),
                    asyncio.to_thread(lambda: stock.ticker_news()),
                    asyncio.to_thread(lambda: stock.ticker_inside_trader()),
                    asyncio.to_thread(lambda: stock.ticker_signal()),
                    asyncio.to_thread(lambda: stock.ticker_full_info())
                ]
                return await asyncio.gather(*tasks, return_exceptions=True)
            except Exception as e:
                log_error(e, {"symbol": symbol, "operation": "gather_data"})
                raise HTTPException(status_code=500, detail=f"Error gathering data: {str(e)}")
        
        # Gather all data concurrently
        fundamentals, description, ratings, news, insider, signal, full_info = await gather_data()
        
        # Log performance
        duration = time.time() - start_time
        log_performance("get_stock_info", duration, symbol=symbol)
        
        # Convert DataFrames to dictionaries
        def safe_df_to_dict(data, field_type="list"):
            if data is None or isinstance(data, Exception):
                return [] if field_type == "list" else {}
            if isinstance(data, dict):
                return data
            if isinstance(data, pd.DataFrame):
                if data.empty:
                    return [] if field_type == "list" else {}
                return data.to_dict('records') if field_type == "list" else data.to_dict()
            if isinstance(data, list):
                return data
            if isinstance(data, str):
                return {"description": data}
            return [] if field_type == "list" else {}
        
        return StockResponse(
            fundamentals=safe_df_to_dict(fundamentals, "dict"),
            description=description if isinstance(description, str) else "",
            ratings=safe_df_to_dict(ratings, "list"),
            news=safe_df_to_dict(news, "list"),
            insider_trading=safe_df_to_dict(insider, "list"),
            signal=safe_df_to_dict(signal, "list"),
            full_info=safe_df_to_dict(full_info, "dict")
        )
    except Exception as e:
        log_error(e, {"symbol": symbol, "screener": screener})
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stock/{symbol}/combined_metrics", response_model=CombinedMetricsResponse, tags=["Stocks"])
async def get_combined_metrics(
    symbol: str = Path(..., description="Stock symbol (e.g., AAPL, MSFT)")
):
    """Get combined metrics analysis for a stock"""
    try:
        # Get the analysis from the combined_metrics module
        analysis_result = analyze_stock(symbol)
        
        if not analysis_result:
            raise HTTPException(status_code=404, detail=f"No data available for {symbol}")
        
        # Transform the data to match the expected response model
        analyst_scores = {}
        for analyst, data in analysis_result.items():
            if analyst != 'ticker' and analyst != 'combined_score' and analyst != 'technical_analysis':
                # Extract the score from the analyst's data
                if isinstance(data, dict):
                    # For analysts with multiple metrics, use the average score
                    scores = []
                    for metric, metric_data in data.items():
                        if isinstance(metric_data, dict) and 'score' in metric_data:
                            scores.append(metric_data['score'])
                    if scores:
                        analyst_scores[analyst] = sum(scores) / len(scores)
                    else:
                        analyst_scores[analyst] = 0.0
                elif isinstance(data, tuple) and len(data) > 0:
                    # For analysts that return a tuple with score as first element
                    analyst_scores[analyst] = data[0]
                else:
                    analyst_scores[analyst] = 0.0
        
        # Create the response object
        response = CombinedMetricsResponse(
            symbol=symbol,
            analysis=analysis_result,
            combined_score=analysis_result.get('combined_score', 0.0),
            analyst_scores=analyst_scores,
            details={
                "technical_analysis": analysis_result.get('technical_analysis', {}),
                "summary": f"Analysis for {symbol} with combined score of {analysis_result.get('combined_score', 0.0):.2f}"
            }
        )
        
        return response
    except Exception as e:
        log_error(e, {"symbol": symbol})
        raise HTTPException(status_code=500, detail=f"Error fetching combined metrics: {str(e)}")

@app.get("/stock/{symbol}/analysts", response_model=AnalystMetricsResponse, tags=["Stocks"])
@async_cached(stock_cache)
async def get_analyst_metrics(
    symbol: str = Path(..., description="Stock symbol (e.g., AAPL, MSFT)")
):
    """
    Get comprehensive analyst metrics for a given stock including ratings, price targets, and earnings estimates.
    
    Parameters:
    - symbol: Stock symbol (e.g., AAPL, MSFT)
    """
    try:
        # Initialize stock object
        stock = finvizfinance(symbol)
        
        # Define a safe wrapper for ticker_full_info to handle the IndexError
        async def safe_ticker_full_info():
            try:
                print(f"\nDEBUG: Starting ticker_full_info for {symbol}")
                
                # Get the raw HTML content
                print(f"DEBUG: Getting HTML content for {symbol}")
                html_content = stock.soup.prettify()
                print(f"DEBUG: HTML content length: {len(html_content)}")
                print(f"DEBUG: First 500 chars of HTML: {html_content[:500]}")
                
                # Get all tables in the HTML
                tables = stock.soup.find_all("table")
                print(f"DEBUG: Found {len(tables)} tables in HTML")
                
                for i, table in enumerate(tables):
                    headers = [th.text.strip() for th in table.find_all('th')]
                    print(f"DEBUG: Table {i} headers: {headers}")
                    
                    # Print first row of data for each table
                    rows = table.find_all('tr')
                    if len(rows) > 1:  # If there's at least one data row
                        first_row = rows[1]  # Skip header row
                        cells = [td.text.strip() for td in first_row.find_all('td')]
                        print(f"DEBUG: Table {i} first row data: {cells}")
                
                print("DEBUG: About to call ticker_full_info")
                
                # Monkey patch the number_covert function to handle empty strings
                from finvizfinance.util import number_covert
                def safe_number_covert(num):
                    if not num or not isinstance(num, str):
                        return num
                    return number_covert(num)
                
                # Replace the original function with our safe version
                import finvizfinance.util
                finvizfinance.util.number_covert = safe_number_covert
                
                result = await asyncio.to_thread(lambda: stock.ticker_full_info())
                print(f"DEBUG: ticker_full_info completed for {symbol}")
                print(f"DEBUG: Result type: {type(result)}")
                if isinstance(result, pd.DataFrame):
                    print(f"DEBUG: DataFrame shape: {result.shape}")
                    print(f"DEBUG: DataFrame columns: {result.columns.tolist()}")
                    print(f"DEBUG: DataFrame head: {result.head(1).to_dict('records') if not result.empty else 'Empty DataFrame'}")
                return result
            except IndexError as e:
                print(f"IndexError in ticker_full_info for {symbol}: {e}")
                print(f"DEBUG: IndexError traceback: {traceback.format_exc()}")
                # Return an empty DataFrame instead of propagating the error
                return pd.DataFrame()
            except Exception as e:
                print(f"Other error in ticker_full_info for {symbol}: {e}")
                print(f"DEBUG: Exception type: {type(e)}")
                print(f"DEBUG: Exception traceback: {traceback.format_exc()}")
                return e
        
        # Get analyst ratings and full info concurrently
        tasks = [
            asyncio.to_thread(lambda: stock.ticker_outer_ratings()),
            safe_ticker_full_info(),
            asyncio.to_thread(lambda: stock.ticker_fundament())
        ]
        ratings, full_info, fundament = await asyncio.gather(*tasks, return_exceptions=True)
        
        print(f"\nDEBUG: Data from finvizfinance for {symbol}:")
        print(f"Ratings type: {type(ratings)}")
        if isinstance(ratings, pd.DataFrame):
            print(f"Ratings columns: {ratings.columns.tolist()}")
            print(f"Ratings sample: {ratings.head(1).to_dict('records')}")
        else:
            print(f"Ratings data: {ratings}")
            
        print(f"\nFull info type: {type(full_info)}")
        if isinstance(full_info, pd.DataFrame):
            print(f"Full info columns: {full_info.columns.tolist()}")
            print(f"Full info sample: {full_info.head(1).to_dict('records')}")
        else:
            print(f"Full info data: {full_info}")
            
        print(f"\nFundament type: {type(fundament)}")
        if isinstance(fundament, dict):
            print(f"Fundament keys: {fundament.keys()}")
            print(f"Fundament sample: {dict(list(fundament.items())[:5])}")
        else:
            print(f"Fundament data: {fundament}")
        
        # Handle ratings
        analyst_ratings = []
        if isinstance(ratings, Exception):
            print(f"Error fetching ratings: {ratings}")
        elif isinstance(ratings, pd.DataFrame):
            if not ratings.empty:
                analyst_ratings = ratings.to_dict('records')
        elif isinstance(ratings, list):
            analyst_ratings = ratings
        elif isinstance(ratings, dict):
            analyst_ratings = [ratings]
        
        # Initialize empty dictionaries for metrics
        price_targets = {
            "avg_price_target": {"value": "N/A"},
            "price_target_low": {"value": "N/A"},
            "price_target_high": {"value": "N/A"}
        }
        recommendation_summary = {
            "strong_buy": {"count": 0},
            "buy": {"count": 0},
            "hold": {"count": 0},
            "sell": {"count": 0},
            "strong_sell": {"count": 0}
        }
        earnings_estimates = {
            "current_quarter": {"value": "N/A"},
            "next_quarter": {"value": "N/A"},
            "current_year": {"value": "N/A"},
            "next_year": {"value": "N/A"}
        }
        revenue_estimates = {
            "current_quarter": {"value": "N/A"},
            "next_quarter": {"value": "N/A"},
            "current_year": {"value": "N/A"},
            "next_year": {"value": "N/A"}
        }
        eps_estimates = {
            "current_quarter": {"value": "N/A"},
            "next_quarter": {"value": "N/A"},
            "current_year": {"value": "N/A"},
            "next_year": {"value": "N/A"}
        }
        
        # Helper function to safely convert data to dictionary
        def safe_df_to_dict(data, field_type="list"):
            if data is None or isinstance(data, Exception):
                return [] if field_type == "list" else {}
            if isinstance(data, dict):
                return data
            if isinstance(data, pd.DataFrame):
                if data.empty:
                    return [] if field_type == "list" else {}
                try:
                    return data.to_dict('records') if field_type == "list" else data.to_dict()
                except Exception as e:
                    print(f"Error converting DataFrame to dict: {e}")
                    return [] if field_type == "list" else {}
            if isinstance(data, list):
                return data
            if isinstance(data, str):
                return {"description": data}
            return [] if field_type == "list" else {}
        
        # Handle full info
        info_dict = safe_df_to_dict(full_info, "dict")
        
        # Try to extract price targets from full_info
        if info_dict:
            # Check if info_dict is a nested dictionary (from DataFrame.to_dict())
            if isinstance(info_dict, dict) and any(isinstance(v, dict) for v in info_dict.values()):
                # Try to find the first non-empty value
                for key, value in info_dict.items():
                    if isinstance(value, dict) and "Target Price" in value:
                        price_targets = {
                            "avg_price_target": {"value": value.get("Target Price", "N/A")},
                            "price_target_low": {"value": value.get("Price Target Low", "N/A")},
                            "price_target_high": {"value": value.get("Price Target High", "N/A")}
                        }
                        break
            else:
                # Direct dictionary with keys
                price_targets = {
                    "avg_price_target": {"value": info_dict.get("Target Price", "N/A")},
                    "price_target_low": {"value": info_dict.get("Price Target Low", "N/A")},
                    "price_target_high": {"value": info_dict.get("Price Target High", "N/A")}
                }
        
        # If price targets are still N/A, try to extract from fundamentals
        if price_targets["avg_price_target"]["value"] == "N/A" and isinstance(fundament, dict):
            # Check for price target in fundamentals
            if "Target Price" in fundament:
                price_targets["avg_price_target"]["value"] = fundament["Target Price"]
            if "Price Target Low" in fundament:
                price_targets["price_target_low"]["value"] = fundament["Price Target Low"]
            if "Price Target High" in fundament:
                price_targets["price_target_high"]["value"] = fundament["Price Target High"]
        
        print(f"\nExtracted price targets: {price_targets}")
        
        # Extract recommendation summary from analyst ratings
        if analyst_ratings:
            for rating in analyst_ratings:
                rating_text = rating.get("Rating", "").lower()
                if "strong buy" in rating_text or "outperform" in rating_text:
                    recommendation_summary["strong_buy"]["count"] += 1
                elif "buy" in rating_text:
                    recommendation_summary["buy"]["count"] += 1
                elif "hold" in rating_text or "neutral" in rating_text or "sector weight" in rating_text:
                    recommendation_summary["hold"]["count"] += 1
                elif "sell" in rating_text or "underperform" in rating_text:
                    recommendation_summary["sell"]["count"] += 1
                elif "strong sell" in rating_text:
                    recommendation_summary["strong_sell"]["count"] += 1
        print(f"\nExtracted recommendation summary: {recommendation_summary}")
        
        # Handle fundamentals for estimates
        if isinstance(fundament, dict):
            # Extract earnings estimates
            earnings_estimates = {
                "current_quarter": {"value": fundament.get("EPS next Q", "N/A")},
                "next_quarter": {"value": fundament.get("EPS next Y", "N/A")},
                "current_year": {"value": fundament.get("EPS this Y", "N/A")},
                "next_year": {"value": fundament.get("EPS next Y", "N/A")}
            }
            print(f"\nExtracted earnings estimates: {earnings_estimates}")
            
            # Extract revenue estimates
            revenue_estimates = {
                "current_quarter": {"value": fundament.get("Sales Q/Q", "N/A")},
                "next_quarter": {"value": fundament.get("Sales next Q", "N/A")},
                "current_year": {"value": fundament.get("Sales this Y", "N/A")},
                "next_year": {"value": fundament.get("Sales next Y", "N/A")}
            }
            print(f"\nExtracted revenue estimates: {revenue_estimates}")
            
            # Extract EPS estimates
            eps_estimates = {
                "current_quarter": {"value": fundament.get("EPS Q/Q", "N/A")},
                "next_quarter": {"value": fundament.get("EPS next Q", "N/A")},
                "current_year": {"value": fundament.get("EPS this Y", "N/A")},
                "next_year": {"value": fundament.get("EPS next Y", "N/A")}
            }
            print(f"\nExtracted EPS estimates: {eps_estimates}")
        
        return AnalystMetricsResponse(
            symbol=symbol,
            analyst_ratings=analyst_ratings,
            price_targets=price_targets,
            recommendation_summary=recommendation_summary,
            earnings_estimates=earnings_estimates,
            revenue_estimates=revenue_estimates,
            eps_estimates=eps_estimates
        )
    except Exception as e:
        print(f"Error in get_analyst_metrics: {str(e)}")
        print(f"Exception type: {type(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Error fetching analyst metrics: {str(e)}")

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
async def get_insider_trading(option: str = Query(
    "latest",
    description="Type of insider trading data to retrieve. Available options: 'latest' (default), 'latest buys', 'latest sales', 'top week', 'top week buys', 'top week sales', 'top owner trade', 'top owner buys', 'top owner sales', or an insider ID number"
)):
    """Get insider trading information from FinViz.
    
    Returns a list of insider trading activities including:
    - Ticker symbol
    - Owner name
    - Relationship to company
    - Transaction date
    - Transaction type (Buy/Sale/Option Exercise)
    - Cost per share
    - Number of shares
    - Total value
    - Total shares owned
    - SEC Form 4 filing link
    
    The data is sorted by date (most recent first) and includes detailed information about insider trading activities
    such as executive stock sales, director purchases, and major shareholder transactions.
    """
    try:
        finsider = Insider(option=option)
        
        # Get all tables from the page
        tables = finsider.soup.find_all("table")
        if not tables:
            print(f"No tables found for option: {option}")
            return []
            
        # Find the table with insider trading data by looking for expected headers
        insider_table = None
        expected_headers = ["Ticker", "Owner", "Relationship", "Date", "Transaction", "Cost", "#Shares", "Value ($)", "#Shares Total", "SEC Form 4"]
        
        for table in tables:
            headers = [th.text.strip() for th in table.find_all("th")]
            if all(header in headers for header in expected_headers):
                insider_table = table
                break
                
        if insider_table is None:
            print(f"Could not find insider trading table with expected headers for option: {option}")
            return []
            
        # Process the table rows
        rows = insider_table.find_all("tr")[1:]  # Skip header row
        frame = []
        
        for row in rows:
            cols = row.find_all("td")
            if len(cols) < 10:  # We expect at least 10 columns
                continue
                
            info_dict = {
                "Ticker": cols[0].text.strip(),
                "Owner": cols[1].text.strip(),
                "Relationship": cols[2].text.strip(),
                "Date": cols[3].text.strip(),
                "Transaction": cols[4].text.strip(),
                "Cost": cols[5].text.strip(),
                "#Shares": cols[6].text.strip(),
                "Value ($)": cols[7].text.strip(),
                "#Shares Total": cols[8].text.strip(),
                "SEC Form 4": cols[9].find("a")["href"] if cols[9].find("a") else None
            }
            frame.append(info_dict)
            
        df = pd.DataFrame(frame)
        if df.empty:
            print(f"No insider trading data found for option: {option}")
            return []
            
        # Replace NaN values with None before converting to JSON
        df = df.replace({pd.NA: None, pd.NaT: None, float('nan'): None})
        return df.to_dict('records')
    except Exception as e:
        print(f"Error in insider trading: {str(e)}")
        return []

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
    """Get economic calendar data"""
    try:
        calendar = CustomCalendar()
        df = await calendar.calendar()
        
        if df is None or df.empty:
            # Return empty response with 200 status
            return CalendarResponse(
                events=[],
                total_events=0,
                available_dates=[]
            )
            
        # Convert DataFrame to list of events
        events = []
        for _, row in df.iterrows():
            try:
                event = CalendarEvent(
                    Date=row["Date"],
                    Time=row["Time"],
                    Datetime=row["Datetime"],
                    Release=row["Release"],
                    Impact=row["Impact"],
                    For=row["For"],
                    Actual=row.get("Actual"),
                    Expected=row.get("Expected"),
                    Prior=row.get("Prior")
                )
                events.append(event)
            except Exception as e:
                print(f"Error processing calendar event: {e}")
                continue
                
        # Get unique dates for available_dates
        available_dates = sorted(df["Date"].unique().tolist())
        
        return CalendarResponse(
            events=events,
            total_events=len(events),
            available_dates=available_dates
        )
    except Exception as e:
        # Log the error but return empty response with 200 status
        print(f"Calendar endpoint error: {e}")
        return CalendarResponse(
            events=[],
            total_events=0,
            available_dates=[]
        )

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

@app.get("/health", response_model=HealthCheck, tags=["Health"])
async def health_check():
    """Check API health status"""
    try:
        # Check if services are responding
        calendar_status = "healthy"
        try:
            # Test calendar service
            calendar = CustomCalendar()
            await calendar.calendar()
        except Exception as e:
            print(f"Calendar service health check failed: {e}")
            calendar_status = "unhealthy"
            
        # Calculate uptime
        uptime_seconds = int(time.time() - app.state.start_time)
        uptime = str(datetime.timedelta(seconds=uptime_seconds))
        
        return HealthCheck(
            status="healthy",
            version="1.0.0",
            timestamp=datetime.now().isoformat(),
            uptime=uptime,
            services={
                "calendar": calendar_status
            }
        )
    except Exception as e:
        # Even if health check fails, return 200 with degraded status
        print(f"Health check error: {e}")
        return HealthCheck(
            status="degraded",
            version="1.0.0",
            timestamp=datetime.now().isoformat(),
            uptime="unknown",
            services={
                "calendar": "unknown"
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
                # Fetch minutes text if minutes link exists
                if latest_meeting['Minutes_Link']:
                    latest_meeting['Minutes_Text'] = await calendar._fetch_minutes_text(latest_meeting['Minutes_Link'])
                    if latest_meeting['Minutes_Text']:
                        latest_meeting['Minutes_Summary'] = await calendar.get_minutes_summary(
                            latest_meeting['Minutes_Link'], 
                            latest_meeting['Minutes_Text']
                        )
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

@app.get("/screener/volume", response_model=VolumeResponse, tags=["Screener"])
@async_cached(volume_cache)
async def get_highest_volume(limit: int = Query(100, description="Number of stocks to return", ge=1, le=1000)):
    """Get stocks with highest trading volume"""
    try:
        # Initialize screener with specific filters
        screener = Overview()
        
        # Set filters to reduce the dataset size
        screener.set_filter(filters_dict={
            'Exchange': 'Any',  # All exchanges
            'Current Volume': 'Over 1M',  # Only high volume stocks
        })
        
        # Get screener data with minimal pages
        df = screener.screener_view(order='Volume', limit=min(limit, 100), ascend=False)
        
        # Clean the Volume column if needed
        if df['Volume'].dtype == object:
            df['Volume'] = df['Volume'].str.replace(',', '').astype(int)
        
        # Sort by volume descending
        df = df.sort_values(by='Volume', ascending=False)
        
        # Convert to response model
        stocks = [
            VolumeStock(
                ticker=row['Ticker'],
                company=row['Company'],
                volume=int(row['Volume']),
                price=float(row['Price'])
            )
            for _, row in df.iterrows()
        ]
        
        return VolumeResponse(
            stocks=stocks,
            total_stocks=len(stocks)
        )
    except Exception as e:
        log_error(e, {"endpoint": "get_highest_volume", "limit": limit})
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))  # Default to 8000 if PORT is not set
    uvicorn.run("api:app", host="0.0.0.0", port=port, reload=True)
