from fastapi import APIRouter, HTTPException, Query
from finvizfinance.screener.overview import Overview
from finvizfinance.screener.valuation import Valuation
from finvizfinance.screener.financial import Financial
import pandas as pd
import asyncio
from models import ScreenerFilters, VolumeResponse, VolumeStock
from cachetools import TTLCache
from logger import log_error
from functools import wraps

# Initialize cache
screener_cache = TTLCache(maxsize=50, ttl=600)  # 10 minutes cache for screener data
volume_cache = TTLCache(maxsize=1, ttl=300)  # 5 minutes cache for volume data

router = APIRouter(prefix="/screener", tags=["Screener"])

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

@router.post("/overview")
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

@router.post("/valuation")
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

@router.post("/financial")
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

@router.get("/volume", response_model=VolumeResponse)
@async_cached(volume_cache)
async def get_highest_volume(limit: int = Query(100, description="Number of stocks to return", ge=1, le=1000)):
    """Get stocks with highest trading volume"""
    try:
        # Initialize screener with specific filters
        screener = Overview()
        
        # Calculate how many pages we need based on the limit (20 results per page)
        pages_needed = (limit + 19) // 20  # Ceiling division
        df = pd.DataFrame()
        
        # Fetch each page up to the needed number
        for page in range(1, pages_needed + 1):
            page_df = screener.screener_view(
                order='Volume', 
                verbose=1, 
                select_page=page,
                ascend=False
            )
            df = pd.concat([df, page_df], ignore_index=True)
            
            # If we've reached our limit, stop fetching more pages
            if len(df) >= limit:
                break
                
            # Add delay to avoid rate limiting
            await asyncio.sleep(1)
        
        # Clean the Volume column if needed
        if df['Volume'].dtype == object:
            df['Volume'] = df['Volume'].str.replace(',', '').astype(int)
        
        # Sort by volume descending and ensure we only process up to the limit
        df = df.sort_values(by='Volume', ascending=False).head(limit)
        
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