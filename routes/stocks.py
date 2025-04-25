from fastapi import APIRouter, HTTPException, Path, Query
from typing import Optional
from finvizfinance.quote import finvizfinance
import asyncio
import pandas as pd
from models import StockResponse, CombinedMetricsResponse, AnalystMetricsResponse
from tools.combined_metrics import analyze_stock
from cachetools import TTLCache
from functools import wraps
from logger import log_error, log_endpoint_access, log_performance
import time

# Initialize cache
stock_cache = TTLCache(maxsize=100, ttl=300)  # 5 minutes cache for stock data

router = APIRouter(prefix="/stock", tags=["Stocks"])

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

@router.get("/{symbol}", response_model=StockResponse)
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

@router.get("/{symbol}/combined_metrics", response_model=CombinedMetricsResponse)
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

@router.get("/{symbol}/analysts", response_model=AnalystMetricsResponse)
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
        
        # Define a safe wrapper for ticker_full_info to handle errors
        async def safe_ticker_full_info():
            try:
                # Monkey patch the number_covert function to handle empty strings
                from finvizfinance.util import number_covert
                def safe_number_covert(num):
                    if not num or not isinstance(num, str):
                        return num
                    return number_covert(num)
                
                # Replace the original function with our safe version
                import finvizfinance.util
                finvizfinance.util.number_covert = safe_number_covert
                
                return await asyncio.to_thread(lambda: stock.ticker_full_info())
            except Exception as e:
                log_error(e, {"symbol": symbol, "operation": "ticker_full_info"})
                return pd.DataFrame()
        
        # Get analyst ratings and full info concurrently
        tasks = [
            asyncio.to_thread(lambda: stock.ticker_outer_ratings()),
            safe_ticker_full_info(),
            asyncio.to_thread(lambda: stock.ticker_fundament())
        ]
        ratings, full_info, fundament = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Helper function to safely convert data to dictionary
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
        
        # Handle ratings
        analyst_ratings = []
        if isinstance(ratings, pd.DataFrame) and not ratings.empty:
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
        
        # Handle full info
        info_dict = safe_df_to_dict(full_info, "dict")
        
        # Extract price targets from full_info or fundamentals
        if info_dict:
            if isinstance(info_dict, dict) and any(isinstance(v, dict) for v in info_dict.values()):
                for key, value in info_dict.items():
                    if isinstance(value, dict) and "Target Price" in value:
                        price_targets = {
                            "avg_price_target": {"value": value.get("Target Price", "N/A")},
                            "price_target_low": {"value": value.get("Price Target Low", "N/A")},
                            "price_target_high": {"value": value.get("Price Target High", "N/A")}
                        }
                        break
            else:
                price_targets = {
                    "avg_price_target": {"value": info_dict.get("Target Price", "N/A")},
                    "price_target_low": {"value": info_dict.get("Price Target Low", "N/A")},
                    "price_target_high": {"value": info_dict.get("Price Target High", "N/A")}
                }
        
        # If price targets are still N/A, try to extract from fundamentals
        if price_targets["avg_price_target"]["value"] == "N/A" and isinstance(fundament, dict):
            price_targets = {
                "avg_price_target": {"value": fundament.get("Target Price", "N/A")},
                "price_target_low": {"value": fundament.get("Price Target Low", "N/A")},
                "price_target_high": {"value": fundament.get("Price Target High", "N/A")}
            }
        
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
        
        # Handle fundamentals for estimates
        if isinstance(fundament, dict):
            # Extract earnings estimates
            earnings_estimates = {
                "current_quarter": {"value": fundament.get("EPS next Q", "N/A")},
                "next_quarter": {"value": fundament.get("EPS next Y", "N/A")},
                "current_year": {"value": fundament.get("EPS this Y", "N/A")},
                "next_year": {"value": fundament.get("EPS next Y", "N/A")}
            }
            
            # Extract revenue estimates
            revenue_estimates = {
                "current_quarter": {"value": fundament.get("Sales Q/Q", "N/A")},
                "next_quarter": {"value": fundament.get("Sales next Q", "N/A")},
                "current_year": {"value": fundament.get("Sales this Y", "N/A")},
                "next_year": {"value": fundament.get("Sales next Y", "N/A")}
            }
            
            # Extract EPS estimates
            eps_estimates = {
                "current_quarter": {"value": fundament.get("EPS Q/Q", "N/A")},
                "next_quarter": {"value": fundament.get("EPS next Q", "N/A")},
                "current_year": {"value": fundament.get("EPS this Y", "N/A")},
                "next_year": {"value": fundament.get("EPS next Y", "N/A")}
            }
        
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
        log_error(e, {"symbol": symbol})
        raise HTTPException(status_code=500, detail=f"Error fetching analyst metrics: {str(e)}") 