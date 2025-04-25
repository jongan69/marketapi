from fastapi import APIRouter, HTTPException, Path, Query
from typing import Optional
import yfinance as yf
import numpy as np
from scipy.stats import norm
from datetime import datetime
import pandas as pd
from models import MaxPainResponse, MaxPainOpportunitiesResponse, MaxPainStock
from finvizfinance.screener.overview import Overview
import asyncio
from cachetools import TTLCache
from concurrent.futures import ThreadPoolExecutor
import time

router = APIRouter(prefix="/maxpain", tags=["Max Pain"])

# Initialize caches
ticker_cache = TTLCache(maxsize=100, ttl=300)  # 5 minutes cache for ticker objects
historical_data_cache = TTLCache(maxsize=100, ttl=300)  # 5 minutes cache for historical data
options_data_cache = TTLCache(maxsize=100, ttl=300)  # 5 minutes cache for options data

def get_cached_ticker(ticker_symbol: str):
    """Get or create a cached ticker object"""
    if ticker_symbol not in ticker_cache:
        ticker_cache[ticker_symbol] = yf.Ticker(ticker_symbol)
    return ticker_cache[ticker_symbol]

def get_cached_historical_data(ticker_symbol: str, days: int = 30):
    """Get or create cached historical data"""
    cache_key = f"{ticker_symbol}_{days}"
    if cache_key not in historical_data_cache:
        ticker = get_cached_ticker(ticker_symbol)
        historical_data_cache[cache_key] = ticker.history(period=f"{days}d")
    return historical_data_cache[cache_key]

def get_cached_options_data(ticker_symbol: str, expiration_date: str):
    """Get or create cached options data"""
    cache_key = f"{ticker_symbol}_{expiration_date}"
    if cache_key not in options_data_cache:
        ticker = get_cached_ticker(ticker_symbol)
        options_data_cache[cache_key] = ticker.option_chain(expiration_date)
    return options_data_cache[cache_key]

def calculate_historical_volatility(ticker_symbol: str, days: int = 30):
    """Calculate historical volatility over the specified number of days"""
    hist = get_cached_historical_data(ticker_symbol, days)
    returns = np.log(hist['Close'] / hist['Close'].shift(1))
    return returns.std() * np.sqrt(252)  # Annualized volatility

def calculate_probability_of_profit(spot_price, strike_price, days_to_expiry, volatility):
    """Calculate probability of profit using Black-Scholes model"""
    try:
        # Convert days to years
        t = max(days_to_expiry, 1) / 365.0  # Ensure at least 1 day
        
        # Calculate d1 and d2
        d1 = (np.log(spot_price / strike_price) + (0.5 * volatility**2) * t) / (volatility * np.sqrt(t))
        d2 = d1 - volatility * np.sqrt(t)
        
        # Calculate probability of profit
        if strike_price > spot_price:  # For calls
            return norm.cdf(d2)
        else:  # For puts
            return norm.cdf(-d2)
    except Exception:
        return 0.5  # Return 50% if calculation fails

def get_best_options_by_max_pain(ticker_symbol, expiration_date=None):
    """
    Get the best options based on max pain analysis for a given ticker with profitability metrics.
    """
    try:
        ticker = get_cached_ticker(ticker_symbol)
        
        # Get historical data with error handling
        try:
            hist = get_cached_historical_data(ticker_symbol)
            if hist.empty:
                raise ValueError(f"No data available for {ticker_symbol}")
            spot_price = hist['Close'].iloc[-1]
        except Exception as e:
            raise ValueError(f"No data available for {ticker_symbol}: {str(e)}")
        
        # Get available expirations
        try:
            expirations = ticker.options
            if not expirations:
                raise ValueError(f"No options data available for {ticker_symbol}")
        except Exception as e:
            raise ValueError(f"No options data available for {ticker_symbol}: {str(e)}")
        
        # Use specified expiration or next available
        target_exp = expiration_date if expiration_date else expirations[0]
        if target_exp not in expirations:
            raise ValueError(f"Expiration date {target_exp} not available")
        
        # Calculate historical volatility
        try:
            hist_vol = calculate_historical_volatility(ticker_symbol)
        except Exception as e:
            hist_vol = 0.3  # Default to 30% if calculation fails
        
        # Get option chain
        try:
            chain = get_cached_options_data(ticker_symbol, target_exp)
            calls, puts = chain.calls, chain.puts
        except Exception as e:
            raise ValueError(f"Failed to get option chain for {ticker_symbol}: {str(e)}")
        
        # Filter out options with zero open interest
        calls = calls[calls['openInterest'] > 0]
        puts = puts[puts['openInterest'] > 0]
        
        if len(calls) == 0 or len(puts) == 0:
            raise ValueError(f"No options with open interest found for {ticker_symbol}")
        
        # Calculate max pain
        strikes = sorted(set(calls['strike']).union(set(puts['strike'])))
        pain_values = []
        
        for strike in strikes:
            call_pain = ((strike - calls['strike']).clip(lower=0) * calls['openInterest']).sum()
            put_pain = ((puts['strike'] - strike).clip(lower=0) * puts['openInterest']).sum()
            total_pain = call_pain + put_pain
            pain_values.append((strike, total_pain))
        
        pain_df = pd.DataFrame(pain_values, columns=['strike', 'pain'])
        max_pain_strike = pain_df.loc[pain_df['pain'].idxmin(), 'strike']
        
        # Find best options near max pain
        best_call = calls.iloc[(calls['strike'] - max_pain_strike).abs().argmin()]
        best_put = puts.iloc[(puts['strike'] - max_pain_strike).abs().argmin()]
        
        # Calculate days to expiration
        exp_date = datetime.strptime(target_exp, '%Y-%m-%d')
        days_to_expiry = max(1, (exp_date - datetime.now()).days)  # Ensure at least 1 day
        
        # Calculate probability of profit
        call_pop = calculate_probability_of_profit(spot_price, best_call['strike'], days_to_expiry, hist_vol)
        put_pop = calculate_probability_of_profit(spot_price, best_put['strike'], days_to_expiry, hist_vol)
        
        # Calculate bid-ask spread percentage
        call_spread_pct = (best_call['ask'] - best_call['bid']) / best_call['ask'] * 100 if best_call['ask'] > 0 else 0
        put_spread_pct = (best_put['ask'] - best_put['bid']) / best_put['ask'] * 100 if best_put['ask'] > 0 else 0
        
        # Calculate profitability score (0-100)
        def calculate_profitability_score(option, is_call):
            try:
                distance_score = 100 * (1 - abs(option['strike'] - max_pain_strike) / spot_price)
                volume_score = min(100, option['volume'] / 1000)  # Normalize volume
                oi_score = min(100, option['openInterest'] / 1000)  # Normalize open interest
                spread_score = max(0, 100 - (option['ask'] - option['bid']) / option['ask'] * 1000) if option['ask'] > 0 else 0
                pop_score = calculate_probability_of_profit(spot_price, option['strike'], days_to_expiry, hist_vol) * 100
                
                return (distance_score * 0.3 + volume_score * 0.2 + oi_score * 0.2 + 
                        spread_score * 0.15 + pop_score * 0.15)
            except Exception:
                return 50
        
        call_score = calculate_profitability_score(best_call, True)
        put_score = calculate_profitability_score(best_put, False)
        
        return {
            'max_pain_strike': max_pain_strike,
            'spot_price': spot_price,
            'historical_volatility': hist_vol,
            'best_call': {
                'strike': best_call['strike'],
                'last_price': best_call['lastPrice'],
                'volume': best_call['volume'],
                'open_interest': best_call['openInterest'],
                'bid': best_call['bid'],
                'ask': best_call['ask'],
                'bid_ask_spread_pct': call_spread_pct,
                'probability_of_profit': call_pop,
                'profitability_score': call_score,
                'implied_volatility': best_call['impliedVolatility']
            },
            'best_put': {
                'strike': best_put['strike'],
                'last_price': best_put['lastPrice'],
                'volume': best_put['volume'],
                'open_interest': best_put['openInterest'],
                'bid': best_put['bid'],
                'ask': best_put['ask'],
                'bid_ask_spread_pct': put_spread_pct,
                'probability_of_profit': put_pop,
                'profitability_score': put_score,
                'implied_volatility': best_put['impliedVolatility']
            },
            'expiration': target_exp,
            'days_to_expiry': days_to_expiry
        }
    except Exception as e:
        raise ValueError(f"Error analyzing {ticker_symbol}: {str(e)}")

@router.get("/options/{symbol}", response_model=MaxPainResponse)
async def get_options_analysis(
    symbol: str = Path(..., description="Stock symbol (e.g., AAPL, MSFT)"),
    expiration_date: Optional[str] = Query(None, description="Specific expiration date (YYYY-MM-DD)")
):
    """Get options analysis for a specific stock based on max pain theory"""
    try:
        options_data = get_best_options_by_max_pain(symbol, expiration_date)
        return MaxPainResponse(**options_data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

async def process_stock(stock_data):
    """Process a single stock's options data"""
    ticker = stock_data['Ticker']
    try:
        # Get best options - pass the ticker symbol string
        try:
            options_data = get_best_options_by_max_pain(ticker)
        except ValueError as e:
            print(f"Error analyzing {ticker}: {str(e)}")
            return None
        except Exception as e:
            print(f"Unexpected error analyzing {ticker}: {str(e)}")
            return None
        
        # Clean NaN values in the options data
        def clean_nan_values(data):
            if isinstance(data, dict):
                return {k: clean_nan_values(v) for k, v in data.items()}
            elif isinstance(data, list):
                return [clean_nan_values(v) for v in data]
            elif isinstance(data, (int, float)) and np.isnan(data):
                return 0
            return data
        
        options_data = clean_nan_values(options_data)
        
        # Calculate overall opportunity score
        opportunity_score = (
            options_data['best_call']['profitability_score'] * 0.5 +
            options_data['best_put']['profitability_score'] * 0.5
        )
        
        return MaxPainStock(
            ticker=ticker,
            company=stock_data['Company'],
            volume=stock_data['Volume'],
            price=stock_data['Price'],
            max_pain_strike=options_data['max_pain_strike'],
            spot_price=options_data['spot_price'],
            call_strike=options_data['best_call']['strike'],
            call_price=options_data['best_call']['last_price'],
            call_volume=options_data['best_call']['volume'],
            call_oi=options_data['best_call']['open_interest'],
            call_pop=options_data['best_call']['probability_of_profit'],
            put_strike=options_data['best_put']['strike'],
            put_price=options_data['best_put']['last_price'],
            put_volume=options_data['best_put']['volume'],
            put_oi=options_data['best_put']['open_interest'],
            put_pop=options_data['best_put']['probability_of_profit'],
            opportunity_score=opportunity_score,
            expiration=options_data['expiration'],
            days_to_expiry=options_data['days_to_expiry']
        )
    except Exception as e:
        print(f"Error processing {ticker}: {str(e)}")
        return None

@router.get("/opportunities", response_model=MaxPainOpportunitiesResponse)
async def get_options_opportunities(
    limit: int = Query(100, description="Number of high volume stocks to analyze", ge=1, le=100),
    top_n: int = Query(10, description="Number of best opportunities to return", ge=1, le=20)
):
    """Get the best options opportunities from high volume stocks"""
    try:
        start_time = time.time()
        
        # Get high volume stocks using the more efficient approach
        screener = Overview()
        print(limit)
        
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
        
        # Process stocks in parallel using ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=10) as executor:
            # Convert DataFrame to list of dictionaries for parallel processing
            stock_list = df.to_dict('records')
            
            # Process stocks in parallel
            tasks = [process_stock(stock) for stock in stock_list]
            results = await asyncio.gather(*tasks)
        
        # Filter out None results and sort by opportunity score
        valid_results = [r for r in results if r is not None]
        valid_results.sort(key=lambda x: x.opportunity_score, reverse=True)
        
        # Take top_n opportunities
        top_opportunities = valid_results[:top_n]
        
        end_time = time.time()
        print(f"Total processing time: {end_time - start_time:.2f} seconds")
        
        return MaxPainOpportunitiesResponse(
            opportunities=top_opportunities,
            total_opportunities=len(top_opportunities)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 