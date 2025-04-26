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
import logging
import sys
import traceback
import threading
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/maxpain", tags=["Max Pain"])

# Initialize caches with production-safe limits
MAX_CACHE_SIZE = 1000  # Maximum number of items in cache
CACHE_TTL = 300  # 5 minutes in seconds
ticker_cache = TTLCache(maxsize=MAX_CACHE_SIZE, ttl=CACHE_TTL)
historical_data_cache = TTLCache(maxsize=MAX_CACHE_SIZE, ttl=CACHE_TTL)
options_data_cache = TTLCache(maxsize=MAX_CACHE_SIZE, ttl=CACHE_TTL)
finviz_cache = TTLCache(maxsize=1, ttl=CACHE_TTL)

# Lock for thread-safe cache updates
cache_lock = threading.Lock()

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((ConnectionError, TimeoutError))
)
def get_cached_ticker(ticker_symbol: str):
    """Get or create a cached ticker object with retry logic"""
    logger.debug(f"Getting cached ticker for {ticker_symbol}")
    
    with cache_lock:
        if ticker_symbol not in ticker_cache:
            logger.debug(f"Ticker {ticker_symbol} not in cache, creating new ticker")
            try:
                # Log environment and request details
                logger.debug(f"Environment details for {ticker_symbol}:")
                logger.debug(f"Python version: {sys.version}")
                logger.debug(f"yfinance version: {yf.__version__}")
                logger.debug(f"Current time: {datetime.now()}")
                logger.debug(f"Timezone: {datetime.now().astimezone().tzinfo}")
                
                # Create ticker and log basic info
                ticker = yf.Ticker(ticker_symbol)
                logger.debug(f"Created yfinance Ticker object for {ticker_symbol}")
                
                # Try to get basic info first
                try:
                    info = ticker.info
                    logger.debug(f"Basic info available for {ticker_symbol}: {bool(info)}")
                    if info:
                        logger.debug(f"Ticker info keys: {list(info.keys())}")
                        logger.debug(f"Exchange: {info.get('exchange', 'N/A')}")
                        logger.debug(f"Market: {info.get('market', 'N/A')}")
                        logger.debug(f"Quote type: {info.get('quoteType', 'N/A')}")
                except Exception as e:
                    logger.warning(f"Error getting basic info for {ticker_symbol}: {str(e)}")
                
                # Try to get history to verify data availability
                try:
                    hist = ticker.history(period="1d")
                    logger.debug(f"History data available: {not hist.empty}")
                    if not hist.empty:
                        logger.debug(f"Latest price: {hist['Close'].iloc[-1]}")
                except Exception as e:
                    logger.warning(f"Error getting history for {ticker_symbol}: {str(e)}")
                
                # Check if options are available
                try:
                    options = ticker.options
                    logger.debug(f"Options data available for {ticker_symbol}: {bool(options)}")
                    if options:
                        logger.debug(f"Available expirations: {options}")
                        # Try to get one option chain to verify
                        try:
                            chain = ticker.option_chain(options[0])
                            logger.debug(f"Option chain available: {not chain.calls.empty or not chain.puts.empty}")
                            if not chain.calls.empty:
                                logger.debug(f"Number of calls: {len(chain.calls)}")
                            if not chain.puts.empty:
                                logger.debug(f"Number of puts: {len(chain.puts)}")
                        except Exception as e:
                            logger.warning(f"Error getting option chain for {ticker_symbol}: {str(e)}")
                    else:
                        logger.warning(f"No options available for {ticker_symbol}")
                        raise ValueError(f"No options data available for {ticker_symbol}")
                except Exception as e:
                    logger.error(f"Error checking options for {ticker_symbol}: {str(e)}")
                    logger.error(f"Error type: {type(e).__name__}")
                    logger.error(f"Error traceback: {traceback.format_exc()}")
                    raise ValueError(f"No options data available for {ticker_symbol}")
                
                ticker_cache[ticker_symbol] = ticker
                logger.debug(f"Successfully cached ticker for {ticker_symbol}")
            except Exception as e:
                logger.error(f"Error creating ticker for {ticker_symbol}: {str(e)}")
                logger.error(f"Error type: {type(e).__name__}")
                logger.error(f"Error traceback: {traceback.format_exc()}")
                raise
    return ticker_cache[ticker_symbol]

def get_cached_historical_data(ticker_symbol: str, days: int = 30):
    """Get or create cached historical data"""
    logger.debug(f"Getting cached historical data for {ticker_symbol} for {days} days")
    cache_key = f"{ticker_symbol}_{days}"
    if cache_key not in historical_data_cache:
        logger.debug(f"Historical data not in cache for {cache_key}, fetching new data")
        try:
            ticker = get_cached_ticker(ticker_symbol)
            hist = ticker.history(period=f"{days}d")
            logger.debug(f"Historical data fetch response for {ticker_symbol}:")
            logger.debug(f"Data shape: {hist.shape if not hist.empty else 'Empty DataFrame'}")
            logger.debug(f"Columns: {hist.columns.tolist() if not hist.empty else 'No columns'}")
            logger.debug(f"Date range: {hist.index[0] if not hist.empty else 'No dates'} to {hist.index[-1] if not hist.empty else 'No dates'}")
            logger.debug(f"Sample data: {hist.head(1).to_dict() if not hist.empty else 'No data'}")
            historical_data_cache[cache_key] = hist
        except Exception as e:
            logger.error(f"Error fetching historical data for {ticker_symbol}: {str(e)}")
            logger.error(f"Error type: {type(e).__name__}")
            raise
    return historical_data_cache[cache_key]

def get_cached_options_data(ticker_symbol: str, expiration_date: str):
    """Get or create cached options data"""
    logger.debug(f"Getting cached options data for {ticker_symbol} with expiration {expiration_date}")
    cache_key = f"{ticker_symbol}_{expiration_date}"
    if cache_key not in options_data_cache:
        logger.debug(f"Options data not in cache for {cache_key}, fetching new data")
        try:
            ticker = get_cached_ticker(ticker_symbol)
            chain = ticker.option_chain(expiration_date)
            logger.debug(f"Options chain fetch response for {ticker_symbol}:")
            logger.debug(f"Calls shape: {chain.calls.shape if not chain.calls.empty else 'Empty DataFrame'}")
            logger.debug(f"Puts shape: {chain.puts.shape if not chain.puts.empty else 'Empty DataFrame'}")
            logger.debug(f"Calls columns: {chain.calls.columns.tolist() if not chain.calls.empty else 'No columns'}")
            logger.debug(f"Puts columns: {chain.puts.columns.tolist() if not chain.puts.empty else 'No columns'}")
            logger.debug(f"Sample calls: {chain.calls.head(1).to_dict() if not chain.calls.empty else 'No calls'}")
            logger.debug(f"Sample puts: {chain.puts.head(1).to_dict() if not chain.puts.empty else 'No puts'}")
            options_data_cache[cache_key] = chain
        except Exception as e:
            logger.error(f"Error fetching options data for {ticker_symbol}: {str(e)}")
            logger.error(f"Error type: {type(e).__name__}")
            raise
    return options_data_cache[cache_key]

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((ConnectionError, TimeoutError))
)
def get_cached_finviz_data(limit: int):
    """Get or create cached Finviz screener data with retry logic"""
    cache_key = f"finviz_{limit}"
    
    with cache_lock:
        if cache_key not in finviz_cache:
            logger.debug(f"Finviz data not in cache for limit {limit}, fetching new data")
            try:
                screener = Overview()
                pages_needed = (limit + 19) // 20
                df = pd.DataFrame()
                
                for page in range(1, pages_needed + 1):
                    try:
                        logger.debug(f"Fetching page {page} of screener data")
                        page_df = screener.screener_view(
                            order='Volume', 
                            verbose=1, 
                            select_page=page,
                            ascend=False
                        )
                        df = pd.concat([df, page_df], ignore_index=True)
                        
                        if len(df) >= limit:
                            logger.debug(f"Reached desired limit of {limit} stocks")
                            break
                            
                        time.sleep(2)  # Increased rate limiting for production
                    except Exception as e:
                        logger.error(f"Error fetching page {page}: {str(e)}")
                        if page == 1:  # If first page fails, raise the error
                            raise
                        break  # If subsequent page fails, use what we have
                
                if df.empty:
                    raise ValueError("No data retrieved from Finviz")
                
                if df['Volume'].dtype == object:
                    df['Volume'] = df['Volume'].str.replace(',', '').astype(int)
                
                df = df.sort_values(by='Volume', ascending=False).head(limit)
                logger.debug(f"Retrieved {len(df)} high volume stocks")
                
                finviz_cache[cache_key] = df
            except Exception as e:
                logger.error(f"Error fetching Finviz data: {str(e)}")
                if cache_key in finviz_cache:
                    logger.warning("Using stale cache data due to fetch error")
                    return finviz_cache[cache_key]
                raise
        return finviz_cache[cache_key]

def calculate_historical_volatility(ticker_symbol: str, days: int = 30):
    """Calculate historical volatility over the specified number of days"""
    logger.debug(f"Calculating historical volatility for {ticker_symbol} over {days} days")
    hist = get_cached_historical_data(ticker_symbol, days)
    returns = np.log(hist['Close'] / hist['Close'].shift(1))
    volatility = returns.std() * np.sqrt(252)  # Annualized volatility
    logger.debug(f"Calculated volatility for {ticker_symbol}: {volatility:.4f}")
    return volatility

def calculate_probability_of_profit(spot_price, strike_price, days_to_expiry, volatility):
    """Calculate probability of profit using Black-Scholes model"""
    logger.debug(f"Calculating probability of profit - Spot: {spot_price}, Strike: {strike_price}, Days: {days_to_expiry}, Vol: {volatility}")
    try:
        t = max(days_to_expiry, 1) / 365.0
        d1 = (np.log(spot_price / strike_price) + (0.5 * volatility**2) * t) / (volatility * np.sqrt(t))
        d2 = d1 - volatility * np.sqrt(t)
        
        if strike_price > spot_price:
            pop = norm.cdf(d2)
        else:
            pop = norm.cdf(-d2)
            
        logger.debug(f"Calculated probability of profit: {pop:.4f}")
        return pop
    except Exception as e:
        logger.error(f"Error calculating probability of profit: {str(e)}")
        return 0.5

def get_best_options_by_max_pain(ticker_symbol, expiration_date=None):
    """
    Get the best options based on max pain analysis for a given ticker with profitability metrics.
    """
    logger.info(f"Starting max pain analysis for {ticker_symbol}")
    try:
        logger.debug(f"Attempting to fetch ticker data for {ticker_symbol}")
        ticker = get_cached_ticker(ticker_symbol)
        
        try:
            logger.debug(f"Fetching historical data for {ticker_symbol}")
            hist = get_cached_historical_data(ticker_symbol)
            
            if hist is None:
                logger.error(f"Historical data fetch returned None for {ticker_symbol}")
                raise ValueError(f"No data available for {ticker_symbol}")
                
            if hist.empty:
                logger.error(f"Historical data is empty for {ticker_symbol}")
                logger.debug(f"Historical data columns: {hist.columns.tolist() if not hist.empty else 'No columns'}")
                raise ValueError(f"No data available for {ticker_symbol}")
                
            spot_price = hist['Close'].iloc[-1]
            logger.debug(f"Current spot price for {ticker_symbol}: {spot_price}")
            logger.debug(f"Historical data range: {hist.index[0]} to {hist.index[-1]}")
            logger.debug(f"Historical data points: {len(hist)}")
        except Exception as e:
            logger.error(f"Error getting historical data for {ticker_symbol}: {str(e)}")
            logger.error(f"Error type: {type(e).__name__}")
            raise ValueError(f"No data available for {ticker_symbol}: {str(e)}")
        
        try:
            logger.debug(f"Fetching options data for {ticker_symbol}")
            expirations = ticker.options
            
            if expirations is None:
                logger.error(f"Options data fetch returned None for {ticker_symbol}")
                raise ValueError(f"No options data available for {ticker_symbol}")
                
            if not expirations:
                logger.error(f"No expiration dates available for {ticker_symbol}")
                raise ValueError(f"No options data available for {ticker_symbol}")
                
            logger.debug(f"Available expirations for {ticker_symbol}: {expirations}")
        except Exception as e:
            logger.error(f"Error getting expirations for {ticker_symbol}: {str(e)}")
            logger.error(f"Error type: {type(e).__name__}")
            raise ValueError(f"No options data available for {ticker_symbol}: {str(e)}")
        
        target_exp = expiration_date if expiration_date else expirations[0]
        if target_exp not in expirations:
            logger.error(f"Expiration date {target_exp} not available for {ticker_symbol}")
            logger.debug(f"Available expirations: {expirations}")
            raise ValueError(f"Expiration date {target_exp} not available")
        
        try:
            hist_vol = calculate_historical_volatility(ticker_symbol)
            logger.debug(f"Historical volatility for {ticker_symbol}: {hist_vol:.4f}")
        except Exception as e:
            logger.warning(f"Error calculating historical volatility for {ticker_symbol}: {str(e)}")
            logger.warning(f"Error type: {type(e).__name__}")
            hist_vol = 0.3
        
        try:
            logger.debug(f"Fetching option chain for {ticker_symbol} with expiration {target_exp}")
            chain = get_cached_options_data(ticker_symbol, target_exp)
            
            if chain is None:
                logger.error(f"Option chain fetch returned None for {ticker_symbol}")
                raise ValueError(f"Failed to get option chain for {ticker_symbol}")
                
            calls, puts = chain.calls, chain.puts
            logger.debug(f"Retrieved options chain for {ticker_symbol} - Calls: {len(calls)}, Puts: {len(puts)}")
            logger.debug(f"Call strikes: {calls['strike'].tolist() if not calls.empty else 'No calls'}")
            logger.debug(f"Put strikes: {puts['strike'].tolist() if not puts.empty else 'No puts'}")
        except Exception as e:
            logger.error(f"Error getting option chain for {ticker_symbol}: {str(e)}")
            logger.error(f"Error type: {type(e).__name__}")
            raise ValueError(f"Failed to get option chain for {ticker_symbol}: {str(e)}")
        
        calls = calls[calls['openInterest'] > 0]
        puts = puts[puts['openInterest'] > 0]
        logger.debug(f"Filtered options with OI - Calls: {len(calls)}, Puts: {len(puts)}")
        
        if len(calls) == 0 or len(puts) == 0:
            logger.error(f"No options with open interest found for {ticker_symbol}")
            logger.debug(f"Available calls after filtering: {len(calls)}")
            logger.debug(f"Available puts after filtering: {len(puts)}")
            raise ValueError(f"No options with open interest found for {ticker_symbol}")
        
        strikes = sorted(set(calls['strike']).union(set(puts['strike'])))
        pain_values = []
        
        for strike in strikes:
            call_pain = ((strike - calls['strike']).clip(lower=0) * calls['openInterest']).sum()
            put_pain = ((puts['strike'] - strike).clip(lower=0) * puts['openInterest']).sum()
            total_pain = call_pain + put_pain
            pain_values.append((strike, total_pain))
        
        pain_df = pd.DataFrame(pain_values, columns=['strike', 'pain'])
        max_pain_strike = pain_df.loc[pain_df['pain'].idxmin(), 'strike']
        logger.debug(f"Calculated max pain strike for {ticker_symbol}: {max_pain_strike}")
        
        best_call = calls.iloc[(calls['strike'] - max_pain_strike).abs().argmin()]
        best_put = puts.iloc[(puts['strike'] - max_pain_strike).abs().argmin()]
        logger.debug(f"Best call strike: {best_call['strike']}, Best put strike: {best_put['strike']}")
        
        exp_date = datetime.strptime(target_exp, '%Y-%m-%d')
        days_to_expiry = max(1, (exp_date - datetime.now()).days)
        
        call_pop = calculate_probability_of_profit(spot_price, best_call['strike'], days_to_expiry, hist_vol)
        put_pop = calculate_probability_of_profit(spot_price, best_put['strike'], days_to_expiry, hist_vol)
        
        call_spread_pct = (best_call['ask'] - best_call['bid']) / best_call['ask'] * 100 if best_call['ask'] > 0 else 0
        put_spread_pct = (best_put['ask'] - best_put['bid']) / best_put['ask'] * 100 if best_put['ask'] > 0 else 0
        
        def calculate_profitability_score(option, is_call):
            try:
                distance_score = 100 * (1 - abs(option['strike'] - max_pain_strike) / spot_price)
                volume_score = min(100, option['volume'] / 1000)
                oi_score = min(100, option['openInterest'] / 1000)
                spread_score = max(0, 100 - (option['ask'] - option['bid']) / option['ask'] * 1000) if option['ask'] > 0 else 0
                pop_score = calculate_probability_of_profit(spot_price, option['strike'], days_to_expiry, hist_vol) * 100
                
                score = (distance_score * 0.3 + volume_score * 0.2 + oi_score * 0.2 + 
                        spread_score * 0.15 + pop_score * 0.15)
                logger.debug(f"Calculated profitability score for {'call' if is_call else 'put'}: {score:.2f}")
                return score
            except Exception as e:
                logger.error(f"Error calculating profitability score: {str(e)}")
                return 50
        
        call_score = calculate_profitability_score(best_call, True)
        put_score = calculate_profitability_score(best_put, False)
        
        logger.info(f"Completed max pain analysis for {ticker_symbol}")
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
        logger.error(f"Error in max pain analysis for {ticker_symbol}: {str(e)}")
        raise ValueError(f"Error analyzing {ticker_symbol}: {str(e)}")

@router.get("/options/{symbol}", response_model=MaxPainResponse)
async def get_options_analysis(
    symbol: str = Path(..., description="Stock symbol (e.g., AAPL, MSFT)"),
    expiration_date: Optional[str] = Query(None, description="Specific expiration date (YYYY-MM-DD)")
):
    """Get options analysis for a specific stock based on max pain theory"""
    logger.info(f"Received request for options analysis - Symbol: {symbol}, Expiration: {expiration_date}")
    try:
        # Log environment information
        logger.debug("Environment Information:")
        logger.debug(f"Python version: {sys.version}")
        logger.debug(f"yfinance version: {yf.__version__}")
        logger.debug(f"pandas version: {pd.__version__}")
        logger.debug(f"numpy version: {np.__version__}")
        logger.debug(f"Current time: {datetime.now()}")
        logger.debug(f"Timezone: {datetime.now().astimezone().tzinfo}")
        
        # Add validation for the symbol
        if not symbol or not isinstance(symbol, str):
            logger.error(f"Invalid symbol format: {symbol}")
            raise HTTPException(status_code=400, detail="Invalid symbol format")
            
        # Add delay for rate limiting
        await asyncio.sleep(1)  # Add a small delay between requests
        
        options_data = get_best_options_by_max_pain(symbol, expiration_date)
        logger.info(f"Successfully completed options analysis for {symbol}")
        return MaxPainResponse(**options_data)
    except ValueError as e:
        logger.error(f"ValueError in options analysis for {symbol}: {str(e)}")
        logger.error(f"Error traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error in options analysis endpoint for {symbol}: {str(e)}")
        logger.error(f"Error type: {type(e).__name__}")
        logger.error(f"Error traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))

async def process_stock(stock_data):
    """Process a single stock's options data"""
    ticker = stock_data['Ticker']
    logger.info(f"Processing stock: {ticker}")
    try:
        # First check if options are available
        try:
            ticker_obj = get_cached_ticker(ticker)
            options = ticker_obj.options
            if not options:
                logger.warning(f"No options available for {ticker}, skipping")
                return None
        except ValueError as e:
            logger.warning(f"Skipping {ticker} due to no options data")
            return None
        except Exception as e:
            logger.error(f"Unexpected error checking options for {ticker}: {str(e)}")
            return None
        
        # If we get here, options are available, proceed with analysis
        try:
            options_data = get_best_options_by_max_pain(ticker)
        except Exception as e:
            logger.error(f"Error analyzing {ticker}: {str(e)}")
            return None
        
        def clean_nan_values(data):
            if isinstance(data, dict):
                return {k: clean_nan_values(v) for k, v in data.items()}
            elif isinstance(data, list):
                return [clean_nan_values(v) for v in data]
            elif isinstance(data, (int, float)) and np.isnan(data):
                return 0
            return data
        
        options_data = clean_nan_values(options_data)
        
        opportunity_score = (
            options_data['best_call']['profitability_score'] * 0.5 +
            options_data['best_put']['profitability_score'] * 0.5
        )
        logger.debug(f"Calculated opportunity score for {ticker}: {opportunity_score:.2f}")
        
        logger.info(f"Successfully processed stock: {ticker}")
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
        logger.error(f"Error processing {ticker}: {str(e)}")
        return None

@router.get("/opportunities", response_model=MaxPainOpportunitiesResponse)
async def get_options_opportunities(
    limit: int = Query(100, description="Number of high volume stocks to analyze", ge=1, le=100),
    top_n: int = Query(10, description="Number of best opportunities to return", ge=1, le=20)
):
    """Get the best options opportunities from high volume stocks"""
    logger.info(f"Received request for opportunities - Limit: {limit}, Top N: {top_n}")
    try:
        start_time = time.time()
        
        # Get cached Finviz data with error handling
        try:
            df = get_cached_finviz_data(limit)
            logger.debug(f"Using cached Finviz data with {len(df)} stocks")
        except Exception as e:
            logger.error(f"Failed to get Finviz data: {str(e)}")
            raise HTTPException(
                status_code=503,
                detail="Service temporarily unavailable. Please try again later."
            )
        
        if df.empty:
            raise HTTPException(
                status_code=404,
                detail="No stock data available at this time."
            )
        
        # Pre-filter stocks that likely don't have options
        filtered_stocks = []
        for stock in df.to_dict('records'):
            try:
                ticker = stock['Ticker']
                ticker_obj = get_cached_ticker(ticker)
                if ticker_obj.options:  # If options are available
                    filtered_stocks.append(stock)
                else:
                    logger.debug(f"Skipping {ticker} - no options available")
            except Exception as e:
                logger.debug(f"Skipping {ticker} - error checking options: {str(e)}")
        
        if not filtered_stocks:
            raise HTTPException(
                status_code=404,
                detail="No stocks with options data available at this time."
            )
        
        logger.debug(f"After filtering, {len(filtered_stocks)} stocks have options data")
        
        # Process stocks with error handling
        try:
            with ThreadPoolExecutor(max_workers=min(10, len(filtered_stocks))) as executor:
                logger.debug(f"Processing {len(filtered_stocks)} stocks in parallel")
                tasks = [process_stock(stock) for stock in filtered_stocks]
                results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Filter out exceptions and None results
            valid_results = []
            for result in results:
                if isinstance(result, Exception):
                    logger.error(f"Error processing stock: {str(result)}")
                    continue
                if result is not None:
                    valid_results.append(result)
            
            if not valid_results:
                raise HTTPException(
                    status_code=404,
                    detail="No valid opportunities found at this time."
                )
            
            valid_results.sort(key=lambda x: x.opportunity_score, reverse=True)
            logger.debug(f"Found {len(valid_results)} valid opportunities")
            
            top_opportunities = valid_results[:top_n]
            
            end_time = time.time()
            logger.info(f"Completed opportunities analysis in {end_time - start_time:.2f} seconds")
            
            return MaxPainOpportunitiesResponse(
                opportunities=top_opportunities,
                total_opportunities=len(top_opportunities)
            )
        except Exception as e:
            logger.error(f"Error processing stocks: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail="Error processing stock data. Please try again later."
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in opportunities endpoint: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred. Please try again later."
        ) 
