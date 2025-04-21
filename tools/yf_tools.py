from typing import Optional
import yfinance as yf
import pandas as pd
from colorama import Fore
import requests

_cached_tickers = None

def stock_price(stock_ticker: str) -> pd.Series:
    """
    Get the stock price for a given stock ticker.
    
    Args:
        stock_ticker (str): The stock ticker symbol.
        
    Returns:
        pd.Series: The stock price.
    """
    dat = yf.Ticker(stock_ticker)
    prices = dat.history(period='1mo')
    closes = prices['Close']
    print(Fore.YELLOW + str(closes))
    print('stock_price type', type(closes))
    return closes

# Get company background info
def stock_info(stock_ticker: str) -> dict:
    """
    Get company background information for a given stock ticker.
    
    Args:
        stock_ticker (str): The stock ticker symbol.
        
    Returns:
        dict: Company background information.
    """
    dat = yf.Ticker(stock_ticker)
    print('stock_info type', type(dat.info))
    return dat

def balance_sheet(stock_ticker: str) -> pd.DataFrame:
    """
    Get the balance sheet for a given stock ticker.
    
    Args:
        stock_ticker (str): The stock ticker symbol.
        
    Returns:
        pd.DataFrame: Balance sheet information.
    """
    dat = yf.Ticker(stock_ticker)
    print('balance_sheet type', type(dat.balance_sheet))
    return dat.balance_sheet

def cash_flow(stock_ticker: str) -> pd.DataFrame:
    """
    Get the cash flow statement for a given stock ticker.
    
    Args:
        stock_ticker (str): The stock ticker symbol.
        
    Returns:
        pd.DataFrame: Cash flow statement information.
    """
    dat = yf.Ticker(stock_ticker)
    print('cash_flow type', type(dat.cash_flow))
    return dat.cash_flow

def stock_dividends(stock_ticker: str) -> pd.Series:
    """
    Get the stock dividends for a given stock ticker.

    Args:
        stock_ticker (str): The stock ticker symbol.

    Returns:
        pd.Series: Stock dividends information.
    """
    dat = yf.Ticker(stock_ticker)
    print('stock_dividends type', type(dat.dividends))
    return dat.dividends

def get_income_statement(stock_ticker: str, frequency: str = 'quarterly') -> pd.DataFrame:
    """
    Get the income statement for a given stock ticker.
    
    Args:
        stock_ticker (str): The stock ticker symbol.
        frequency (str): 'quarterly' or 'annual' for the frequency of data.
        
    Returns:
        pd.DataFrame: Income statement information at the specified frequency.
    """
    dat = yf.Ticker(stock_ticker)
    if frequency.lower() == 'quarterly':
        statement = dat.quarterly_income_stmt
        print('quarterly income statement type', type(statement))
    else:
        statement = dat.financials
        print('annual income statement type', type(statement))
    return statement

# Get financial metrics (quarterly)
def financial_metrics(stock_ticker: str) -> pd.DataFrame:
    """
    Get financial metrics (quarterly) for a given stock ticker.
    
    Args:
        stock_ticker (str): The stock ticker symbol.
        
    Returns:
        pd.DataFrame: Financial metrics (quarterly) information.
    """
    dat = yf.Ticker(stock_ticker)
    financials = dat.quarterly_financials
    print('financial_metrics type', type(financials))
    return financials

def company_news(stock_ticker: str) -> pd.DataFrame:
    """ 
    Get company news for a given stock ticker.
    
    Args:
        stock_ticker (str): The stock ticker symbol.
        
    Returns:
        pd.DataFrame: Company news information.
    """
    dat = yf.Ticker(stock_ticker)
    news = dat.news
    print('company_news type', type(news))
    return news

def insider_trades(stock_ticker: str) -> pd.DataFrame:
    """
    Get insider trades (transactions) for a given stock ticker.
    
    Args:
        stock_ticker (str): The stock ticker symbol.
        
    Returns:
        pd.DataFrame: Insider trades information.
    """
    dat = yf.Ticker(stock_ticker)
    trades = dat.get_insider_transactions()
    print('insider_trades type', type(trades))
    return trades

# Convert price history to DataFrame
def prices_to_df(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Convert price history to a DataFrame.
    
    Args:
        prices: Price history data.
        
    Returns:
        pd.DataFrame: Price history data as a DataFrame.
    """
    df = prices.copy()
    df['Date'] = pd.to_datetime(df.index)
    numeric_cols = ['Open', 'Close', 'High', 'Low', 'Volume']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df.set_index("Date", inplace=True)
    return df

# Get custom price range data
def get_price_data(stock_ticker: str, start_date: str = None, end_date: str = None) -> pd.DataFrame:
    """
    Get price data for a given stock ticker within a custom date range.
    If no dates provided, returns last month's data.
    
    Args:
        stock_ticker (str): The stock ticker symbol.
        start_date (str, optional): The start date in YYYY-MM-DD format. Defaults to 1 month ago.
        end_date (str, optional): The end date in YYYY-MM-DD format. Defaults to today.

    Returns:
        pd.DataFrame: Price data for the given date range.
    """
    dat = yf.Ticker(stock_ticker)
    if start_date and end_date:
        prices = dat.history(start=start_date, end=end_date)
    else:
        prices = dat.history(period='1mo')
    return prices_to_df(prices)

def get_market_cap(
    ticker: str
) -> Optional[float]:
    """Fetch market capitalization data."""
    try:
        # Get data from yfinance
        yf_ticker = yf.Ticker(ticker)
        info = yf_ticker.info
        
        if "marketCap" in info:
            market_cap = float(info["marketCap"])
            
            # Cache the result
            return market_cap
            
        return None
        
    except Exception as e:
        print(f"Error fetching market cap for {ticker}: {e}")
        return None
    
# Calculate Return on Equity (ROE)
def calculate_roe(stock_ticker: str) -> pd.Series:
    """
    Calculate Return on Equity (ROE) for a given stock ticker.
    
    Args:
        stock_ticker (str): The stock ticker symbol.
        
    Returns:
        pd.Series: ROE values for each available period.
    """
    dat = yf.Ticker(stock_ticker)
    
    # Get income statement and balance sheet
    income_stmt = dat.financials
    balance_sheet = dat.balance_sheet
    
    try:
        # Extract Net Income from the income statement
        net_income = income_stmt.loc['Net Income']
        
        # Extract Stockholders Equity from the balance sheet
        # Try different possible column names
        if 'Total Stockholder Equity' in balance_sheet.index:
            shareholders_equity = balance_sheet.loc['Total Stockholder Equity']
        elif 'Stockholders Equity' in balance_sheet.index:
            shareholders_equity = balance_sheet.loc['Stockholders Equity']
        else:
            # If neither exists, try to find a similar column
            equity_rows = [row for row in balance_sheet.index if 'equity' in row.lower()]
            if equity_rows:
                shareholders_equity = balance_sheet.loc[equity_rows[0]]
            else:
                raise ValueError("Could not find stockholders equity in balance sheet")
        
        # Calculate ROE
        roe = net_income / shareholders_equity
        
        return roe
    except Exception as e:
        print(f"Error calculating ROE: {str(e)}")
        return pd.Series()
    
def get_research_and_development(stock_ticker: str) -> pd.Series:
    """
    Get research and development expenses for a given stock ticker.
    """
    dat = yf.Ticker(stock_ticker).financials.loc["Research And Development"]
    return dat

# Calculate Current Ratio
def calculate_current_ratio(stock_ticker: str) -> pd.Series:
    """
    Calculate Current Ratio for a given stock ticker.
    
    Args:
        stock_ticker (str): The stock ticker symbol.
        
    Returns:
        pd.Series: Current Ratio values for each available period.
    """
    dat = yf.Ticker(stock_ticker)
    balance_sheet = dat.balance_sheet
    
    try:
        # Extract Total Current Assets and Total Current Liabilities
        # Try different possible field names
        if 'Total Current Assets' in balance_sheet.index:
            current_assets = balance_sheet.loc['Total Current Assets']
        elif 'Current Assets' in balance_sheet.index:
            current_assets = balance_sheet.loc['Current Assets']
        else:
            # If neither exists, try to find a similar column
            asset_rows = [row for row in balance_sheet.index if 'current asset' in row.lower()]
            if asset_rows:
                current_assets = balance_sheet.loc[asset_rows[0]]
            else:
                raise ValueError("Could not find current assets in balance sheet")
        
        if 'Total Current Liabilities' in balance_sheet.index:
            current_liabilities = balance_sheet.loc['Total Current Liabilities']
        elif 'Current Liabilities' in balance_sheet.index:
            current_liabilities = balance_sheet.loc['Current Liabilities']
        else:
            # If neither exists, try to find a similar column
            liability_rows = [row for row in balance_sheet.index if 'current liab' in row.lower()]
            if liability_rows:
                current_liabilities = balance_sheet.loc[liability_rows[0]]
            else:
                raise ValueError("Could not find current liabilities in balance sheet")
        
        # Calculate Current Ratio
        current_ratio = current_assets / current_liabilities
        
        return current_ratio
    except Exception as e:
        print(f"Error calculating Current Ratio: {str(e)}")
        return pd.Series()

# Calculate Debt-to-Equity Ratio
def calculate_debt_to_equity(stock_ticker: str) -> pd.Series:
    """
    Calculate Debt-to-Equity Ratio for a given stock ticker.
    
    Args:
        stock_ticker (str): The stock ticker symbol.
        
    Returns:
        pd.Series: Debt-to-Equity Ratio values for each available period.
    """
    dat = yf.Ticker(stock_ticker)
    balance_sheet = dat.balance_sheet
    
    try:
        # Extract Total Liabilities
        # Try different possible field names
        if 'Total Liab' in balance_sheet.index:
            total_liabilities = balance_sheet.loc['Total Liab']
        elif 'Total Liabilities' in balance_sheet.index:
            total_liabilities = balance_sheet.loc['Total Liabilities']
        else:
            # If neither exists, try to find a similar column
            liability_rows = [row for row in balance_sheet.index if 'total liab' in row.lower()]
            if liability_rows:
                total_liabilities = balance_sheet.loc[liability_rows[0]]
            else:
                raise ValueError("Could not find total liabilities in balance sheet")
        
        # Extract Stockholders Equity
        if 'Total Stockholder Equity' in balance_sheet.index:
            shareholders_equity = balance_sheet.loc['Total Stockholder Equity']
        elif 'Stockholders Equity' in balance_sheet.index:
            shareholders_equity = balance_sheet.loc['Stockholders Equity']
        else:
            # If neither exists, try to find a similar column
            equity_rows = [row for row in balance_sheet.index if 'equity' in row.lower()]
            if equity_rows:
                shareholders_equity = balance_sheet.loc[equity_rows[0]]
            else:
                raise ValueError("Could not find stockholders equity in balance sheet")
        
        # Calculate Debt-to-Equity Ratio
        debt_to_equity = total_liabilities / shareholders_equity
        
        return debt_to_equity
    except Exception as e:
        print(f"Error calculating Debt-to-Equity Ratio: {str(e)}")
        return pd.Series()

# Calculate all financial ratios
def calculate_financial_ratios(stock_ticker: str) -> dict:
    """
    Calculate all financial ratios for a given stock ticker.
    
    Args:
        stock_ticker (str): The stock ticker symbol.
        
    Returns:
        dict: Dictionary containing all calculated financial ratios.
    """
    ratios = {}
    
    try:
        # Calculate ROE
        roe = calculate_roe(stock_ticker)
        if not roe.empty:
            ratios['ROE'] = roe
        
        # Calculate Current Ratio
        current_ratio = calculate_current_ratio(stock_ticker)
        if not current_ratio.empty:
            ratios['Current Ratio'] = current_ratio
        
        # Calculate Debt-to-Equity Ratio
        debt_to_equity = calculate_debt_to_equity(stock_ticker)
        if not debt_to_equity.empty:
            ratios['Debt-to-Equity'] = debt_to_equity
        
        return ratios
    except Exception as e:
        print(f"Error calculating financial ratios: {str(e)}")
        return {}

def fetch_and_cache_stock_tickers() -> list:
    """Fetch stock tickers from investassist.app and cache the result."""
    print("\n=== Fetching stock tickers from API ===")
    global _cached_tickers
    if _cached_tickers is None:
        print("No cached tickers found, fetching from API...")
        url = "https://investassist.app/api/tickers"
        try:
            print(f"Making request to {url}")
            response = requests.get(url)
            if response.status_code == 200:
                print("Successfully retrieved tickers from API")
                _cached_tickers = response.json()
                print(f"Cached {len(_cached_tickers)} tickers")
            else:
                print(f"WARNING: API request failed with status code {response.status_code}")
                _cached_tickers = []
        except Exception as e:
            print(f"ERROR: Failed to fetch tickers from API: {str(e)}")
            _cached_tickers = []
    else:
        print(f"Using {len(_cached_tickers)} cached tickers")
    return _cached_tickers

# Calculate Profit Margin
def calculate_profit_margin(stock_ticker: str) -> pd.Series:
    """
    Calculate Profit Margin for a given stock ticker.
    
    Args:
        stock_ticker (str): The stock ticker symbol.
        
    Returns:
        pd.Series: Profit Margin values for each available period.
    """
    dat = yf.Ticker(stock_ticker)
    income_stmt = dat.financials
    
    try:
        # Extract Net Income and Total Revenue
        net_income = income_stmt.loc['Net Income']
        total_revenue = income_stmt.loc['Total Revenue']
        
        # Calculate Profit Margin
        profit_margin = net_income / total_revenue
        
        return profit_margin
    except Exception as e:
        print(f"Error calculating Profit Margin: {str(e)}")
        return pd.Series()

# Calculate Operating Margin
def calculate_operating_margin(stock_ticker: str) -> pd.Series:
    """
    Calculate Operating Margin for a given stock ticker.
    
    Args:
        stock_ticker (str): The stock ticker symbol.
        
    Returns:
        pd.Series: Operating Margin values for each available period.
    """
    dat = yf.Ticker(stock_ticker)
    income_stmt = dat.financials
    
    try:
        # Extract Operating Income and Total Revenue
        operating_income = income_stmt.loc['Operating Income']
        total_revenue = income_stmt.loc['Total Revenue']
        
        # Calculate Operating Margin
        operating_margin = operating_income / total_revenue
        
        return operating_margin
    except Exception as e:
        print(f"Error calculating Operating Margin: {str(e)}")
        return pd.Series()

# Calculate Gross Margin
def calculate_gross_margin(stock_ticker: str) -> pd.Series:
    """
    Calculate Gross Margin for a given stock ticker.
    
    Args:
        stock_ticker (str): The stock ticker symbol.
        
    Returns:
        pd.Series: Gross Margin values for each available period.
    """
    dat = yf.Ticker(stock_ticker)
    income_stmt = dat.financials
    
    try:
        # Extract Gross Profit and Total Revenue
        gross_profit = income_stmt.loc['Gross Profit']
        total_revenue = income_stmt.loc['Total Revenue']
        
        # Calculate Gross Margin
        gross_margin = gross_profit / total_revenue
        
        return gross_margin
    except Exception as e:
        print(f"Error calculating Gross Margin: {str(e)}")
        return pd.Series()

# Calculate Earnings Growth
def calculate_earnings_growth(stock_ticker: str) -> pd.Series:
    """
    Calculate Earnings Growth for a given stock ticker.
    
    Args:
        stock_ticker (str): The stock ticker symbol.
        
    Returns:
        pd.Series: Earnings Growth values for each available period.
    """
    dat = yf.Ticker(stock_ticker)
    income_stmt = dat.financials
    
    try:
        # Extract Net Income
        net_income = income_stmt.loc['Net Income']
        
        # Calculate year-over-year growth with fill_method=None to avoid the warning
        earnings_growth = net_income.pct_change(fill_method=None)
        
        return earnings_growth
    except Exception as e:
        print(f"Error calculating Earnings Growth: {str(e)}")
        return pd.Series()

# Calculate Dividend Yield
def calculate_dividend_yield(stock_ticker: str) -> float:
    """
    Calculate Dividend Yield for a given stock ticker.
    
    Args:
        stock_ticker (str): The stock ticker symbol.
        
    Returns:
        float: Dividend Yield value.
    """
    dat = yf.Ticker(stock_ticker)
    
    try:
        # Get dividend yield from info
        dividend_yield = dat.info.get('dividendYield', 0)
        
        # Convert to percentage if not already
        if dividend_yield and dividend_yield < 1:
            dividend_yield = dividend_yield * 100
            
        return dividend_yield
    except Exception as e:
        print(f"Error calculating Dividend Yield: {str(e)}")
        return 0.0

# Calculate Outstanding Shares
def get_outstanding_shares(stock_ticker: str) -> int:
    """
    Get Outstanding Shares for a given stock ticker.
    
    Args:
        stock_ticker (str): The stock ticker symbol.
        
    Returns:
        int: Number of outstanding shares.
    """
    dat = yf.Ticker(stock_ticker)
    
    try:
        # Get shares outstanding from info
        shares = dat.info.get('sharesOutstanding', 0)
        return int(shares)
    except Exception as e:
        print(f"Error getting Outstanding Shares: {str(e)}")
        return 0

# Calculate Free Cash Flow
def calculate_free_cash_flow(stock_ticker: str) -> pd.Series:
    """
    Calculate Free Cash Flow for a given stock ticker.
    
    Args:
        stock_ticker (str): The stock ticker symbol.
        
    Returns:
        pd.Series: Free Cash Flow values for each available period.
    """
    dat = yf.Ticker(stock_ticker)
    cash_flow_stmt = dat.cash_flow
    
    try:
        # Extract Operating Cash Flow and Capital Expenditure
        operating_cash_flow = cash_flow_stmt.loc['Operating Cash Flow']
        capital_expenditure = cash_flow_stmt.loc['Capital Expenditure']
        
        # Calculate Free Cash Flow
        free_cash_flow = operating_cash_flow + capital_expenditure  # Capital expenditure is negative
        
        return free_cash_flow
    except Exception as e:
        print(f"Error calculating Free Cash Flow: {str(e)}")
        return pd.Series()

# Calculate Operating Cash Flow
def get_operating_cash_flow(stock_ticker: str) -> pd.Series:
    """
    Get Operating Cash Flow for a given stock ticker.
    
    Args:
        stock_ticker (str): The stock ticker symbol.
        
    Returns:
        pd.Series: Operating Cash Flow values for each available period.
    """
    dat = yf.Ticker(stock_ticker)
    cash_flow_stmt = dat.cash_flow
    
    try:
        # Extract Operating Cash Flow
        operating_cash_flow = cash_flow_stmt.loc['Operating Cash Flow']
        
        return operating_cash_flow
    except Exception as e:
        print(f"Error getting Operating Cash Flow: {str(e)}")
        return pd.Series()

# Calculate EBITDA
def calculate_ebitda(stock_ticker: str) -> pd.Series:
    """
    Calculate EBITDA for a given stock ticker.
    
    Args:
        stock_ticker (str): The stock ticker symbol.
        
    Returns:
        pd.Series: EBITDA values for each available period.
    """
    dat = yf.Ticker(stock_ticker)
    income_stmt = dat.financials
    
    try:
        # Extract Operating Income, Depreciation, and Amortization
        operating_income = income_stmt.loc['Operating Income']
        
        # Try to get Depreciation and Amortization
        try:
            depreciation = income_stmt.loc['Depreciation']
        except:
            depreciation = pd.Series(0, index=operating_income.index)
            
        try:
            amortization = income_stmt.loc['Amortization']
        except:
            amortization = pd.Series(0, index=operating_income.index)
        
        # Calculate EBITDA
        ebitda = operating_income + depreciation + amortization
        
        return ebitda
    except Exception as e:
        print(f"Error calculating EBITDA: {str(e)}")
        return pd.Series()

# Get Research and Development
def get_research_and_development(stock_ticker: str) -> pd.Series:
    """
    Get Research and Development expenses for a given stock ticker.
    
    Args:
        stock_ticker (str): The stock ticker symbol.
        
    Returns:
        pd.Series: R&D values for each available period.
    """
    dat = yf.Ticker(stock_ticker)
    income_stmt = dat.financials
    
    try:
        # Try different possible field names for R&D
        if 'Research Development' in income_stmt.index:
            r_and_d = income_stmt.loc['Research Development']
        elif 'Research and Development' in income_stmt.index:
            r_and_d = income_stmt.loc['Research and Development']
        elif 'Research & Development' in income_stmt.index:
            r_and_d = income_stmt.loc['Research & Development']
        else:
            # If none exist, try to find a similar column
            rd_rows = [row for row in income_stmt.index if 'research' in row.lower() and 'development' in row.lower()]
            if rd_rows:
                r_and_d = income_stmt.loc[rd_rows[0]]
            else:
                raise ValueError("Could not find research and development in income statement")
        
        return r_and_d
    except Exception as e:
        print(f"Error getting Research and Development: {str(e)}")
        return pd.Series()

# Get Selling, General and Administrative
def get_selling_general_and_admin(stock_ticker: str) -> pd.Series:
    """
    Get Selling, General and Administrative expenses for a given stock ticker.
    
    Args:
        stock_ticker (str): The stock ticker symbol.
        
    Returns:
        pd.Series: SG&A values for each available period.
    """
    dat = yf.Ticker(stock_ticker)
    income_stmt = dat.financials
    
    try:
        # Extract Selling, General and Administrative
        sga = income_stmt.loc['Selling General And Administration']
        
        return sga
    except Exception as e:
        print(f"Error getting Selling, General and Administrative: {str(e)}")
        return pd.Series()

# Get Total Assets
def get_total_assets(stock_ticker: str) -> pd.Series:
    """
    Get Total Assets for a given stock ticker.
    
    Args:
        stock_ticker (str): The stock ticker symbol.
        
    Returns:
        pd.Series: Total Assets values for each available period.
    """
    dat = yf.Ticker(stock_ticker)
    balance_sheet = dat.balance_sheet
    
    try:
        # Extract Total Assets
        total_assets = balance_sheet.loc['Total Assets']
        
        return total_assets
    except Exception as e:
        print(f"Error getting Total Assets: {str(e)}")
        return pd.Series()

# Get Total Liabilities
def get_total_liabilities(stock_ticker: str) -> pd.Series:
    """
    Get Total Liabilities for a given stock ticker.
    
    Args:
        stock_ticker (str): The stock ticker symbol.
        
    Returns:
        pd.Series: Total Liabilities values for each available period.
    """
    dat = yf.Ticker(stock_ticker)
    balance_sheet = dat.balance_sheet
    
    try:
        # Try different possible field names
        if 'Total Liab' in balance_sheet.index:
            total_liabilities = balance_sheet.loc['Total Liab']
        elif 'Total Liabilities' in balance_sheet.index:
            total_liabilities = balance_sheet.loc['Total Liabilities']
        else:
            # If neither exists, try to find a similar column
            liability_rows = [row for row in balance_sheet.index if 'total liab' in row.lower()]
            if liability_rows:
                total_liabilities = balance_sheet.loc[liability_rows[0]]
            else:
                raise ValueError("Could not find total liabilities in balance sheet")
        
        return total_liabilities
    except Exception as e:
        print(f"Error getting Total Liabilities: {str(e)}")
        return pd.Series()

# Get Current Assets
def get_current_assets(stock_ticker: str) -> pd.Series:
    """
    Get Current Assets for a given stock ticker.
    
    Args:
        stock_ticker (str): The stock ticker symbol.
        
    Returns:
        pd.Series: Current Assets values for each available period.
    """
    dat = yf.Ticker(stock_ticker)
    balance_sheet = dat.balance_sheet
    
    try:
        # Try different possible field names
        if 'Total Current Assets' in balance_sheet.index:
            current_assets = balance_sheet.loc['Total Current Assets']
        elif 'Current Assets' in balance_sheet.index:
            current_assets = balance_sheet.loc['Current Assets']
        else:
            # If neither exists, try to find a similar column
            asset_rows = [row for row in balance_sheet.index if 'current asset' in row.lower()]
            if asset_rows:
                current_assets = balance_sheet.loc[asset_rows[0]]
            else:
                raise ValueError("Could not find current assets in balance sheet")
        
        return current_assets
    except Exception as e:
        print(f"Error getting Current Assets: {str(e)}")
        return pd.Series()

# Get Current Liabilities
def get_current_liabilities(stock_ticker: str) -> pd.Series:
    """
    Get Current Liabilities for a given stock ticker.
    
    Args:
        stock_ticker (str): The stock ticker symbol.
        
    Returns:
        pd.Series: Current Liabilities values for each available period.
    """
    dat = yf.Ticker(stock_ticker)
    balance_sheet = dat.balance_sheet
    
    try:
        # Try different possible field names
        if 'Total Current Liabilities' in balance_sheet.index:
            current_liabilities = balance_sheet.loc['Total Current Liabilities']
        elif 'Current Liabilities' in balance_sheet.index:
            current_liabilities = balance_sheet.loc['Current Liabilities']
        else:
            # If neither exists, try to find a similar column
            liability_rows = [row for row in balance_sheet.index if 'current liab' in row.lower()]
            if liability_rows:
                current_liabilities = balance_sheet.loc[liability_rows[0]]
            else:
                raise ValueError("Could not find current liabilities in balance sheet")
        
        return current_liabilities
    except Exception as e:
        print(f"Error getting Current Liabilities: {str(e)}")
        return pd.Series()

# Get Cash
def get_cash(stock_ticker: str) -> pd.Series:
    """
    Get Cash for a given stock ticker.
    
    Args:
        stock_ticker (str): The stock ticker symbol.
        
    Returns:
        pd.Series: Cash values for each available period.
    """
    dat = yf.Ticker(stock_ticker)
    balance_sheet = dat.balance_sheet
    
    try:
        # Try different possible field names
        if 'Cash' in balance_sheet.index:
            cash = balance_sheet.loc['Cash']
        elif 'Cash And Cash Equivalents' in balance_sheet.index:
            cash = balance_sheet.loc['Cash And Cash Equivalents']
        else:
            # If neither exists, try to find a similar column
            cash_rows = [row for row in balance_sheet.index if 'cash' in row.lower()]
            if cash_rows:
                cash = balance_sheet.loc[cash_rows[0]]
            else:
                raise ValueError("Could not find cash in balance sheet")
        
        return cash
    except Exception as e:
        print(f"Error getting Cash: {str(e)}")
        return pd.Series()

# Get Inventory
def get_inventory(stock_ticker: str) -> pd.Series:
    """
    Get Inventory for a given stock ticker.
    
    Args:
        stock_ticker (str): The stock ticker symbol.
        
    Returns:
        pd.Series: Inventory values for each available period.
    """
    dat = yf.Ticker(stock_ticker)
    balance_sheet = dat.balance_sheet
    
    try:
        # Extract Inventory
        inventory = balance_sheet.loc['Inventory']
        
        return inventory
    except Exception as e:
        print(f"Error getting Inventory: {str(e)}")
        return pd.Series()

# Get Accounts Receivable
def get_accounts_receivable(stock_ticker: str) -> pd.Series:
    """
    Get Accounts Receivable for a given stock ticker.
    
    Args:
        stock_ticker (str): The stock ticker symbol.
        
    Returns:
        pd.Series: Accounts Receivable values for each available period.
    """
    dat = yf.Ticker(stock_ticker)
    balance_sheet = dat.balance_sheet
    
    try:
        # Try different possible field names
        if 'Net Receivables' in balance_sheet.index:
            accounts_receivable = balance_sheet.loc['Net Receivables']
        elif 'Accounts Receivable' in balance_sheet.index:
            accounts_receivable = balance_sheet.loc['Accounts Receivable']
        else:
            # If neither exists, try to find a similar column
            receivable_rows = [row for row in balance_sheet.index if 'receivable' in row.lower()]
            if receivable_rows:
                accounts_receivable = balance_sheet.loc[receivable_rows[0]]
            else:
                raise ValueError("Could not find accounts receivable in balance sheet")
        
        return accounts_receivable
    except Exception as e:
        print(f"Error getting Accounts Receivable: {str(e)}")
        return pd.Series()

# Get Accounts Payable
def get_accounts_payable(stock_ticker: str) -> pd.Series:
    """
    Get Accounts Payable for a given stock ticker.
    
    Args:
        stock_ticker (str): The stock ticker symbol.
        
    Returns:
        pd.Series: Accounts Payable values for each available period.
    """
    dat = yf.Ticker(stock_ticker)
    balance_sheet = dat.balance_sheet
    
    try:
        # Extract Accounts Payable
        accounts_payable = balance_sheet.loc['Accounts Payable']
        
        return accounts_payable
    except Exception as e:
        print(f"Error getting Accounts Payable: {str(e)}")
        return pd.Series()

# Get Shareholders Equity
def get_shareholders_equity(stock_ticker: str) -> pd.Series:
    """
    Get Shareholders Equity for a given stock ticker.
    
    Args:
        stock_ticker (str): The stock ticker symbol.
        
    Returns:
        pd.Series: Shareholders Equity values for each available period.
    """
    dat = yf.Ticker(stock_ticker)
    balance_sheet = dat.balance_sheet
    
    try:
        # Extract Shareholders Equity
        if 'Total Stockholder Equity' in balance_sheet.index:
            shareholders_equity = balance_sheet.loc['Total Stockholder Equity']
        elif 'Stockholders Equity' in balance_sheet.index:
            shareholders_equity = balance_sheet.loc['Stockholders Equity']
        else:
            # If neither exists, try to find a similar column
            equity_rows = [row for row in balance_sheet.index if 'equity' in row.lower()]
            if equity_rows:
                shareholders_equity = balance_sheet.loc[equity_rows[0]]
            else:
                raise ValueError("Could not find stockholders equity in balance sheet")
        
        return shareholders_equity
    except Exception as e:
        print(f"Error getting Shareholders Equity: {str(e)}")
        return pd.Series()

# Get Revenue
def get_revenue(stock_ticker: str) -> pd.Series:
    """
    Get Revenue for a given stock ticker.
    
    Args:
        stock_ticker (str): The stock ticker symbol.
        
    Returns:
        pd.Series: Revenue values for each available period.
    """
    dat = yf.Ticker(stock_ticker)
    income_stmt = dat.financials
    
    try:
        # Extract Revenue
        revenue = income_stmt.loc['Total Revenue']
        
        return revenue
    except Exception as e:
        print(f"Error getting Revenue: {str(e)}")
        return pd.Series()

# Get Net Income
def get_net_income(stock_ticker: str) -> pd.Series:
    """
    Get Net Income for a given stock ticker.
    
    Args:
        stock_ticker (str): The stock ticker symbol.
        
    Returns:
        pd.Series: Net Income values for each available period.
    """
    dat = yf.Ticker(stock_ticker)
    income_stmt = dat.financials
    
    try:
        # Extract Net Income
        net_income = income_stmt.loc['Net Income']
        
        return net_income
    except Exception as e:
        print(f"Error getting Net Income: {str(e)}")
        return pd.Series()

# Get EPS
def get_eps(stock_ticker: str) -> float:
    """
    Get Earnings Per Share (EPS) for a given stock ticker.
    
    Args:
        stock_ticker (str): The stock ticker symbol.
        
    Returns:
        float: EPS value.
    """
    dat = yf.Ticker(stock_ticker)
    
    try:
        # Get EPS from info
        eps = dat.info.get('trailingEps', 0)
        
        return eps
    except Exception as e:
        print(f"Error getting EPS: {str(e)}")
        return 0.0

# Get Gross Profit
def get_gross_profit(stock_ticker: str) -> pd.Series:
    """
    Get Gross Profit for a given stock ticker.
    
    Args:
        stock_ticker (str): The stock ticker symbol.
        
    Returns:
        pd.Series: Gross Profit values for each available period.
    """
    dat = yf.Ticker(stock_ticker)
    income_stmt = dat.financials
    
    try:
        # Extract Gross Profit
        gross_profit = income_stmt.loc['Gross Profit']
        
        return gross_profit
    except Exception as e:
        print(f"Error getting Gross Profit: {str(e)}")
        return pd.Series()

# Get Operating Income
def get_operating_income(stock_ticker: str) -> pd.Series:
    """
    Get Operating Income for a given stock ticker.
    
    Args:
        stock_ticker (str): The stock ticker symbol.
        
    Returns:
        pd.Series: Operating Income values for each available period.
    """
    dat = yf.Ticker(stock_ticker)
    income_stmt = dat.financials
    
    try:
        # Extract Operating Income
        operating_income = income_stmt.loc['Operating Income']
        
        return operating_income
    except Exception as e:
        print(f"Error getting Operating Income: {str(e)}")
        return pd.Series()

def get_all_fundamental_metrics(stock_ticker: str) -> dict:
    """
    Get all fundamental metrics for a given stock ticker.
    This is the preferred method for getting comprehensive financial data.
    Individual getter functions are available for targeted queries.
    
    Args:
        stock_ticker (str): The stock ticker symbol.
        
    Returns:
        dict: Dictionary containing all fundamental metrics including:
            - Basic metrics (revenue, net income, eps, etc.)
            - Cash flow metrics (free cash flow, operating cash flow)
            - Income statement metrics (ebitda, gross profit, operating income)
            - Balance sheet metrics (assets, liabilities, equity)
            - Financial ratios (margins, ROE, debt ratios, etc.)
    """
    metrics = {}
    dat = yf.Ticker(stock_ticker)
    
    try:
        # Get basic metrics
        income_stmt = dat.financials
        balance_sheet = dat.balance_sheet
        cash_flow_stmt = dat.cash_flow
        
        # Income statement metrics
        metrics['revenue'] = income_stmt.loc['Total Revenue'] if 'Total Revenue' in income_stmt.index else pd.Series()
        metrics['net_income'] = income_stmt.loc['Net Income'] if 'Net Income' in income_stmt.index else pd.Series()
        metrics['eps'] = dat.info.get('trailingEps', 0)
        metrics['gross_profit'] = income_stmt.loc['Gross Profit'] if 'Gross Profit' in income_stmt.index else pd.Series()
        metrics['operating_income'] = income_stmt.loc['Operating Income'] if 'Operating Income' in income_stmt.index else pd.Series()
        
        # Research and admin expenses
        rd_rows = [row for row in income_stmt.index if 'research' in row.lower() and 'development' in row.lower()]
        metrics['research_and_development'] = income_stmt.loc[rd_rows[0]] if rd_rows else pd.Series()
        metrics['selling_general_and_admin'] = income_stmt.loc['Selling General And Administration'] if 'Selling General And Administration' in income_stmt.index else pd.Series()
        
        # Cash flow metrics
        metrics['operating_cash_flow'] = cash_flow_stmt.loc['Operating Cash Flow'] if 'Operating Cash Flow' in cash_flow_stmt.index else pd.Series()
        metrics['capital_expenditure'] = cash_flow_stmt.loc['Capital Expenditure'] if 'Capital Expenditure' in cash_flow_stmt.index else pd.Series()
        if not metrics['operating_cash_flow'].empty and not metrics['capital_expenditure'].empty:
            metrics['free_cash_flow'] = metrics['operating_cash_flow'] + metrics['capital_expenditure']
        
        # Balance sheet metrics
        metrics['total_assets'] = balance_sheet.loc['Total Assets'] if 'Total Assets' in balance_sheet.index else pd.Series()
        metrics['total_liabilities'] = balance_sheet.loc['Total Liab'] if 'Total Liab' in balance_sheet.index else pd.Series()
        metrics['current_assets'] = balance_sheet.loc['Total Current Assets'] if 'Total Current Assets' in balance_sheet.index else pd.Series()
        metrics['current_liabilities'] = balance_sheet.loc['Total Current Liabilities'] if 'Total Current Liabilities' in balance_sheet.index else pd.Series()
        metrics['cash'] = balance_sheet.loc['Cash'] if 'Cash' in balance_sheet.index else pd.Series()
        metrics['inventory'] = balance_sheet.loc['Inventory'] if 'Inventory' in balance_sheet.index else pd.Series()
        metrics['accounts_receivable'] = balance_sheet.loc['Net Receivables'] if 'Net Receivables' in balance_sheet.index else pd.Series()
        metrics['accounts_payable'] = balance_sheet.loc['Accounts Payable'] if 'Accounts Payable' in balance_sheet.index else pd.Series()
        metrics['shareholders_equity'] = balance_sheet.loc['Total Stockholder Equity'] if 'Total Stockholder Equity' in balance_sheet.index else pd.Series()
        
        # Calculate financial ratios
        if not metrics['net_income'].empty and not metrics['shareholders_equity'].empty:
            metrics['return_on_equity'] = metrics['net_income'] / metrics['shareholders_equity']
            
        if not metrics['current_assets'].empty and not metrics['current_liabilities'].empty:
            metrics['current_ratio'] = metrics['current_assets'] / metrics['current_liabilities']
            
        if not metrics['total_liabilities'].empty and not metrics['shareholders_equity'].empty:
            metrics['debt_to_equity'] = metrics['total_liabilities'] / metrics['shareholders_equity']
            
        if not metrics['net_income'].empty and not metrics['revenue'].empty:
            metrics['profit_margin'] = metrics['net_income'] / metrics['revenue']
            
        if not metrics['operating_income'].empty and not metrics['revenue'].empty:
            metrics['operating_margin'] = metrics['operating_income'] / metrics['revenue']
            
        if not metrics['gross_profit'].empty and not metrics['revenue'].empty:
            metrics['gross_margin'] = metrics['gross_profit'] / metrics['revenue']
        
        # Market metrics
        metrics['outstanding_shares'] = dat.info.get('sharesOutstanding', 0)
        metrics['dividend_yield'] = dat.info.get('dividendYield', 0) * 100 if dat.info.get('dividendYield') else 0
        
        # Calculate EBITDA
        if not metrics['operating_income'].empty:
            depreciation = income_stmt.loc['Depreciation'] if 'Depreciation' in income_stmt.index else pd.Series(0, index=metrics['operating_income'].index)
            amortization = income_stmt.loc['Amortization'] if 'Amortization' in income_stmt.index else pd.Series(0, index=metrics['operating_income'].index)
            metrics['ebitda'] = metrics['operating_income'] + depreciation + amortization
        
        # Calculate earnings growth
        if not metrics['net_income'].empty:
            metrics['earnings_growth'] = metrics['net_income'].pct_change(fill_method=None)
        
        return metrics
    except Exception as e:
        print(f"Error getting all fundamental metrics: {str(e)}")
        return {}
    
def safe_get(d, key, default=None):
    if isinstance(d, pd.Series):
        # For pandas.Series, return the value of the specified key or the default
        return d.get(key, default) if d is not None else default
    return d.get(key, default) if d else default

def prepare_financial_line_items(ticker_symbol: str) -> list:
    print(f"Starting prepare_financial_line_items for ticker: {ticker_symbol}")
    ticker = yf.Ticker(ticker_symbol)

    try:
        income_stmt = ticker.get_income_stmt(as_dict=True)
        cashflow_stmt = ticker.get_cashflow(as_dict=True)
        financials = ticker.financials
        
    except Exception as e:
        print(f"Error fetching financial data for {ticker_symbol}: {e}")
        return []
    
    print("\nFetched financial statements:")
    print(f"Income Statement Periods: {list(income_stmt.keys())}")
    print(f"Cash Flow Periods: {list(cashflow_stmt.keys())}")

    periods = income_stmt.keys()  # Use income statement periods as base

    financial_line_items = []

    for period in periods:
        print(f"\nProcessing period: {period}")

        revenue = safe_get(income_stmt.get(period), 'TotalRevenue')
        gross_profit = safe_get(income_stmt.get(period), 'GrossProfit')
        operating_income = safe_get(income_stmt.get(period), 'OperatingIncome')
        rnd = safe_get(financials.get(period), 'Research And Development')
        op_expenses = safe_get(income_stmt.get(period), 'TotalExpenses')

        free_cash_flow = safe_get(cashflow_stmt.get(period), 'FreeCashFlow')
        capex = safe_get(cashflow_stmt.get(period), 'CapitalExpenditure')
        dividends = safe_get(cashflow_stmt.get(period), 'CashDividendsPaid')

        # Calculate margins, handle None (NaN) values
        gross_margin = (gross_profit / revenue) if revenue and gross_profit else None
        operating_margin = (operating_income / revenue) if revenue and operating_income else None

        print(f"Revenue: {revenue}, Gross Profit: {gross_profit}, Gross Margin: {gross_margin}")
        print(f"Operating Income: {operating_income}, Operating Margin: {operating_margin}")
        print(f"Free Cash Flow: {free_cash_flow}, CapEx: {capex}, Dividends: {dividends}")

        item = {
            'period': period,
            'revenue': revenue,
            'gross_margin': gross_margin,
            'operating_expense': op_expenses,
            'research_and_development': rnd,
            'operating_margin': operating_margin,
            'free_cash_flow': free_cash_flow,
            'capital_expenditure': capex,
            'dividends_and_other_cash_distributions': dividends
        }

        # Filter out line items where critical values are NaN
        if pd.isna(revenue) and pd.isna(gross_profit) and pd.isna(operating_income) and pd.isna(free_cash_flow):
            print(f"Skipping empty line item for period {period}")
            continue  # Skip this item as it has no meaningful data

        print(f"Prepared financial line item: {item}")
        financial_line_items.append(item)

    print(f"\nFinished preparing {len(financial_line_items)} financial line items for {ticker_symbol}.")
    return financial_line_items



def test_all_functions(ticker="AAPL"):
    """
    Test all functions in the module with a sample ticker.
    
    Args:
        ticker (str): The stock ticker to use for testing. Defaults to "AAPL".
    
    Returns:
        dict: A dictionary containing the results of each function test.
    """
    print(f"\n=== Testing all functions with ticker: {ticker} ===")
    results = {}
    
    # List of functions to test
    functions_to_test = [
        ('stock_price', lambda: stock_price(ticker)),
        ('balance_sheet', lambda: balance_sheet(ticker)),
        ('cash_flow', lambda: cash_flow(ticker)),
        ('stock_dividends', lambda: stock_dividends(ticker)),
        ('get_income_statement_quarterly', lambda: get_income_statement(ticker, 'quarterly')),
        ('get_income_statement_annual', lambda: get_income_statement(ticker, 'annual')),
        ('insider_trades', lambda: insider_trades(ticker)),
        ('company_news', lambda: company_news(ticker)),
        ('get_price_data', lambda: get_price_data(ticker)),
        ('fetch_and_cache_stock_tickers', lambda: fetch_and_cache_stock_tickers()),
        ('get_all_fundamental_metrics', lambda: get_all_fundamental_metrics(ticker))
    ]
    
    for func_name, func in functions_to_test:
        try:
            print(f"Testing {func_name}...")
            results[func_name] = func()
            print(f"✓ {func_name} test passed")
        except Exception as e:
            print(f"✗ {func_name} test failed: {str(e)}")
            results[func_name] = f"Error: {str(e)}"
    
    print("\n=== All tests completed ===")
    return results

if __name__ == "__main__":
    test_all_functions()