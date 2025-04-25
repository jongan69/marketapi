from fastapi import APIRouter, Query
from finvizfinance.insider import Insider
import pandas as pd
from logger import log_error, log_endpoint_access, log_performance
import time

router = APIRouter(prefix="/insider", tags=["Insider Trading"])

@router.get("", tags=["Insider Trading"])
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
    start_time = time.time()
    log_endpoint_access("/insider", option=option)
    
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
        
        # Log performance
        duration = time.time() - start_time
        log_performance("get_insider_trading", duration, option=option)
        
        return df.to_dict('records')
    except Exception as e:
        log_error(e, {"option": option})
        return [] 