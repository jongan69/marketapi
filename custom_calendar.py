import re
import pandas as pd
import aiohttp
from bs4 import BeautifulSoup

class CustomCalendar:
    """Custom Calendar implementation for FinViz economic calendar"""
    
    def __init__(self):
        """Initialize the calendar"""
        self.url = "https://finviz.com/calendar.ashx"
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Cache-Control': 'max-age=0',
            'Cookie': 'customTable=0; screenerView=1; activeTab=1'
        }
    
    async def calendar(self):
        """Get economic calendar table.
        
        Returns:
            df(pandas.DataFrame): economic calendar table
            
        Raises:
            Exception: If there's an error fetching or processing the calendar data
        """
        try:
            print(f"Making request to URL: {self.url}")
            
            # Make the request with headers to avoid being blocked
            async with aiohttp.ClientSession() as session:
                try:
                    async with session.get(self.url, headers=self.headers, timeout=10) as response:
                        print(f"Response status: {response.status}")
                        
                        if response.status != 200:
                            error_text = await response.text()
                            print(f"Error response body: {error_text}")
                            if response.status == 403:
                                raise Exception("Access forbidden - FinViz is blocking our requests. Please try again later or contact support if the issue persists.")
                            elif response.status == 429:
                                raise Exception("Rate limited - Too many requests to FinViz. Please wait a few minutes before trying again.")
                            else:
                                response.raise_for_status()
                            
                        html = await response.text()
                        print(f"Received HTML response of length: {len(html)}")
                        
                        if not html or len(html.strip()) == 0:
                            raise Exception("Received empty response from FinViz - the service may be temporarily unavailable")
                            
                except aiohttp.ClientError as e:
                    print(f"HTTP request failed: {e}")
                    if isinstance(e, aiohttp.ClientTimeout):
                        raise Exception("Request to FinViz timed out - please try again")
                    elif isinstance(e, aiohttp.ClientConnectorError):
                        raise Exception("Could not connect to FinViz - please check your internet connection")
                    else:
                        raise Exception(f"Failed to fetch calendar data: {str(e)}")
            
            # Parse the HTML
            soup = BeautifulSoup(html, 'html.parser')
            
            # Find all calendar tables
            tables = soup.find_all("table", class_="calendar_table")
            print(f"Found {len(tables)} calendar tables")
            
            # If no tables found with the expected class, try to find any table
            if not tables:
                print("No tables found with class 'calendar_table', trying to find any table...")
                tables = soup.find_all("table")
                print(f"Found {len(tables)} tables without class filter")
                
                if not tables:
                    raise Exception("No calendar tables found in the response - FinViz page structure may have changed")
            
            frame = []
            current_date = None
            
            for table in tables:
                # Find the date in the header
                header = table.find('thead')
                if header:
                    date_cell = header.find('th')
                    if date_cell:
                        date_text = date_cell.text.strip()
                        # Extract date using regex
                        date_match = re.search(r'(Mon|Tue|Wed|Thu|Fri|Sat|Sun)\s+(Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec|Jan|Feb|Mar)\s+(\d+)', date_text)
                        if date_match:
                            current_date = date_text
                            print(f"Processing date: {current_date}")
                
                # Process rows
                rows = table.find_all('tr', class_="styled-row")
                print(f"Found {len(rows)} rows for date {current_date}")
                
                # If no rows found with the expected class, try to find any row
                if not rows:
                    print("No rows found with class 'styled-row', trying to find any row...")
                    rows = table.find_all('tr')
                    print(f"Found {len(rows)} rows without class filter")
                
                for row in rows:
                    cols = row.find_all('td')
                    if len(cols) >= 8:  # We need at least Time, Release, Impact, For, Actual, Expected, Prior
                        try:
                            # Extract time
                            time_text = cols[0].text.strip()
                            
                            # Extract impact from image if available
                            impact = "unknown"
                            impact_img = cols[3].find('img')
                            if impact_img and 'src' in impact_img.attrs:
                                impact_match = re.findall(r"impact_(.*)\.gif", impact_img['src'])
                                if impact_match:
                                    impact = impact_match[0]
                            
                            # Create a record for this economic event
                            info_dict = {
                                "Date": current_date,
                                "Time": time_text,
                                "Datetime": f"{current_date}, {time_text}" if current_date else time_text,
                                "Release": cols[2].text.strip(),
                                "Impact": impact,
                                "For": cols[4].text.strip(),
                                "Actual": cols[5].text.strip(),
                                "Expected": cols[6].text.strip(),
                                "Prior": cols[7].text.strip()
                            }
                            
                            frame.append(info_dict)
                            print(f"Added event: {info_dict['Release']} at {info_dict['Time']} on {info_dict['Date']}")
                        except Exception as e:
                            print(f"Error processing row: {e}")
                            continue
            
            # Create DataFrame from the collected data
            if frame:
                df = pd.DataFrame(frame)
                
                # Clean up the data
                df = df.replace('NA', None)
                df = df.replace('-', None)
                df = df.replace('', None)
                
                # Sort by date and time
                if not df.empty and 'Datetime' in df.columns:
                    df = df.sort_values('Datetime')
                
                print(f"Successfully processed {len(df)} events")
                return df
            else:
                print("No events found in the calendar")
                raise Exception("No economic events found in the calendar - the data may be temporarily unavailable")
        except Exception as e:
            print(f"Error fetching calendar data: {e}")
            raise
    
    def _empty_dataframe(self):
        """Return an empty DataFrame with the expected columns"""
        return pd.DataFrame(columns=[
            "Date", "Time", "Datetime", "Release", "Impact", "For", "Actual", "Expected", "Prior"
        ])

# Test the custom calendar implementation
if __name__ == "__main__":
    import asyncio
    
    async def main():
        calendar = CustomCalendar()
        calendar_data = await calendar.calendar()
        
        print("\nCalendar data type:", type(calendar_data))
        print("DataFrame shape:", calendar_data.shape)
        print("DataFrame columns:", calendar_data.columns.tolist())
        
        if not calendar_data.empty:
            print("\nFirst few rows:")
            print(calendar_data.head())
            
            # Print unique dates
            if "Date" in calendar_data.columns:
                print("\nUnique dates:", calendar_data["Date"].unique())
                
            # Print sample of events for each date
            for date in calendar_data["Date"].unique():
                if date:
                    print(f"\nEvents for {date}:")
                    date_events = calendar_data[calendar_data["Date"] == date].head(3)
                    print(date_events[["Time", "Release", "Impact", "For", "Actual", "Expected", "Prior"]])
        else:
            print("\nDataFrame is empty")
    
    asyncio.run(main()) 