import re
import pandas as pd
import aiohttp
from bs4 import BeautifulSoup
import datetime

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
        """
        try:
            # Make the request with headers to avoid being blocked
            async with aiohttp.ClientSession() as session:
                async with session.get(self.url, headers=self.headers, timeout=10) as response:
                    if response.status != 200:
                        response.raise_for_status()
                    html = await response.text()
            
            # Parse the HTML
            soup = BeautifulSoup(html, 'html.parser')
            
            # Find all calendar tables
            tables = soup.find_all("table", class_="calendar_table")
            print(f"Found {len(tables)} calendar tables")
            
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
                return self._empty_dataframe()
        except Exception as e:
            print(f"Error fetching calendar data: {e}")
            return self._empty_dataframe()
    
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