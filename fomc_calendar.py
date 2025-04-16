import aiohttp
import pandas as pd
from bs4 import BeautifulSoup
import re
from typing import Optional, Tuple
from urllib.parse import urljoin
from fomc_summarizer import FOMCSummarizer

class FOMCCalendar:
    """Class to scrape FOMC meeting data from the Federal Reserve website"""
    
    MONTH_MAP = {
        'January': '01',
        'February': '02',
        'March': '03',
        'April': '04',
        'May': '05',
        'June': '06',
        'July': '07',
        'August': '08',
        'September': '09',
        'October': '10',
        'November': '11',
        'December': '12'
    }
    
    BASE_URL = "https://www.federalreserve.gov"
    CALENDAR_URL = f"{BASE_URL}/monetarypolicy/fomccalendars.htm"
    
    def __init__(self, openai_api_key: Optional[str] = None):
        """Initialize the FOMC calendar scraper with optional OpenAI API key for summarization."""
        self.url = self.CALENDAR_URL
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
        }
        self.summarizer = FOMCSummarizer(openai_api_key)
    
    def _extract_year_from_text(self, text: str) -> Optional[int]:
        """Extract year from text, handling various formats."""
        year_match = re.search(r'\b(20\d{2})\b', text)
        if year_match:
            return int(year_match.group(1))
        return None

    def _parse_date_text(self, date_text: str) -> tuple[int, int]:
        """Parse date text into start and end days, handling special cases."""
        # Remove any parenthetical notes
        date_text = re.sub(r'\([^)]*\)', '', date_text).strip()
        
        # Handle date ranges
        if '-' in date_text:
            parts = date_text.split('-')
            day_start = int(re.search(r'\d+', parts[0]).group())
            day_end = int(re.search(r'\d+', parts[1]).group())
        else:
            day_start = int(re.search(r'\d+', date_text).group())
            day_end = day_start
            
        return day_start, day_end

    def _parse_month_text(self, month_text: str) -> str:
        """Parse month text, handling special cases like 'Apr/May'."""
        # Take the first month if there's a slash
        if '/' in month_text:
            month_text = month_text.split('/')[0].strip()
            
        # Remove any HTML tags
        month_text = month_text.replace('<strong>', '').replace('</strong>', '').strip()
        
        return month_text

    def _parse_date_to_datetime(self, month: str, day_start: int, day_end: int, year: int) -> pd.Timestamp:
        """Convert date components to a pandas Timestamp."""
        # Use the end day of the meeting for sorting to ensure proper chronological order
        date_str = f"{month} {day_end}, {year}"
        return pd.to_datetime(date_str)

    async def _fetch_minutes_text(self, minutes_link: str) -> str:
        """Fetch and extract text from minutes link."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(minutes_link, headers=self.headers) as response:
                    html = await response.text()
                    
            soup = BeautifulSoup(html, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
                
            # Find the main content - Fed minutes are typically in article tags
            content = soup.find('article')
            if not content:
                content = soup.find('div', class_='col-xs-12')
            if not content:
                content = soup.find(['main', 'div'], class_='content')
            if not content:
                content = soup.body
                
            # Get text and clean it up
            text = content.get_text(separator=' ', strip=True)
            
            # Clean up extra whitespace and newlines
            lines = []
            for line in text.splitlines():
                line = line.strip()
                if line and not line.isspace() and not line == "FOMC Minutes":
                    lines.append(line)
            
            text = ' '.join(lines)
            
            # If we still don't have meaningful content, try a different approach
            if not text or text == "FOMC Minutes":
                # Try finding all paragraphs
                paragraphs = soup.find_all('p')
                lines = []
                for p in paragraphs:
                    text = p.get_text(strip=True)
                    if text and not text.isspace() and not text == "FOMC Minutes":
                        lines.append(text)
                text = ' '.join(lines)
            
            # Remove common navigation text patterns if they exist
            if "The Federal Reserve, the central bank of the United States" in text:
                text = re.sub(r'The Federal Reserve, the central bank of the United States, provides.*?system\.', '', text, flags=re.DOTALL)
            
            # Remove navigation menu if it exists
            if "Federal Open Market Committee\n\nMonetary Policy Principles and Practice" in text:
                text = re.sub(r'Federal Open Market Committee\n\nMonetary Policy Principles and Practice.*?Resources for Consumers\n', '', text, flags=re.DOTALL)
            
            # Remove footnotes if they exist
            text = re.sub(r'\n\d+\. .*?Return to text\n', '', text, flags=re.DOTALL)
            
            # Clean up any remaining excessive whitespace
            text = re.sub(r'\s+', ' ', text)
            
            return text.strip()
            
        except Exception as e:
            print(f"Error fetching minutes text: {e}")
            return ""
    
    async def get_minutes_summary(self, minutes_link: str, minutes_text: Optional[str] = None) -> Optional[str]:
        """
        Generate an AI summary of the minutes text if a summarizer is available.
        
        Args:
            minutes_link: URL to the FOMC minutes
            minutes_text: Optional pre-fetched minutes text. If not provided, it will be fetched.
            
        Returns:
            A summary of the minutes if summarizer is available, otherwise None
        """
        if not self.summarizer:
            return None
            
        try:
            # If minutes_text is not provided, fetch it
            if minutes_text is None:
                minutes_text = await self._fetch_minutes_text(minutes_link)
                
            if not minutes_text:
                return None
                
            # Create a new summary using the summarizer without chunking
            summary = await self.summarizer.summarize_text(minutes_text, use_chunking=False)
            return summary
            
        except Exception as e:
            print(f"Error generating minutes summary: {e}")
            return None
    
    def _empty_dataframe(self) -> pd.DataFrame:
        """Return an empty DataFrame with the correct columns."""
        return pd.DataFrame(columns=[
            'Year', 'Month', 'Day_Start', 'Day_End', 'Date',
            'Is_Projection', 'Has_Press_Conference',
            'Statement_Link', 'Minutes_Link'
        ])

    async def calendar(self, year: Optional[int] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Fetch and parse FOMC calendar data.
        
        Args:
            year: Optional year to filter meetings. If not provided, uses current year.
            
        Returns:
            Tuple containing past and future meetings for the specified year
        """
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(self.CALENDAR_URL, headers=self.headers) as response:
                    if response.status != 200:
                        raise Exception(f"Failed to fetch FOMC calendar: {response.status}")
                    
                    html = await response.text()
                    
            soup = BeautifulSoup(html, 'html.parser')
            meetings_data = []
            
            # Use current year if no year specified
            if year is None:
                year = pd.Timestamp.now().year
            
            # Find all year panels
            year_panels = soup.find_all('div', class_='panel-default')
            
            for panel in year_panels:
                try:
                    # Extract year from panel heading
                    year_elem = panel.find('h4')
                    if not year_elem:
                        continue
                        
                    year_text = year_elem.text.strip()
                    current_year = self._extract_year_from_text(year_text)
                    
                    # Skip if not the requested year
                    if current_year != year:
                        continue

                    # Find all meeting entries in this panel
                    meeting_entries = panel.find_all('div', class_='fomc-meeting')
                    
                    for entry in meeting_entries:
                        try:
                            # Extract month and date
                            month_elem = entry.find('div', class_='fomc-meeting__month')
                            date_elem = entry.find('div', class_='fomc-meeting__date')
                            
                            if not month_elem or not date_elem:
                                continue

                            month = self._parse_month_text(month_elem.text)
                            date_text = date_elem.text.strip()
                            
                            try:
                                day_start, day_end = self._parse_date_text(date_text)
                            except (ValueError, AttributeError):
                                print(f"Could not parse date: {date_text}")
                                continue

                            # Check for statement and minutes links
                            statement_link = None
                            minutes_link = None
                            
                            # Look for links in the entire panel
                            links = panel.find_all('a')
                            for link in links:
                                href = link.get('href', '')
                                text = link.text.strip().lower()
                                
                                if 'statement' in text or 'statement' in href.lower():
                                    statement_link = urljoin(self.BASE_URL, href)
                                elif 'minutes' in text or 'minutes' in href.lower():
                                    minutes_link = urljoin(self.BASE_URL, href)
                            
                            # For meetings with minutes but no statement, construct the statement link
                            # using the standard Fed URL pattern
                            if minutes_link and not statement_link:
                                # Extract the date part from minutes link (e.g., '20250319' from fomcminutes20250319.htm)
                                date_match = re.search(r'fomcminutes(\d{8})\.htm', minutes_link)
                                if date_match:
                                    date_str = date_match.group(1)
                                    statement_link = f"{self.BASE_URL}/monetarypolicy/files/monetary{date_str}a1.pdf"

                            # Check if meeting has press conference
                            has_press = bool(entry.find(text=re.compile('Press Conference', re.IGNORECASE)))
                            
                            # A meeting is a projection if it's marked with an asterisk AND it's in the future
                            # For past meetings, we determine based on actual data (minutes/statement links)
                            meeting_date = self._parse_date_to_datetime(month, day_start, day_end, current_year)
                            reference_date = pd.Timestamp('2025-04-13')  # Today's date
                            
                            is_projection = False
                            if meeting_date > reference_date:
                                # Future meeting - use asterisk to determine if it's a projection
                                is_projection = '*' in entry.text
                            else:
                                # Past meeting - determine based on actual data
                                # If it has minutes or statement links, it's not a projection
                                is_projection = not (statement_link or minutes_link)

                            # Create meeting record
                            meetings_data.append({
                                'Year': current_year,
                                'Month': month,
                                'Day_Start': day_start,
                                'Day_End': day_end,
                                'Date': f"{month} {day_start}-{day_end}, {current_year}" if day_end != day_start else f"{month} {day_start}, {current_year}",
                                'Is_Projection': is_projection,
                                'Has_Press_Conference': has_press,
                                'Statement_Link': statement_link,
                                'Minutes_Link': minutes_link
                            })

                        except Exception as e:
                            print(f"Error processing meeting entry: {e}")
                            continue

                except Exception as e:
                    print(f"Error processing year panel: {e}")
                    continue

            if not meetings_data:
                print(f"No meetings found for year {year}")
                return self._empty_dataframe(), self._empty_dataframe()

            # Create DataFrame and sort by date
            df = pd.DataFrame(meetings_data)
            
            # Convert dates to datetime using the first day of each meeting
            df['Date'] = df.apply(lambda row: self._parse_date_to_datetime(
                row['Month'], row['Day_Start'], row['Day_End'], row['Year']
            ), axis=1)
            
            # Split into past and future meetings
            now = pd.Timestamp.now()
            past_meetings = df[df['Date'] <= now].sort_values('Date', ascending=False)
            future_meetings = df[df['Date'] > now].sort_values('Date')
            
            return past_meetings, future_meetings
            
        except Exception as e:
            print(f"Error fetching FOMC calendar: {e}")
            return self._empty_dataframe(), self._empty_dataframe()
    
    def _get_next_month(self, month: str) -> str:
        """Get the next month in sequence."""
        months = ['January', 'February', 'March', 'April', 'May', 'June',
                'July', 'August', 'September', 'October', 'November', 'December']
        try:
            idx = months.index(month)
            return months[(idx + 1) % 12]
        except ValueError:
            return month
            
    def _parse_date_string(self, date_str: str) -> str:
        """Parse a date string into a standardized format."""
        try:
            # Handle date ranges
            if '-' in date_str:
                parts = date_str.split('-')
                start_date = parts[0].strip()
                return start_date  # Use start date for sorting
            return date_str
        except Exception:
            return date_str 