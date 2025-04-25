from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import pandas as pd

# =============================================
# Stock Market Related Models
# =============================================

class ScreenerFilters(BaseModel):
    """Filters for stock screener"""
    Exchange: Optional[str] = None
    Sector: Optional[str] = None
    Industry: Optional[str] = None
    Country: Optional[str] = None

class StockResponse(BaseModel):
    """Response model for comprehensive stock information including fundamentals, ratings, and news.
    
    Attributes:
        fundamentals: Dictionary containing fundamental financial metrics
        description: Company description
        ratings: List of analyst ratings
        news: List of recent news articles
        insider_trading: List of insider trading activities
        signal: List of trading signals
        full_info: Complete stock information dictionary
    """
    fundamentals: Dict[str, Any]
    description: str
    ratings: List[Dict[str, Any]]
    news: List[Dict[str, Any]]
    insider_trading: List[Dict[str, Any]]
    signal: List[str]
    full_info: Dict[str, Any]

class AnalystMetricsResponse(BaseModel):
    """Response model for detailed analyst metrics and estimates.
    
    Attributes:
        symbol: Stock ticker symbol
        analyst_ratings: List of individual analyst ratings
        price_targets: Price target information
        recommendation_summary: Summary of analyst recommendations
        earnings_estimates: Earnings estimates data
        revenue_estimates: Revenue estimates data
        eps_estimates: EPS estimates data
    """
    symbol: str
    analyst_ratings: List[Dict[str, Any]]
    price_targets: Dict[str, Any]
    recommendation_summary: Dict[str, Any]
    earnings_estimates: Dict[str, Any]
    revenue_estimates: Dict[str, Any]
    eps_estimates: Dict[str, Any]

class CombinedMetricsResponse(BaseModel):
    """Response model for combined analysis of multiple metrics.
    
    Attributes:
        symbol: Stock ticker symbol
        analysis: Detailed analysis results
        combined_score: Overall score combining multiple metrics
        analyst_scores: Individual scores from different analysts
        details: Additional detailed information
    """
    symbol: str
    analysis: Dict[str, Any]
    combined_score: float
    analyst_scores: Dict[str, float]
    details: Dict[str, Any]

# =============================================
# Volume and Trading Related Models
# =============================================

class VolumeStock(BaseModel):
    """Model for stock volume data.
    
    Attributes:
        ticker: Stock ticker symbol
        company: Company name
        volume: Trading volume
        price: Current stock price
    """
    ticker: str
    company: str
    volume: int
    price: float

class VolumeResponse(BaseModel):
    """Response model for volume-related data.
    
    Attributes:
        stocks: List of stocks with volume data
        total_stocks: Total number of stocks in the response
    """
    stocks: List[VolumeStock]
    total_stocks: int

# =============================================
# Options Trading Related Models
# =============================================

class OptionDetails(BaseModel):
    """Model for detailed option contract information.
    
    Attributes:
        strike: Strike price
        last_price: Last traded price
        volume: Trading volume
        open_interest: Open interest
        bid: Current bid price
        ask: Current ask price
        bid_ask_spread_pct: Bid-ask spread percentage
        probability_of_profit: Probability of profit
        profitability_score: Score indicating potential profitability
        implied_volatility: Implied volatility
    """
    strike: float
    last_price: float
    volume: int
    open_interest: int
    bid: float
    ask: float
    bid_ask_spread_pct: float
    probability_of_profit: float
    profitability_score: float
    implied_volatility: float

class MaxPainResponse(BaseModel):
    """Response model for options max pain analysis.
    
    Attributes:
        max_pain_strike: Strike price with maximum pain
        spot_price: Current spot price
        historical_volatility: Historical volatility
        best_call: Details of best call option
        best_put: Details of best put option
        expiration: Option expiration date
        days_to_expiry: Days remaining until expiration
    """
    max_pain_strike: float
    spot_price: float
    historical_volatility: float
    best_call: OptionDetails
    best_put: OptionDetails
    expiration: str
    days_to_expiry: int

class MaxPainStock(BaseModel):
    """Model for stock with options max pain analysis.
    
    Attributes:
        ticker: Stock ticker symbol
        company: Company name
        volume: Trading volume
        price: Current stock price
        max_pain_strike: Strike price with maximum pain
        spot_price: Current spot price
        call_strike: Best call strike price
        call_price: Best call price
        call_volume: Call trading volume
        call_oi: Call open interest
        call_pop: Call probability of profit
        put_strike: Best put strike price
        put_price: Best put price
        put_volume: Put trading volume
        put_oi: Put open interest
        put_pop: Put probability of profit
        opportunity_score: Score indicating trading opportunity
        expiration: Option expiration date
        days_to_expiry: Days remaining until expiration
    """
    ticker: str
    company: str
    volume: int
    price: float
    max_pain_strike: float
    spot_price: float
    call_strike: float
    call_price: float
    call_volume: int
    call_oi: int
    call_pop: float
    put_strike: float
    put_price: float
    put_volume: int
    put_oi: int
    put_pop: float
    opportunity_score: float
    expiration: str
    days_to_expiry: int

class MaxPainOpportunitiesResponse(BaseModel):
    """Response model for options trading opportunities.
    
    Attributes:
        opportunities: List of potential trading opportunities
        total_opportunities: Total number of opportunities found
    """
    opportunities: List[MaxPainStock]
    total_opportunities: int

# =============================================
# Calendar and Event Related Models
# =============================================

class CalendarFilter(BaseModel):
    """Filter model for economic calendar events.
    
    Attributes:
        date: Specific date to filter
        impact: Impact level to filter
        release: Release type to filter
    """
    date: Optional[str] = None
    impact: Optional[str] = None
    release: Optional[str] = None

class CalendarEvent(BaseModel):
    """Model for economic calendar events.
    
    Attributes:
        Date: Event date
        Time: Event time
        Datetime: Combined date and time
        Release: Type of release
        Impact: Impact level
        For: Event description
        Actual: Actual value (if available)
        Expected: Expected value (if available)
        Prior: Prior value (if available)
    """
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
    """Response model for calendar events.
    
    Attributes:
        events: List of calendar events
        total_events: Total number of events
        available_dates: List of dates with events
    """
    events: List[CalendarEvent]
    total_events: int
    available_dates: List[str]

class CalendarSummary(BaseModel):
    """Summary model for calendar events.
    
    Attributes:
        overall_summary: Summary of all events
        today_summary: Summary of today's events
        total_events: Total number of events
        today_events: Number of events today
    """
    overall_summary: Dict[str, int]
    today_summary: Dict[str, int]
    total_events: int
    today_events: int

# =============================================
# FOMC Related Models
# =============================================

class FOMCMeeting(BaseModel):
    """Model for Federal Open Market Committee meeting data.
    
    Attributes:
        Date: Meeting date
        Is_Projection: Whether the data is a projection
        Has_Press_Conference: Whether there was a press conference
        Statement_Link: Link to meeting statement
        Minutes_Link: Link to meeting minutes
        Minutes_Text: Text of meeting minutes
        Minutes_Summary: Summary of meeting minutes
    """
    Date: str
    Is_Projection: bool
    Has_Press_Conference: bool
    Statement_Link: Optional[str] = None
    Minutes_Link: Optional[str] = None
    Minutes_Text: Optional[str] = None
    Minutes_Summary: Optional[str] = None

class FOMCLatestResponse(BaseModel):
    """Response model for latest FOMC meeting information.
    
    Attributes:
        meeting: Current meeting data
        next_meeting: Next meeting data
        status: Response status
        error: Error message if any
    """
    meeting: Optional[FOMCMeeting] = None
    next_meeting: Optional[FOMCMeeting] = None
    status: str
    error: Optional[str] = None

# =============================================
# News Related Models
# =============================================

class NewsItem(BaseModel):
    """Model for news article information.
    
    Attributes:
        Title: Article title
        Source: News source
        Date: Publication date
        Link: Article URL
    """
    Title: str
    Source: str
    Date: str
    Link: str

class NewsResponse(BaseModel):
    """Response model for news articles.
    
    Attributes:
        news: List of news articles
        blogs: List of blog posts
    """
    news: List[NewsItem]
    blogs: List[NewsItem]

# =============================================
# System Related Models
# =============================================

class HealthCheck(BaseModel):
    """Model for system health check information.
    
    Attributes:
        status: System status
        version: API version
        timestamp: Current timestamp
        uptime: System uptime in seconds
        services: Status of various services
    """
    status: str
    version: str
    timestamp: str
    uptime: float
    services: Dict[str, str]