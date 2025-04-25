from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from custom_calendar import CustomCalendar
from fomc_calendar import FOMCCalendar
from models import CalendarEvent, CalendarResponse, CalendarFilter, CalendarSummary, FOMCLatestResponse, FOMCMeeting
from datetime import datetime
from cachetools import TTLCache
from functools import wraps

# Initialize cache
calendar_cache = TTLCache(maxsize=10, ttl=1800)  # 30 minutes cache for calendar data

router = APIRouter(prefix="/calendar", tags=["Calendar"])

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

@router.get("", response_model=CalendarResponse)
async def get_calendar():
    """Get economic calendar data"""
    try:
        calendar = CustomCalendar()
        df = await calendar.calendar()
        
        if df is None or df.empty:
            # Return empty response with 200 status
            return CalendarResponse(
                events=[],
                total_events=0,
                available_dates=[]
            )
            
        # Convert DataFrame to list of events
        events = []
        for _, row in df.iterrows():
            try:
                event = CalendarEvent(
                    Date=row["Date"],
                    Time=row["Time"],
                    Datetime=row["Datetime"],
                    Release=row["Release"],
                    Impact=row["Impact"],
                    For=row["For"],
                    Actual=row.get("Actual"),
                    Expected=row.get("Expected"),
                    Prior=row.get("Prior")
                )
                events.append(event)
            except Exception as e:
                print(f"Error processing calendar event: {e}")
                continue
                
        # Get unique dates for available_dates
        available_dates = sorted(df["Date"].unique().tolist())
        
        return CalendarResponse(
            events=events,
            total_events=len(events),
            available_dates=available_dates
        )
    except Exception as e:
        # Log the error but return empty response with 200 status
        print(f"Calendar endpoint error: {e}")
        return CalendarResponse(
            events=[],
            total_events=0,
            available_dates=[]
        )

@router.post("/filter", response_model=CalendarResponse)
async def filter_calendar(filter: CalendarFilter):
    """
    Filter economic calendar events by date, impact, or release name.
    
    Parameters:
    - date: Filter by date (e.g., "Mon Apr 07")
    - impact: Filter by impact level (1-3)
    - release: Filter by release name (e.g., "CPI")
    """
    try:
        calendar = CustomCalendar()
        df = await calendar.calendar()
        
        if df.empty:
            return JSONResponse(
                status_code=404,
                content={"message": "No calendar data available"}
            )
        
        # Apply filters
        if filter.date:
            df = df[df['Date'] == filter.date]
        if filter.impact:
            df = df[df['Impact'] == filter.impact]
        if filter.release:
            df = df[df['Release'].str.contains(filter.release, case=False, na=False)]
        
        events = df.to_dict('records')
        available_dates = sorted(df['Date'].unique().tolist())
        
        return {
            "events": events,
            "total_events": len(events),
            "available_dates": available_dates
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/summary", response_model=CalendarSummary)
async def get_calendar_summary():
    """
    Get a summary of economic calendar events grouped by impact level.
    
    Returns summary statistics including:
    - Overall summary of events by impact level
    - Today's events summary
    - Total number of events
    - Number of events today
    """
    try:
        calendar = CustomCalendar()
        df = await calendar.calendar()
        
        if df.empty:
            return JSONResponse(
                status_code=404,
                content={"message": "No calendar data available"}
            )
        
        # Get today's date in the format used in the calendar
        today = datetime.now().strftime("%a %b %d")
        
        # Calculate summaries
        overall_summary = df['Impact'].value_counts().to_dict()
        today_df = df[df['Date'] == today]
        today_summary = today_df['Impact'].value_counts().to_dict()
        
        return {
            "overall_summary": overall_summary,
            "today_summary": today_summary,
            "total_events": len(df),
            "today_events": len(today_df)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/fomc", tags=["FOMC"])
async def get_fomc_calendar():
    """Get FOMC meeting calendar data."""
    try:
        fomc = FOMCCalendar()
        past_meetings_df, future_meetings_df = await fomc.calendar()
        
        # Convert DataFrames to dictionaries
        past_meetings = past_meetings_df.to_dict(orient='records')
        future_meetings = future_meetings_df.to_dict(orient='records')
        
        # Get unique years
        all_years = sorted(list(set([meeting['Year'] for meeting in past_meetings + future_meetings])))
        
        return {
            "past_meetings": past_meetings,
            "future_meetings": future_meetings,
            "total_meetings": len(past_meetings) + len(future_meetings),
            "years": all_years
        }
    except Exception as e:
        print(f"Error in FOMC calendar endpoint: {e}")
        return {
            "past_meetings": [],
            "future_meetings": [],
            "total_meetings": 0,
            "years": []
        }

@router.get("/fomc/latest", response_model=FOMCLatestResponse)
async def get_latest_fomc_meeting():
    """Get detailed information about the latest and next FOMC meetings."""
    try:
        calendar = FOMCCalendar()
        past_meetings, future_meetings = await calendar.calendar()
        
        # Get the latest non-projection meeting
        latest_meeting = None
        for _, meeting in past_meetings.iterrows():
            if not meeting['Is_Projection']:
                latest_meeting = meeting.copy()
                # Convert Timestamp to string
                latest_meeting['Date'] = latest_meeting['Date'].strftime('%Y-%m-%d')
                # Fetch minutes text if minutes link exists
                if latest_meeting['Minutes_Link']:
                    latest_meeting['Minutes_Text'] = await calendar._fetch_minutes_text(latest_meeting['Minutes_Link'])
                    if latest_meeting['Minutes_Text']:
                        latest_meeting['Minutes_Summary'] = await calendar.get_minutes_summary(
                            latest_meeting['Minutes_Link'], 
                            latest_meeting['Minutes_Text']
                        )
                break
        
        # Get the next meeting
        next_meeting = None
        if not future_meetings.empty:
            next_meeting = future_meetings.iloc[0].copy()
            # Convert Timestamp to string
            next_meeting['Date'] = next_meeting['Date'].strftime('%Y-%m-%d')
            # For future meetings, set links to None
            next_meeting['Statement_Link'] = None
            next_meeting['Minutes_Link'] = None
            next_meeting['Minutes_Text'] = None
            next_meeting['Minutes_Summary'] = None
        
        return FOMCLatestResponse(
            meeting=FOMCMeeting(**latest_meeting.to_dict()) if latest_meeting is not None else None,
            next_meeting=FOMCMeeting(**next_meeting.to_dict()) if next_meeting is not None else None,
            status="success"
        )
    except Exception as e:
        print(f"Error in get_latest_fomc_meeting: {e}")
        return FOMCLatestResponse(
            meeting=None,
            next_meeting=None,
            status="error",
            error=str(e)
        ) 