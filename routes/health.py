from fastapi import APIRouter, Request
from models import HealthCheck
from custom_calendar import CustomCalendar
from datetime import datetime
import time

router = APIRouter(prefix="/health", tags=["Health"])

@router.get("", response_model=HealthCheck)
async def health_check(request: Request):
    """Check API health status"""
    try:
        # Check if services are responding
        calendar_status = "healthy"
        try:
            # Test calendar service
            calendar = CustomCalendar()
            await calendar.calendar()
        except Exception as e:
            print(f"Calendar service health check failed: {e}")
            calendar_status = "unhealthy"
            
        # Calculate uptime
        uptime_seconds = int(time.time() - request.app.state.start_time)
        
        return HealthCheck(
            status="healthy",
            version="1.0.0",
            timestamp=datetime.now().isoformat(),
            uptime=float(uptime_seconds),
            services={
                "calendar": calendar_status
            }
        )
    except Exception as e:
        # Even if health check fails, return 200 with degraded status
        print(f"Health check error: {e}")
        return HealthCheck(
            status="degraded",
            version="1.0.0",
            timestamp=datetime.now().isoformat(),
            uptime=0.0,  # Return 0.0 instead of "unknown"
            services={
                "calendar": "unknown"
            }
        ) 