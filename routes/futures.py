from fastapi import APIRouter, HTTPException
from finvizfinance.future import Future

router = APIRouter(
    prefix="/futures",
    tags=["Futures"]
)

@router.get("")
async def get_futures():
    """Get futures market performance"""
    try:
        future = Future()
        print(future)
        return future.performance().to_dict('records')
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 