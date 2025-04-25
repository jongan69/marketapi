from fastapi import APIRouter, HTTPException
from finvizfinance.news import News
from models import NewsResponse

router = APIRouter(prefix="/news", tags=["News"])

@router.get("", response_model=NewsResponse)
async def get_news():
    """Get latest financial news and blog posts"""
    try:
        fnews = News()
        news_data = fnews.get_news()
        return NewsResponse(
            news=news_data['news'].to_dict('records'),
            blogs=news_data['blogs'].to_dict('records')
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 