# Market Data API

A comprehensive API for accessing market data, economic calendar, and stock information.

## Table of Contents
- [Overview](#overview)
- [Installation](#installation)
- [Configuration](#configuration)
- [Authentication](#authentication)
- [Base URL](#base-url)
- [API Endpoints](#api-endpoints)
  - [Stocks](#stocks)
  - [Screener](#screener)
  - [Calendar](#calendar)
  - [FOMC](#fomc)
  - [News](#news)
  - [Insider Trading](#insider-trading)
  - [Futures](#futures)
  - [System](#system)
- [Response Models](#response-models)
- [Rate Limiting](#rate-limiting)
- [Caching](#caching)
- [Error Handling](#error-handling)
- [Development](#development)

## Overview

The Market Data API provides access to various financial data sources including:
- Stock information and fundamentals
- Economic calendar events
- FOMC meeting data and minutes
- Financial news
- Insider trading data
- Futures market data

## Installation

1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your configuration
```
4. Run the API:
```bash
python api.py
```

## Configuration

The API can be configured through environment variables:

```env
# API Configuration
PORT=8000
HOST=0.0.0.0
DEBUG=False

# Rate Limiting
RATE_LIMIT_REQUESTS=100
RATE_LIMIT_WINDOW=60

# Cache Settings
STOCK_CACHE_TTL=300
SCREENER_CACHE_TTL=600
CALENDAR_CACHE_TTL=1800

# OpenAI API Key (for FOMC summaries)
OPENAI_API_KEY=your_api_key_here
```

## Authentication

The API uses RapidAPI for authentication. Include your API key in the request header:

```
X-RapidAPI-Key: YOUR_API_KEY
```

## Base URL

The API is available at:
```
https://markets-api.p.rapidapi.com
```

For local development:
```
http://localhost:8000
```

## API Endpoints

### Stocks

#### GET /stock/{symbol}
Get comprehensive stock information.

**Parameters:**
- `symbol` (path): Stock symbol (e.g., AAPL, MSFT)
- `screener` (query, optional): Screener to use (e.g., 'overview', 'all')

**Response:**
```json
{
    "fundamentals": {},
    "description": "string",
    "ratings": [],
    "news": [],
    "insider_trading": [],
    "signal": [],
    "full_info": {}
}
```

#### GET /stock/{symbol}/analysts
Get comprehensive analyst metrics for a given stock.

**Parameters:**
- `symbol` (path): Stock symbol (e.g., AAPL, MSFT)

**Response:**
```json
{
    "symbol": "string",
    "analyst_ratings": [],
    "price_targets": {
        "avg_price_target": {},
        "price_target_low": {},
        "price_target_high": {}
    },
    "recommendation_summary": {
        "strong_buy": {},
        "buy": {},
        "hold": {},
        "sell": {},
        "strong_sell": {}
    },
    "earnings_estimates": {
        "current_quarter": {},
        "next_quarter": {},
        "current_year": {},
        "next_year": {}
    },
    "revenue_estimates": {
        "current_quarter": {},
        "next_quarter": {},
        "current_year": {},
        "next_year": {}
    },
    "eps_estimates": {
        "current_quarter": {},
        "next_quarter": {},
        "current_year": {},
        "next_year": {}
    }
}
```

### Screener

#### POST /screener/overview
Get stock screener overview with specified filters.

**Request Body:**
```json
{
    "Exchange": "string",
    "Sector": "string",
    "Industry": "string",
    "Country": "string"
}
```

#### POST /screener/valuation
Get stock screener valuation metrics.

**Request Body:**
Same as overview screener.

#### POST /screener/financial
Get stock screener financial metrics.

**Request Body:**
Same as overview screener.

### Calendar

#### GET /calendar
Get economic calendar data.

**Response:**
```json
{
    "events": [
        {
            "Date": "string",
            "Time": "string",
            "Datetime": "string",
            "Release": "string",
            "Impact": "string",
            "For": "string",
            "Actual": "string",
            "Expected": "string",
            "Prior": "string"
        }
    ],
    "total_events": 0,
    "available_dates": []
}
```

#### POST /calendar/filter
Filter economic calendar events.

**Request Body:**
```json
{
    "date": "string",
    "impact": "string",
    "release": "string"
}
```

#### GET /calendar/summary
Get summary of economic calendar events.

**Response:**
```json
{
    "overall_summary": {},
    "today_summary": {},
    "total_events": 0,
    "today_events": 0
}
```

### FOMC

#### GET /fomc/calendar
Get FOMC meeting calendar data.

**Response:**
```json
{
    "past_meetings": [],
    "future_meetings": [],
    "total_meetings": 0,
    "years": []
}
```

#### GET /fomc/latest
Get detailed information about the latest and next FOMC meetings.

**Response:**
```json
{
    "meeting": {
        "Date": "string",
        "Is_Projection": false,
        "Has_Press_Conference": false,
        "Statement_Link": "string",
        "Minutes_Link": "string",
        "Minutes_Text": "string",
        "Minutes_Summary": "string"
    },
    "next_meeting": {},
    "status": "string",
    "error": "string"
}
```

### News

#### GET /news
Get latest financial news and blog posts.

**Response:**
```json
{
    "news": [],
    "blogs": []
}
```

### Insider Trading

#### GET /insider
Get insider trading information.

**Parameters:**
- `option` (query): Type of insider trading data to retrieve
  - Options: 'latest', 'latest buys', 'latest sales', 'top week', 'top week buys', 'top week sales', 'top owner trade', 'top owner buys', 'top owner sales', or an insider ID number

### Futures

#### GET /futures
Get futures market performance.

### System

#### GET /health
Health check endpoint for monitoring API status.

**Response:**
```json
{
    "status": "string",
    "version": "string",
    "timestamp": "string",
    "uptime": 0,
    "services": {}
}
```

## Response Models

### StockResponse
```python
class StockResponse(BaseModel):
    fundamentals: Dict[str, Any]
    description: str
    ratings: List[Dict[str, Any]]
    news: List[Dict[str, Any]]
    insider_trading: List[Dict[str, Any]]
    signal: List[str]
    full_info: Dict[str, Any]
```

### AnalystMetricsResponse
```python
class AnalystMetricsResponse(BaseModel):
    symbol: str
    analyst_ratings: List[Dict[str, Any]]
    price_targets: Dict[str, Any]
    recommendation_summary: Dict[str, Any]
    earnings_estimates: Dict[str, Any]
    revenue_estimates: Dict[str, Any]
    eps_estimates: Dict[str, Any]
```

### CalendarEvent
```python
class CalendarEvent(BaseModel):
    Date: str
    Time: str
    Datetime: str
    Release: str
    Impact: str
    For: str
    Actual: Optional[str]
    Expected: Optional[str]
    Prior: Optional[str]
```

### FOMCMeeting
```python
class FOMCMeeting(BaseModel):
    Date: str
    Is_Projection: bool
    Has_Press_Conference: bool
    Statement_Link: Optional[str]
    Minutes_Link: Optional[str]
    Minutes_Text: Optional[str]
    Minutes_Summary: Optional[str]
```

## Rate Limiting

The API implements rate limiting to prevent abuse. By default, requests are limited to:
- 100 requests per minute per IP address
- Configurable through environment variables

## Caching

The API implements caching for improved performance:

- Stock Data: 5 minutes TTL
- Screener Data: 10 minutes TTL
- Calendar Data: 30 minutes TTL

Cache settings can be configured through environment variables.

## Error Handling

The API uses standard HTTP status codes:
- 200: Success
- 400: Bad Request
- 404: Not Found
- 429: Too Many Requests
- 500: Internal Server Error

Error responses include a detail message:
```json
{
    "detail": "Error message description"
}
```

## Development

### Prerequisites
- Python 3.8+
- pip
- virtualenv (recommended)

### Setup Development Environment
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
pytest

# Run linting
flake8
```

### Running in Development Mode
```bash
# Enable auto-reload
uvicorn api:app --reload --port 8000
```

### API Documentation
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc
- OpenAPI Schema: http://localhost:8000/openapi.json