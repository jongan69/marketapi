# Markets API

![Markets API](https://img.shields.io/badge/API-Markets-blue)
![Version](https://img.shields.io/badge/version-1.0.0-green)
![License](https://img.shields.io/badge/license-MIT-yellow)

A comprehensive API for accessing market data, economic calendar, and stock information from FinViz.

## Overview

Markets API provides real-time and historical financial data through a simple REST API. Access stock information, economic calendar events, and market data with just a few API calls.

## Features

- **Stock Information**: Get comprehensive data about any publicly traded stock
- **Economic Calendar**: Access upcoming economic events and their impact
- **Market Data**: Retrieve market performance metrics and trends
- **Insider Trading**: Monitor insider trading activities
- **Financial News**: Stay updated with the latest financial news

## Getting Started

### Authentication

All API requests require an API key to be included in the header:

```
X-RapidAPI-Key: YOUR_API_KEY
```

You can obtain an API key by subscribing to the Markets API on RapidAPI.

### Base URL

```
https://markets-api.p.rapidapi.com
```

## API Endpoints

### Stock Information

#### Get Stock Info

```
GET /stock/{ticker}
```

Get comprehensive information about a specific stock.

**Parameters:**
- `ticker` (path, required): Stock symbol (e.g., AAPL, MSFT)

**Response:**
```json
{
  "fundamentals": { ... },
  "description": { ... },
  "ratings": { ... },
  "news": { ... },
  "insider_trading": { ... },
  "signal": { ... },
  "full_info": { ... }
}
```

### Economic Calendar

#### Get Calendar

```
GET /calendar
```

Get economic calendar events with detailed information.

**Response:**
```json
{
  "events": [
    {
      "Date": "Mon Apr 07",
      "Time": "14:00",
      "Datetime": "2024-04-07T14:00:00",
      "Release": "Consumer Credit",
      "Impact": "2",
      "For": "Feb",
      "Actual": "14.2B",
      "Expected": "15.0B",
      "Prior": "19.5B"
    },
    ...
  ],
  "total_events": 15,
  "available_dates": ["Mon Apr 07", "Tue Apr 08", ...]
}
```

#### Filter Calendar

```
POST /calendar/filter
```

Filter economic calendar events by date, impact level, or release name.

**Request Body:**
```json
{
  "date": "Mon Apr 07",
  "impact": "2",
  "release": "CPI"
}
```

**Response:**
```json
{
  "events": [ ... ],
  "total_events": 3,
  "available_dates": ["Mon Apr 07"]
}
```

#### Get Calendar Summary

```
GET /calendar/summary
```

Get a summary of economic calendar events grouped by impact level.

**Response:**
```json
{
  "overall_summary": {
    "1": 5,
    "2": 7,
    "3": 3
  },
  "today_summary": {
    "1": 2,
    "2": 3,
    "3": 1
  },
  "total_events": 15,
  "today_events": 6
}
```

### Stock Screener

#### Get Screener Overview

```
POST /screener/overview
```

Get stock screener overview with specified filters.

**Request Body:**
```json
{
  "Exchange": "NASDAQ",
  "Sector": "Technology",
  "Industry": "Software",
  "Country": "USA"
}
```

**Response:**
```json
[
  {
    "Ticker": "AAPL",
    "Company": "Apple Inc.",
    "Sector": "Technology",
    "Industry": "Consumer Electronics",
    "Country": "USA",
    "Market Cap": "2.85T",
    "P/E": "28.45",
    "Price": "175.84",
    "Change": "1.23%"
  },
  ...
]
```

#### Get Screener Valuation

```
POST /screener/valuation
```

Get stock screener valuation metrics with specified filters.

**Request Body:**
```json
{
  "Exchange": "NYSE",
  "Sector": "Financial",
  "Industry": "Banks",
  "Country": "USA"
}
```

**Response:**
```json
[
  {
    "Ticker": "JPM",
    "Company": "JPMorgan Chase & Co.",
    "Market Cap": "500.2B",
    "P/E": "13.2",
    "Fwd P/E": "12.8",
    "PEG": "1.5",
    "P/S": "3.2",
    "P/B": "1.8",
    "P/C": "12.5",
    "P/Free Cash Flow": "15.3"
  },
  ...
]
```

#### Get Screener Financial

```
POST /screener/financial
```

Get stock screener financial metrics with specified filters.

**Request Body:**
```json
{
  "Exchange": "NASDAQ",
  "Sector": "Healthcare",
  "Industry": "Biotechnology",
  "Country": "USA"
}
```

**Response:**
```json
[
  {
    "Ticker": "MRNA",
    "Company": "Moderna Inc.",
    "Market Cap": "35.2B",
    "Dividend Yield": "0.00%",
    "ROA": "15.2%",
    "ROE": "25.8%",
    "ROI": "18.5%",
    "Curr R": "2.8",
    "Quick R": "1.9",
    "LTDebt/Eq": "0.12",
    "Debt/Eq": "0.15",
    "Gross M": "65.8%",
    "Oper M": "35.2%",
    "Profit M": "28.5%"
  },
  ...
]
```

### Financial News

#### Get News

```
GET /news
```

Get latest financial news and blog posts.

**Response:**
```json
{
  "news": [
    {
      "Date": "2024-04-10",
      "Time": "09:30",
      "Title": "Fed Signals Potential Rate Cut in Coming Months",
      "Link": "https://example.com/news/12345",
      "Source": "Financial Times"
    },
    ...
  ],
  "blogs": [
    {
      "Date": "2024-04-10",
      "Time": "08:15",
      "Title": "Tech Stocks Rally as AI Boom Continues",
      "Link": "https://example.com/blog/67890",
      "Source": "Market Insights"
    },
    ...
  ]
}
```

### Insider Trading

#### Get Insider Trading

```
GET /insider
```

Get insider trading information.

**Parameters:**
- `option` (query, optional): Type of insider trading data (default: "top owner trade")

**Response:**
```json
[
  {
    "Date": "2024-04-09",
    "Ticker": "TSLA",
    "Owner": "Elon Musk",
    "Relationship": "CEO",
    "Transaction": "Buy",
    "Shares": "10000",
    "Value": "$1,750,000",
    "Shares Total": "411,062,576",
    "SEC Form 4": "https://www.sec.gov/Archives/edgar/data/..."
  },
  ...
]
```

### Futures

#### Get Futures

```
GET /futures
```

Get futures market performance.

**Response:**
```json
[
  {
    "Ticker": "ES",
    "Name": "E-mini S&P 500",
    "Last": "5,234.50",
    "Change": "+0.75%",
    "Volume": "1,234,567",
    "Open Interest": "2,345,678"
  },
  ...
]
```

## Rate Limits

- Free tier: 100 requests per minute
- Pro tier: 1,000 requests per minute
- Enterprise tier: Custom limits

## Error Handling

The API uses standard HTTP response codes to indicate the success or failure of requests.

- `200 OK`: Request succeeded
- `400 Bad Request`: Invalid request parameters
- `401 Unauthorized`: Missing or invalid API key
- `403 Forbidden`: Insufficient permissions
- `404 Not Found`: Resource not found
- `429 Too Many Requests`: Rate limit exceeded
- `500 Internal Server Error`: Server error

## Support

For support, please contact us at support@marketsapi.com or visit our [documentation](https://marketsapi.com/docs).

## License

This API is licensed under the MIT License. See the LICENSE file for details.
