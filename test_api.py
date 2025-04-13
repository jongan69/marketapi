#!/usr/bin/env python3
import requests
import json
import time
import sys
from datetime import datetime
from typing import Dict, List, Any, Optional

# Configuration
BASE_URL = "http://localhost:8000"  # Change this to your API URL
VERBOSE = True  # Set to False to only show errors
REQUEST_DELAY = 2  # Delay between requests in seconds
MAX_RETRIES = 3  # Maximum number of retries for rate-limited endpoints

# Test results
results = {
    "total": 0,
    "passed": 0,
    "failed": 0,
    "errors": []
}

def log(message: str, is_error: bool = False) -> None:
    """Print log message with timestamp"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if is_error:
        print(f"[{timestamp}] ❌ ERROR: {message}")
    elif VERBOSE:
        print(f"[{timestamp}] ℹ️ INFO: {message}")

def test_endpoint(endpoint: str, method: str = "GET", params: Optional[Dict] = None, 
                 data: Optional[Dict] = None, expected_status: int = 200, 
                 retry_on_rate_limit: bool = False, validate_response: bool = False,
                 expected_response_keys: Optional[List[str]] = None) -> bool:
    """Test an API endpoint and return True if successful, False otherwise"""
    results["total"] += 1
    url = f"{BASE_URL}{endpoint}"
    
    # Add delay between requests to avoid rate limiting
    time.sleep(REQUEST_DELAY)
    
    retries = 0
    while retries <= MAX_RETRIES:
        try:
            log(f"Testing {method} {endpoint}" + (f" (attempt {retries+1}/{MAX_RETRIES+1})" if retries > 0 else ""))
            
            if method == "GET":
                response = requests.get(url, params=params)
            elif method == "POST":
                response = requests.post(url, params=params, json=data)
            else:
                log(f"Unsupported HTTP method: {method}", True)
                results["failed"] += 1
                results["errors"].append(f"Unsupported HTTP method: {method}")
                return False
            
            # Check status code
            if response.status_code != expected_status:
                # Check if it's a rate limit error and we should retry
                if retry_on_rate_limit and response.status_code == 500 and "429 Client Error: Too Many Requests" in response.text:
                    retries += 1
                    if retries <= MAX_RETRIES:
                        wait_time = 5 * retries  # Exponential backoff
                        log(f"Rate limited. Waiting {wait_time} seconds before retry...")
                        time.sleep(wait_time)
                        continue
                
                log(f"Expected status {expected_status}, got {response.status_code}", True)
                log(f"Response: {response.text}", True)
                results["failed"] += 1
                results["errors"].append(f"{method} {endpoint}: Expected status {expected_status}, got {response.status_code}")
                return False
            
            # Try to parse JSON response
            try:
                json_response = response.json()
                log(f"Response: {json.dumps(json_response, indent=2)[:500]}...")
                
                # Validate response structure if requested
                if validate_response and expected_response_keys:
                    missing_keys = [key for key in expected_response_keys if key not in json_response]
                    if missing_keys:
                        log(f"Response missing expected keys: {missing_keys}", True)
                        results["failed"] += 1
                        results["errors"].append(f"{method} {endpoint}: Response missing keys {missing_keys}")
                        return False
            except json.JSONDecodeError:
                log(f"Response is not valid JSON: {response.text[:200]}...")
            
            log(f"✅ {method} {endpoint} passed")
            results["passed"] += 1
            return True
        
        except Exception as e:
            log(f"Exception: {str(e)}", True)
            results["failed"] += 1
            results["errors"].append(f"{method} {endpoint}: {str(e)}")
            return False
    
    # If we've exhausted all retries
    log(f"Failed after {MAX_RETRIES} retries due to rate limiting", True)
    results["failed"] += 1
    results["errors"].append(f"{method} {endpoint}: Rate limited after {MAX_RETRIES} retries")
    return False

def test_stock_endpoints() -> None:
    """Test stock-related endpoints"""
    log("Testing stock endpoints...")
    
    # Test root endpoint
    test_endpoint("/")
    
    # Test stock info endpoint with different symbols
    symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
    for symbol in symbols:
        test_endpoint(f"/stock/{symbol}", validate_response=True, 
                     expected_response_keys=["fundamentals", "description", "ratings", "news", "insider_trading", "signal", "full_info"])
        test_endpoint(f"/stock/{symbol}?screener=overview", validate_response=True, 
                     expected_response_keys=["fundamentals", "description", "ratings", "news", "insider_trading", "signal", "full_info"])
    
    # Test invalid stock symbol - API returns 500 with 404 message from FinViz
    test_endpoint("/stock/INVALID_SYMBOL", expected_status=500)
    
    # Test screener endpoints with valid FinViz filters
    screener_filters = {
        "Exchange": "NASDAQ",
        "Sector": "Technology",
        "Industry": "Software - Application",
        "Country": "USA"
    }
    
    # Add retry_on_rate_limit=True for screener endpoints
    test_endpoint("/screener/overview", "POST", data=screener_filters, retry_on_rate_limit=True)
    test_endpoint("/screener/valuation", "POST", data=screener_filters, retry_on_rate_limit=True)
    test_endpoint("/screener/financial", "POST", data=screener_filters, retry_on_rate_limit=True)
    
    # Test screener with invalid filters - API returns 500 with error message from FinViz
    invalid_filters = {
        "Exchange": "INVALID_EXCHANGE",
        "Sector": "INVALID_SECTOR",
        "Industry": "INVALID_INDUSTRY",
        "Country": "INVALID_COUNTRY"
    }
    test_endpoint("/screener/overview", "POST", data=invalid_filters, expected_status=500)
    
    # Test news endpoint
    test_endpoint("/news", validate_response=True, expected_response_keys=["news", "blogs"])
    
    # Test insider trading endpoint with different options
    insider_options = ["latest", "latest buys", "latest sales", "top week", "top week buys", "top week sales"]
    for option in insider_options:
        test_endpoint(f"/insider?option={option}")
    
    # Test insider trading with invalid option - API returns 200 with empty array
    test_endpoint("/insider?option=invalid_option", expected_status=200)
    
    # Test futures endpoint
    test_endpoint("/futures")

def test_calendar_endpoints() -> None:
    """Test calendar-related endpoints"""
    log("Testing calendar endpoints...")
    
    # Test calendar endpoints
    test_endpoint("/calendar", validate_response=True, expected_response_keys=["events", "total_events", "available_dates"])
    test_endpoint("/calendar/summary", validate_response=True, expected_response_keys=["overall_summary", "today_summary", "total_events", "today_events"])
    
    # Test calendar filter endpoint with different filters
    calendar_filters = [
        {"date": "Mon Apr 07"},
        {"impact": "1"},
        {"release": "CPI"}
    ]
    
    for filter_data in calendar_filters:
        test_endpoint("/calendar/filter", "POST", data=filter_data, validate_response=True, 
                     expected_response_keys=["events", "total_events", "available_dates"])
    
    # Test calendar filter with invalid date - API returns 200 with empty result
    test_endpoint("/calendar/filter", "POST", data={"date": "Invalid Date"}, expected_status=200)
    
    # Test calendar filter with invalid impact - API returns 200 with empty result
    test_endpoint("/calendar/filter", "POST", data={"impact": "invalid"}, expected_status=200)

def test_fomc_endpoints() -> None:
    """Test FOMC-related endpoints"""
    log("Testing FOMC endpoints...")
    
    # Test FOMC calendar endpoint
    test_endpoint("/fomc/calendar", validate_response=True, 
                 expected_response_keys=["past_meetings", "future_meetings", "total_meetings", "years"])
    
    # Test latest FOMC meeting endpoint
    test_endpoint("/fomc/latest", validate_response=True, 
                 expected_response_keys=["meeting", "next_meeting", "status", "error"])

def test_health_endpoint() -> None:
    """Test health check endpoint"""
    log("Testing health endpoint...")
    test_endpoint("/health", validate_response=True, 
                 expected_response_keys=["status", "version", "timestamp", "uptime", "services"])

def print_summary() -> None:
    """Print test summary"""
    print("\n" + "="*50)
    print(f"TEST SUMMARY: {results['passed']}/{results['total']} passed")
    print("="*50)
    
    if results["failed"] > 0:
        print("\nFAILED TESTS:")
        for error in results["errors"]:
            print(f"- {error}")
    else:
        print("\nAll tests passed successfully!")
    
    print("\n" + "="*50)

def main() -> None:
    """Main function to run all tests"""
    log("Starting API tests...")
    
    # Test all endpoint groups
    test_stock_endpoints()
    test_calendar_endpoints()
    test_fomc_endpoints()
    test_health_endpoint()
    
    # Print summary
    print_summary()
    
    # Exit with appropriate status code
    sys.exit(0 if results["failed"] == 0 else 1)

if __name__ == "__main__":
    main() 