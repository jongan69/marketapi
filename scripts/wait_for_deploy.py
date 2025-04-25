#!/usr/bin/env python3
import requests
import time
import sys
import os
from typing import Dict, Any
from dotenv import load_dotenv

load_dotenv(override=True)

RENDER_API_KEY = os.getenv('RENDER_API_KEY')
SERVICE_ID = os.getenv('SERVICE_ID')
MAX_RETRIES = 20  # 10 minutes with 50 second intervals
RETRY_INTERVAL = 50  # seconds

def get_latest_deploy() -> Dict[str, Any]:
    """Get the latest deploy from Render API"""
    url = f'https://api.render.com/v1/services/{SERVICE_ID}/deploys?limit=1'
    headers = {
        'accept': 'application/json',
        'authorization': f'Bearer {RENDER_API_KEY}'
    }
    
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    return response.json()[0]['deploy']

def wait_for_deploy() -> None:
    """Wait for the latest deploy to be live"""
    print("Waiting for deploy to complete...")
    
    for attempt in range(MAX_RETRIES):
        try:
            deploy = get_latest_deploy()
            status = deploy['status']
            
            print(f"Deploy status: {status}")
            
            if status == 'live':
                print("✅ Deploy is live!")
                return
            
            if status in ['update_failed', 'canceled']:
                print(f"❌ Deploy failed with status: {status}")
                sys.exit(1)
            
            print(f"Waiting {RETRY_INTERVAL} seconds before checking again...")
            time.sleep(RETRY_INTERVAL)
            
        except Exception as e:
            print(f"Error checking deploy status: {str(e)}")
            if attempt == MAX_RETRIES - 1:
                print("❌ Max retries reached")
                sys.exit(1)
            time.sleep(RETRY_INTERVAL)
    
    print("❌ Timeout waiting for deploy")
    sys.exit(1)

if __name__ == '__main__':
    if not RENDER_API_KEY:
        print("❌ RENDER_API_KEY environment variable is not set")
        sys.exit(1)
    wait_for_deploy() 