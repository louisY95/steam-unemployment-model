"""
AWS Lambda handler for Steam data collection.

Deploy this to AWS Lambda and set up CloudWatch Events to trigger hourly.

Setup:
1. Create Lambda function (Python 3.10)
2. Add Layer with dependencies (pandas, requests, selenium, etc.)
3. Set environment variable: FRED_API_KEY
4. Create CloudWatch Event Rule: rate(1 hour)
5. Configure S3 bucket for data storage
"""

import json
import os
import boto3
from datetime import datetime

# Your existing imports
import sys
sys.path.insert(0, '/opt/python')  # Lambda layer path

from src.collectors.steam_collector import SteamDataCollector
from src.collectors.fred_collector import FREDDataCollector
import pandas as pd


def lambda_handler(event, context):
    """Lambda function handler."""

    # Initialize S3 client
    s3 = boto3.client('s3')
    bucket_name = os.environ.get('S3_BUCKET', 'steam-unemployment-data')

    # Configuration
    config = {
        'api_keys': {
            'fred_api_key': os.environ.get('FRED_API_KEY')
        },
        'collection': {},
        'browser': {'headless': True}
    }

    try:
        # Collect Steam data
        steam_collector = SteamDataCollector(config)
        steam_df = steam_collector.collect(method='steam_current')

        # Save to S3
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        s3_key = f'raw/steam/steam_{timestamp}.parquet'

        # Convert to parquet bytes
        parquet_buffer = steam_df.to_parquet()
        s3.put_object(Bucket=bucket_name, Key=s3_key, Body=parquet_buffer)

        return {
            'statusCode': 200,
            'body': json.dumps({
                'message': 'Data collected successfully',
                'records': len(steam_df),
                's3_key': s3_key
            })
        }

    except Exception as e:
        return {
            'statusCode': 500,
            'body': json.dumps({
                'error': str(e)
            })
        }
