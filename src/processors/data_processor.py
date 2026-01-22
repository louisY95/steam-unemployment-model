"""
Data Processor

Handles filtering Steam data for US working hours, aggregating data to match
unemployment frequency, and preparing datasets for statistical analysis.
"""

from datetime import datetime, time
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import numpy as np
import pandas as pd
import pytz
from loguru import logger


class DataProcessor:
    """
    Processes Steam and unemployment data for analysis.
    
    Key functions:
    1. Filter Steam data to US working hours (8am-5pm EST and PST)
    2. Aggregate hourly Steam data to weekly/monthly to match unemployment frequency
    3. Merge Steam and unemployment datasets
    4. Handle timezone conversions
    5. Create lagged features for predictive modeling
    """
    
    # US Time Zones
    TZ_EAST = pytz.timezone('America/New_York')
    TZ_WEST = pytz.timezone('America/Los_Angeles')
    TZ_UTC = pytz.UTC
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the data processor.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        
        # Get working hours config
        wh_config = config.get("working_hours", {})
        
        self.east_config = wh_config.get("east_coast", {
            "start_hour": 8,
            "end_hour": 17,
            "timezone": "America/New_York"
        })
        
        self.west_config = wh_config.get("west_coast", {
            "start_hour": 8,
            "end_hour": 17,
            "timezone": "America/Los_Angeles"
        })
        
        self.include_weekends = wh_config.get("include_weekends", False)
    
    def localize_timestamps(
        self,
        df: pd.DataFrame,
        timestamp_col: str = 'timestamp',
        source_tz: str = 'UTC'
    ) -> pd.DataFrame:
        """
        Ensure timestamps are timezone-aware.
        
        Args:
            df: DataFrame with timestamp column
            timestamp_col: Name of timestamp column
            source_tz: Source timezone (default UTC, which is what SteamDB uses)
            
        Returns:
            DataFrame with localized timestamps
        """
        df = df.copy()
        
        # Ensure timestamp column exists
        if timestamp_col not in df.columns:
            if df.index.name == timestamp_col or isinstance(df.index, pd.DatetimeIndex):
                df = df.reset_index()
        
        # Convert to datetime if not already
        df[timestamp_col] = pd.to_datetime(df[timestamp_col])
        
        # Localize if not already timezone-aware
        if df[timestamp_col].dt.tz is None:
            source_timezone = pytz.timezone(source_tz)
            df[timestamp_col] = df[timestamp_col].dt.tz_localize(source_timezone)
        
        return df
    
    def filter_us_working_hours(
        self,
        df: pd.DataFrame,
        timestamp_col: str = 'timestamp',
        value_col: str = 'concurrent_users'
    ) -> pd.DataFrame:
        """
        Filter Steam data to US working hours (8am-5pm) for both coasts.
        
        Working hours are defined as:
        - East Coast: 8:00 AM - 5:00 PM EST/EDT (13:00 - 22:00 UTC)
        - West Coast: 8:00 AM - 5:00 PM PST/PDT (16:00 - 01:00 UTC next day)
        
        We capture hours that fall within working hours for EITHER coast,
        giving us a window from 8am PST to 5pm EST.
        
        Args:
            df: DataFrame with timestamp and value columns
            timestamp_col: Name of timestamp column
            value_col: Name of value column
            
        Returns:
            DataFrame filtered to US working hours
        """
        df = self.localize_timestamps(df, timestamp_col)
        df = df.copy()
        
        # Convert to both timezones
        df['hour_east'] = df[timestamp_col].dt.tz_convert(self.TZ_EAST).dt.hour
        df['hour_west'] = df[timestamp_col].dt.tz_convert(self.TZ_WEST).dt.hour
        df['day_of_week'] = df[timestamp_col].dt.tz_convert(self.TZ_EAST).dt.dayofweek
        
        # Working hours bounds
        east_start = self.east_config.get("start_hour", 8)
        east_end = self.east_config.get("end_hour", 17)
        west_start = self.west_config.get("start_hour", 8)
        west_end = self.west_config.get("end_hour", 17)
        
        # Filter for working hours on either coast
        # A timestamp is "working hours" if it falls within 8am-5pm in EITHER timezone
        is_east_working = (df['hour_east'] >= east_start) & (df['hour_east'] < east_end)
        is_west_working = (df['hour_west'] >= west_start) & (df['hour_west'] < west_end)
        
        # Combined: either coast is in working hours
        is_working_hours = is_east_working | is_west_working
        
        # Filter weekends if configured
        if not self.include_weekends:
            is_weekday = df['day_of_week'] < 5  # Monday=0, Friday=4
            is_working_hours = is_working_hours & is_weekday
        
        # Apply filter
        filtered_df = df[is_working_hours].copy()
        
        # Clean up temporary columns
        filtered_df = filtered_df.drop(columns=['hour_east', 'hour_west', 'day_of_week'])
        
        logger.info(f"Filtered to working hours: {len(filtered_df)}/{len(df)} records "
                   f"({100*len(filtered_df)/len(df):.1f}%)")
        
        return filtered_df
    
    def filter_east_coast_hours(
        self,
        df: pd.DataFrame,
        timestamp_col: str = 'timestamp'
    ) -> pd.DataFrame:
        """
        Filter to East Coast working hours only (8am-5pm EST/EDT).
        
        Args:
            df: DataFrame with timestamp column
            timestamp_col: Name of timestamp column
            
        Returns:
            DataFrame filtered to East Coast working hours
        """
        df = self.localize_timestamps(df, timestamp_col)
        df = df.copy()
        
        df['hour_east'] = df[timestamp_col].dt.tz_convert(self.TZ_EAST).dt.hour
        df['day_of_week'] = df[timestamp_col].dt.tz_convert(self.TZ_EAST).dt.dayofweek
        
        east_start = self.east_config.get("start_hour", 8)
        east_end = self.east_config.get("end_hour", 17)
        
        mask = (df['hour_east'] >= east_start) & (df['hour_east'] < east_end)
        
        if not self.include_weekends:
            mask = mask & (df['day_of_week'] < 5)
        
        return df[mask].drop(columns=['hour_east', 'day_of_week'])
    
    def filter_west_coast_hours(
        self,
        df: pd.DataFrame,
        timestamp_col: str = 'timestamp'
    ) -> pd.DataFrame:
        """
        Filter to West Coast working hours only (8am-5pm PST/PDT).
        
        Args:
            df: DataFrame with timestamp column
            timestamp_col: Name of timestamp column
            
        Returns:
            DataFrame filtered to West Coast working hours
        """
        df = self.localize_timestamps(df, timestamp_col)
        df = df.copy()
        
        df['hour_west'] = df[timestamp_col].dt.tz_convert(self.TZ_WEST).dt.hour
        df['day_of_week'] = df[timestamp_col].dt.tz_convert(self.TZ_WEST).dt.dayofweek
        
        west_start = self.west_config.get("start_hour", 8)
        west_end = self.west_config.get("end_hour", 17)
        
        mask = (df['hour_west'] >= west_start) & (df['hour_west'] < west_end)
        
        if not self.include_weekends:
            mask = mask & (df['day_of_week'] < 5)
        
        return df[mask].drop(columns=['hour_west', 'day_of_week'])
    
    def aggregate_hourly_to_daily(
        self,
        df: pd.DataFrame,
        timestamp_col: str = 'timestamp',
        value_col: str = 'concurrent_users',
        agg_func: str = 'mean'
    ) -> pd.DataFrame:
        """
        Aggregate hourly data to daily.
        
        Args:
            df: DataFrame with hourly data
            timestamp_col: Timestamp column name
            value_col: Value column name
            agg_func: Aggregation function ('mean', 'sum', 'max', 'min', 'median')
            
        Returns:
            DataFrame with daily aggregated data
        """
        df = df.copy()
        
        # Extract date
        df['date'] = pd.to_datetime(df[timestamp_col]).dt.date
        
        # Aggregate
        agg_dict = {
            value_col: agg_func,
            timestamp_col: 'count'  # Count of observations
        }
        
        daily = df.groupby('date').agg(agg_dict).reset_index()
        daily.columns = ['date', f'{value_col}_{agg_func}', 'observation_count']
        daily['date'] = pd.to_datetime(daily['date'])
        
        return daily
    
    def aggregate_hourly_to_weekly(
        self,
        df: pd.DataFrame,
        timestamp_col: str = 'timestamp',
        value_col: str = 'concurrent_users',
        agg_func: str = 'mean'
    ) -> pd.DataFrame:
        """
        Aggregate hourly data to weekly (aligned with FRED initial claims release).
        
        Args:
            df: DataFrame with hourly data
            timestamp_col: Timestamp column name
            value_col: Value column name
            agg_func: Aggregation function
            
        Returns:
            DataFrame with weekly aggregated data
        """
        df = df.copy()
        
        # Convert to datetime and set as index
        df[timestamp_col] = pd.to_datetime(df[timestamp_col])
        df = df.set_index(timestamp_col)
        
        # Resample to weekly (week ending Saturday to align with FRED)
        weekly = df[[value_col]].resample('W-SAT').agg(agg_func)
        weekly = weekly.reset_index()
        weekly.columns = ['week_ending', f'{value_col}_{agg_func}']
        
        # Also add count of observations
        counts = df[[value_col]].resample('W-SAT').count()
        weekly['observation_count'] = counts.values
        
        return weekly
    
    def aggregate_hourly_to_monthly(
        self,
        df: pd.DataFrame,
        timestamp_col: str = 'timestamp',
        value_col: str = 'concurrent_users',
        agg_func: str = 'mean'
    ) -> pd.DataFrame:
        """
        Aggregate hourly data to monthly (to match unemployment rate frequency).
        
        Args:
            df: DataFrame with hourly data
            timestamp_col: Timestamp column name
            value_col: Value column name
            agg_func: Aggregation function
            
        Returns:
            DataFrame with monthly aggregated data
        """
        df = df.copy()
        
        # Convert to datetime and set as index
        df[timestamp_col] = pd.to_datetime(df[timestamp_col])
        df = df.set_index(timestamp_col)
        
        # Resample to monthly (month start to align with FRED)
        monthly = df[[value_col]].resample('MS').agg(agg_func)
        monthly = monthly.reset_index()
        monthly.columns = ['month', f'{value_col}_{agg_func}']

        # Remove timezone to match FRED data
        if monthly['month'].dt.tz is not None:
            monthly['month'] = monthly['month'].dt.tz_localize(None)
        
        # Add statistics
        stats = df[[value_col]].resample('MS').agg(['count', 'std', 'min', 'max'])
        monthly['observation_count'] = stats[(value_col, 'count')].values
        monthly[f'{value_col}_std'] = stats[(value_col, 'std')].values
        monthly[f'{value_col}_min'] = stats[(value_col, 'min')].values
        monthly[f'{value_col}_max'] = stats[(value_col, 'max')].values
        
        return monthly
    
    def merge_steam_unemployment(
        self,
        steam_df: pd.DataFrame,
        unemployment_df: pd.DataFrame,
        steam_date_col: str = 'month',
        steam_value_col: str = 'concurrent_users_mean',
        unemployment_series: str = 'UNRATE'
    ) -> pd.DataFrame:
        """
        Merge Steam and unemployment data on date.
        
        Args:
            steam_df: Aggregated Steam data (monthly or weekly)
            unemployment_df: FRED unemployment data
            steam_date_col: Date column name in Steam data
            steam_value_col: Value column name in Steam data
            unemployment_series: FRED series ID for unemployment
            
        Returns:
            Merged DataFrame ready for analysis
        """
        # Ensure date columns are datetime
        steam_df = steam_df.copy()
        unemployment_df = unemployment_df.copy()
        
        steam_df[steam_date_col] = pd.to_datetime(steam_df[steam_date_col])
        
        # Reset index on unemployment if needed
        if unemployment_df.index.name == 'date' or isinstance(unemployment_df.index, pd.DatetimeIndex):
            unemployment_df = unemployment_df.reset_index()
        
        unemployment_df['date'] = pd.to_datetime(unemployment_df['date'])
        
        # Merge on date
        merged = steam_df.merge(
            unemployment_df[['date', unemployment_series]],
            left_on=steam_date_col,
            right_on='date',
            how='inner'
        )
        
        # Clean up
        if steam_date_col != 'date':
            merged = merged.drop(columns=['date'])
        
        logger.info(f"Merged dataset: {len(merged)} observations")
        
        return merged
    
    def create_lagged_features(
        self,
        df: pd.DataFrame,
        columns: List[str],
        lags: List[int],
        date_col: str = 'month'
    ) -> pd.DataFrame:
        """
        Create lagged features for predictive modeling.
        
        Args:
            df: DataFrame with time series data
            columns: Columns to create lags for
            lags: List of lag periods
            date_col: Date column name
            
        Returns:
            DataFrame with lagged features added
        """
        df = df.copy()
        df = df.sort_values(date_col)
        
        for col in columns:
            for lag in lags:
                df[f'{col}_lag{lag}'] = df[col].shift(lag)
        
        return df
    
    def create_change_features(
        self,
        df: pd.DataFrame,
        columns: List[str],
        periods: List[int] = [1],
        date_col: str = 'month'
    ) -> pd.DataFrame:
        """
        Create change/difference features (useful for stationarity).
        
        Args:
            df: DataFrame with time series data
            columns: Columns to create changes for
            periods: List of difference periods
            date_col: Date column name
            
        Returns:
            DataFrame with change features added
        """
        df = df.copy()
        df = df.sort_values(date_col)
        
        for col in columns:
            for period in periods:
                # Absolute change
                df[f'{col}_change{period}'] = df[col].diff(period)
                # Percentage change
                df[f'{col}_pct_change{period}'] = df[col].pct_change(period)
        
        return df
    
    def prepare_analysis_dataset(
        self,
        steam_hourly: pd.DataFrame,
        unemployment: pd.DataFrame,
        steam_timestamp_col: str = 'timestamp',
        steam_value_col: str = 'concurrent_users'
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Prepare complete analysis datasets:
        1. Working hours filtered Steam data merged with unemployment
        2. Total Steam data merged with unemployment
        
        Args:
            steam_hourly: Raw hourly Steam data
            unemployment: FRED unemployment data
            steam_timestamp_col: Steam timestamp column
            steam_value_col: Steam value column
            
        Returns:
            Tuple of (working_hours_dataset, total_dataset)
        """
        # 1. Process working hours data
        logger.info("Processing working hours filtered data...")
        working_hours = self.filter_us_working_hours(
            steam_hourly, 
            steam_timestamp_col, 
            steam_value_col
        )
        wh_monthly = self.aggregate_hourly_to_monthly(
            working_hours, 
            steam_timestamp_col, 
            steam_value_col
        )
        wh_merged = self.merge_steam_unemployment(
            wh_monthly,
            unemployment,
            steam_date_col='month',
            steam_value_col=f'{steam_value_col}_mean'
        )
        
        # 2. Process total data
        logger.info("Processing total data...")
        total_monthly = self.aggregate_hourly_to_monthly(
            steam_hourly, 
            steam_timestamp_col, 
            steam_value_col
        )
        total_merged = self.merge_steam_unemployment(
            total_monthly,
            unemployment,
            steam_date_col='month',
            steam_value_col=f'{steam_value_col}_mean'
        )
        
        # 3. Add lagged features to both
        steam_col = f'{steam_value_col}_mean'
        lags = [1, 2, 3, 6, 12]
        
        wh_merged = self.create_lagged_features(wh_merged, [steam_col, 'UNRATE'], lags)
        total_merged = self.create_lagged_features(total_merged, [steam_col, 'UNRATE'], lags)
        
        # 4. Add change features
        wh_merged = self.create_change_features(wh_merged, [steam_col, 'UNRATE'])
        total_merged = self.create_change_features(total_merged, [steam_col, 'UNRATE'])
        
        return wh_merged, total_merged
    
    def save_processed_data(
        self,
        df: pd.DataFrame,
        filepath: Path,
        format: str = "parquet"
    ):
        """Save processed data to file."""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        if format == "parquet":
            df.to_parquet(filepath.with_suffix('.parquet'), index=False)
        elif format == "csv":
            df.to_csv(filepath.with_suffix('.csv'), index=False)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logger.info(f"Saved processed data to {filepath}")


# Example usage
if __name__ == "__main__":
    # Create sample data for testing
    import yaml
    
    # Sample hourly Steam data (simulated)
    dates = pd.date_range(start='2023-01-01', end='2023-03-01', freq='H')
    np.random.seed(42)
    steam_df = pd.DataFrame({
        'timestamp': dates,
        'concurrent_users': np.random.randint(20_000_000, 35_000_000, len(dates))
    })
    
    # Load config
    config = {}
    
    processor = DataProcessor(config)
    
    # Test filtering
    print("Testing US working hours filter...")
    filtered = processor.filter_us_working_hours(steam_df)
    print(f"Original: {len(steam_df)}, Filtered: {len(filtered)}")
    
    # Test aggregation
    print("\nTesting monthly aggregation...")
    monthly = processor.aggregate_hourly_to_monthly(steam_df)
    print(monthly.head())
