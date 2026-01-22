"""
FRED Data Collector

Collects unemployment and labor market data from the Federal Reserve Economic Data (FRED) API.
"""

import os
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List, Any

import pandas as pd
import requests
from loguru import logger

try:
    from fredapi import Fred
    FREDAPI_AVAILABLE = True
except ImportError:
    FREDAPI_AVAILABLE = False
    logger.warning("fredapi not installed. Using direct API calls.")


class FREDDataCollector:
    """
    Collects unemployment and labor market data from FRED.
    
    Available series:
    - UNRATE: Unemployment Rate (Monthly, Seasonally Adjusted)
    - UNRATENSA: Unemployment Rate (Monthly, Not Seasonally Adjusted)
    - ICSA: Initial Claims (Weekly, Seasonally Adjusted)
    - CCSA: Continued Claims (Weekly, Seasonally Adjusted)
    - EMRATIO: Employment-Population Ratio (Monthly)
    - U6RATE: Total Unemployed + All Marginally Attached + Part-Time for Economic Reasons
    """
    
    BASE_URL = "https://api.stlouisfed.org/fred"
    
    # Standard series IDs for unemployment data
    SERIES = {
        "unemployment_rate": "UNRATE",
        "unemployment_rate_nsa": "UNRATENSA",
        "initial_claims": "ICSA",
        "continued_claims": "CCSA",
        "employment_ratio": "EMRATIO",
        "underemployment_rate": "U6RATE",
        "labor_force_participation": "CIVPART",
        "job_openings": "JTSJOL",
    }
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the FRED data collector.
        
        Args:
            config: Configuration dictionary with API keys and settings
        """
        self.config = config
        
        # Get API key from config, environment, or raise error
        self.api_key = self._get_api_key()
        
        # Initialize fredapi client if available
        if FREDAPI_AVAILABLE:
            self._fred = Fred(api_key=self.api_key)
        else:
            self._fred = None
        
        # Get date range from config
        collection_config = config.get("collection", {})
        self.start_date = collection_config.get("start_date", "2020-03-01")
        self.end_date = collection_config.get("end_date") or datetime.now().strftime("%Y-%m-%d")
        
        # Get custom series config
        self.series_config = config.get("fred_series", {})
    
    def _get_api_key(self) -> str:
        """Get FRED API key from config or environment."""
        # Try config first
        api_keys = self.config.get("api_keys", {})
        api_key = api_keys.get("fred_api_key")
        
        if api_key and api_key != "YOUR_FRED_API_KEY_HERE":
            return api_key
        
        # Try environment variable
        api_key = os.environ.get("FRED_API_KEY")
        if api_key:
            return api_key
        
        raise ValueError(
            "FRED API key not found. Please either:\n"
            "1. Set it in config/config.yaml under api_keys.fred_api_key\n"
            "2. Set the FRED_API_KEY environment variable\n"
            "Get a free API key at: https://fred.stlouisfed.org/docs/api/api_key.html"
        )
    
    def _fetch_series_direct(
        self,
        series_id: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Fetch series data directly from FRED API without fredapi library.
        
        Args:
            series_id: FRED series ID
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            
        Returns:
            DataFrame with date index and value column
        """
        url = f"{self.BASE_URL}/series/observations"
        
        params = {
            "series_id": series_id,
            "api_key": self.api_key,
            "file_type": "json",
            "observation_start": start_date or self.start_date,
            "observation_end": end_date or self.end_date,
        }
        
        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            if "observations" not in data:
                logger.error(f"Unexpected API response for {series_id}")
                return pd.DataFrame()
            
            observations = data["observations"]
            
            df = pd.DataFrame(observations)
            
            if df.empty:
                return df
            
            # Convert date column
            df['date'] = pd.to_datetime(df['date'])
            
            # Convert value column (handle missing values marked as '.')
            df['value'] = pd.to_numeric(df['value'], errors='coerce')
            
            # Set date as index
            df = df.set_index('date')[['value']]
            df.columns = [series_id]
            
            return df
            
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed for {series_id}: {e}")
            return pd.DataFrame()
    
    def fetch_series(
        self,
        series_id: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        frequency: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Fetch a FRED data series.
        
        Args:
            series_id: FRED series ID (e.g., "UNRATE")
            start_date: Start date (YYYY-MM-DD), defaults to config value
            end_date: End date (YYYY-MM-DD), defaults to config value
            frequency: Optional frequency aggregation (d, w, bw, m, q, sa, a)
            
        Returns:
            DataFrame with date index and value column
        """
        start = start_date or self.start_date
        end = end_date or self.end_date
        
        logger.info(f"Fetching FRED series {series_id} from {start} to {end}")
        
        try:
            if FREDAPI_AVAILABLE and self._fred:
                # Use fredapi library
                kwargs = {
                    'observation_start': start,
                    'observation_end': end
                }
                if frequency:
                    kwargs['frequency'] = frequency

                series = self._fred.get_series(series_id, **kwargs)
                df = series.to_frame(name=series_id)
                df.index.name = 'date'
                return df
            else:
                # Use direct API calls
                return self._fetch_series_direct(series_id, start, end)
                
        except Exception as e:
            logger.error(f"Error fetching series {series_id}: {e}")
            return pd.DataFrame()
    
    def fetch_series_info(self, series_id: str) -> Dict[str, Any]:
        """
        Fetch metadata about a FRED series.
        
        Args:
            series_id: FRED series ID
            
        Returns:
            Dictionary with series metadata
        """
        url = f"{self.BASE_URL}/series"
        
        params = {
            "series_id": series_id,
            "api_key": self.api_key,
            "file_type": "json",
        }
        
        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            if "seriess" in data and len(data["seriess"]) > 0:
                return data["seriess"][0]
            
            return {}
            
        except Exception as e:
            logger.error(f"Error fetching series info for {series_id}: {e}")
            return {}
    
    def collect_unemployment_data(self) -> pd.DataFrame:
        """
        Collect primary unemployment rate data (UNRATE).
        
        Returns:
            DataFrame with monthly unemployment rate data
        """
        series_id = self.series_config.get("unemployment_rate", {}).get("series_id", "UNRATE")
        return self.fetch_series(series_id)
    
    def collect_initial_claims(self) -> pd.DataFrame:
        """
        Collect weekly initial jobless claims (ICSA).
        
        Returns:
            DataFrame with weekly initial claims data
        """
        series_id = self.series_config.get("initial_claims", {}).get("series_id", "ICSA")
        return self.fetch_series(series_id)
    
    def collect_continued_claims(self) -> pd.DataFrame:
        """
        Collect weekly continued jobless claims (CCSA).
        
        Returns:
            DataFrame with weekly continued claims data
        """
        series_id = self.series_config.get("continued_claims", {}).get("series_id", "CCSA")
        return self.fetch_series(series_id)
    
    def collect_all_labor_data(self) -> pd.DataFrame:
        """
        Collect all configured labor market series.
        
        Returns:
            DataFrame with all labor market indicators
        """
        all_data = []
        
        # Collect each configured series
        for name, series_config in self.series_config.items():
            series_id = series_config.get("series_id") if isinstance(series_config, dict) else series_config
            
            if series_id:
                df = self.fetch_series(series_id)
                if not df.empty:
                    all_data.append(df)
        
        # Also collect standard series not in config
        for name, series_id in self.SERIES.items():
            if series_id not in [s.get("series_id") for s in self.series_config.values() 
                                  if isinstance(s, dict)]:
                df = self.fetch_series(series_id)
                if not df.empty:
                    all_data.append(df)
        
        if not all_data:
            logger.warning("No labor data collected")
            return pd.DataFrame()
        
        # Merge all series
        result = all_data[0]
        for df in all_data[1:]:
            result = result.join(df, how='outer')
        
        return result
    
    def collect(self, series: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Main collection method.
        
        Args:
            series: Optional list of series IDs to collect. If None, collects
                   unemployment rate and initial claims.
                   
        Returns:
            DataFrame with requested series
        """
        if series is None:
            # Default to primary unemployment indicators
            series = ["UNRATE", "ICSA"]
        
        all_data = []
        
        for series_id in series:
            df = self.fetch_series(series_id)
            if not df.empty:
                all_data.append(df)
            else:
                logger.warning(f"No data collected for {series_id}")
        
        if not all_data:
            return pd.DataFrame()
        
        # Merge all series
        result = all_data[0]
        for df in all_data[1:]:
            result = result.join(df, how='outer')
        
        logger.info(f"Collected {len(result)} observations across {len(result.columns)} series")
        
        return result
    
    def save_to_file(self, df: pd.DataFrame, filepath: Path, format: str = "parquet"):
        """Save collected data to file."""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Reset index to save date as column
        df_to_save = df.reset_index()
        
        if format == "parquet":
            df_to_save.to_parquet(filepath.with_suffix('.parquet'))
        elif format == "csv":
            df_to_save.to_csv(filepath.with_suffix('.csv'), index=False)
        elif format == "both":
            df_to_save.to_parquet(filepath.with_suffix('.parquet'))
            df_to_save.to_csv(filepath.with_suffix('.csv'), index=False)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logger.info(f"Saved {len(df)} records to {filepath}")
    
    def load_from_file(self, filepath: Path) -> pd.DataFrame:
        """Load previously collected data from file."""
        if not filepath.exists():
            logger.warning(f"Data file not found: {filepath}")
            return pd.DataFrame()
        
        if filepath.suffix == '.parquet':
            df = pd.read_parquet(filepath)
        elif filepath.suffix == '.csv':
            df = pd.read_csv(filepath, parse_dates=['date'])
        else:
            raise ValueError(f"Unsupported file format: {filepath.suffix}")
        
        # Set date as index if present
        if 'date' in df.columns:
            df = df.set_index('date')
        
        return df


# Example usage and testing
if __name__ == "__main__":
    import yaml
    
    # Load config
    config_path = Path(__file__).parent.parent.parent / "config" / "config.yaml"
    if config_path.exists():
        with open(config_path) as f:
            config = yaml.safe_load(f)
    else:
        # Minimal config for testing
        config = {
            "collection": {
                "start_date": "2020-03-01"
            }
        }
    
    try:
        collector = FREDDataCollector(config)
        
        # Test unemployment rate collection
        print("Testing unemployment rate collection...")
        df = collector.collect_unemployment_data()
        if not df.empty:
            print(f"Collected {len(df)} observations")
            print(df.head())
        else:
            print("No data collected")
            
    except ValueError as e:
        print(f"Configuration error: {e}")
