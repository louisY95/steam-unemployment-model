"""
Steam Data Collector

Collects total concurrent Steam user data from SteamDB and Steam official sources.
This module focuses on total platform users (app 753), not individual game data.
"""

import json
import re
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, List, Any
from dataclasses import dataclass

import pandas as pd
import requests
from bs4 import BeautifulSoup
from loguru import logger

try:
    from selenium import webdriver
    from selenium.webdriver.chrome.service import Service
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from webdriver_manager.chrome import ChromeDriverManager
    SELENIUM_AVAILABLE = True
    ChromeDriver = webdriver.Chrome
except ImportError:
    SELENIUM_AVAILABLE = False
    ChromeDriver = Any  # type: ignore
    logger.warning("Selenium not available. Some collection methods will be limited.")


@dataclass
class SteamDataPoint:
    """Single data point of Steam concurrent users."""
    timestamp: datetime
    concurrent_users: int
    source: str


class SteamDataCollector:
    """
    Collects total concurrent Steam user data from multiple sources.
    
    Primary source: SteamDB (https://steamdb.info/app/753/charts/)
    Backup source: Steam Stats page (https://store.steampowered.com/stats/)
    
    The app ID 753 represents the "Steam" application itself, which tracks
    total concurrent users across the entire platform.
    """
    
    STEAMDB_CHART_URL = "https://steamdb.info/app/753/charts/"
    STEAM_STATS_URL = "https://store.steampowered.com/stats/"
    STEAM_API_CHART_URL = "https://steamdb.info/api/GetGraph/?type=concurrent_max&appid=753"
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the Steam data collector.
        
        Args:
            config: Configuration dictionary with collection settings
        """
        self.config = config
        self.collection_config = config.get("collection", {})
        self.browser_config = config.get("browser", {})
        self.start_date = self._parse_date(self.collection_config.get("start_date", "2020-03-01"))
        self.end_date = self._parse_date(self.collection_config.get("end_date")) or datetime.now()
        self.request_delay = self.collection_config.get("request_delay", 2.0)
        self.max_retries = self.collection_config.get("max_retries", 3)
        
        self._driver: Optional[ChromeDriver] = None
        
    def _parse_date(self, date_str: Optional[str]) -> Optional[datetime]:
        """Parse date string to datetime object."""
        if not date_str:
            return None
        try:
            return datetime.strptime(date_str, "%Y-%m-%d")
        except ValueError:
            logger.error(f"Invalid date format: {date_str}. Expected YYYY-MM-DD")
            return None
    
    def _init_selenium_driver(self) -> ChromeDriver:
        """Initialize Selenium WebDriver with Chrome."""
        if not SELENIUM_AVAILABLE:
            raise RuntimeError("Selenium is not installed. Run: pip install selenium webdriver-manager")
        
        chrome_options = Options()
        
        if self.browser_config.get("headless", True):
            chrome_options.add_argument("--headless=new")
        
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--window-size=1920,1080")
        
        user_agent = self.browser_config.get(
            "user_agent",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )
        chrome_options.add_argument(f"--user-agent={user_agent}")
        
        # Disable automation detection
        chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
        chrome_options.add_experimental_option("useAutomationExtension", False)
        
        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=chrome_options)
        
        # Additional stealth measures
        driver.execute_cdp_cmd("Page.addScriptToEvaluateOnNewDocument", {
            "source": """
                Object.defineProperty(navigator, 'webdriver', {
                    get: () => undefined
                });
            """
        })
        
        timeout = self.browser_config.get("timeout", 30)
        driver.set_page_load_timeout(timeout)
        driver.implicitly_wait(10)
        
        return driver
    
    def _get_driver(self) -> ChromeDriver:
        """Get or create Selenium driver."""
        if self._driver is None:
            self._driver = self._init_selenium_driver()
        return self._driver
    
    def _close_driver(self):
        """Close Selenium driver if open."""
        if self._driver is not None:
            try:
                self._driver.quit()
            except Exception:
                pass
            self._driver = None
    
    def collect_from_steamdb_selenium(self) -> pd.DataFrame:
        """
        Collect historical Steam concurrent user data from SteamDB using Selenium.
        
        SteamDB embeds chart data in JavaScript that requires rendering.
        This method extracts the data from the rendered page.
        
        Returns:
            DataFrame with timestamp and concurrent_users columns
        """
        logger.info("Collecting Steam data from SteamDB via Selenium...")
        
        driver = self._get_driver()
        data_points: List[SteamDataPoint] = []
        
        try:
            # Load the SteamDB charts page
            driver.get(self.STEAMDB_CHART_URL)
            time.sleep(3)  # Wait for initial page load
            
            # Wait for chart to render
            WebDriverWait(driver, 20).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, ".highcharts-series"))
            )
            
            # Additional wait for data to fully load
            time.sleep(2)
            
            # Try to extract chart data from page source
            page_source = driver.page_source
            
            # Method 1: Extract from Highcharts data in JavaScript
            data_points = self._extract_highcharts_data(page_source)
            
            if not data_points:
                # Method 2: Try to extract from SVG elements
                logger.info("Trying SVG extraction method...")
                data_points = self._extract_from_svg(driver)
            
            if not data_points:
                # Method 3: Execute JavaScript to get data directly
                logger.info("Trying JavaScript extraction method...")
                data_points = self._extract_via_javascript(driver)
            
            logger.info(f"Collected {len(data_points)} data points from SteamDB")
            
        except Exception as e:
            logger.error(f"Error collecting from SteamDB: {e}")
            raise
        finally:
            self._close_driver()
        
        return self._to_dataframe(data_points)
    
    def _extract_highcharts_data(self, page_source: str) -> List[SteamDataPoint]:
        """Extract chart data from Highcharts JavaScript in page source."""
        data_points = []
        
        # Pattern to match Highcharts series data
        # SteamDB typically stores data as arrays of [timestamp, value] pairs
        patterns = [
            r'data:\s*\[\[([\d,\[\]\s]+)\]\]',
            r'"data":\s*\[\[([\d,\[\]\s]+)\]\]',
            r'series:\s*\[\{[^}]*data:\s*\[([\[\]\d,\s]+)\]',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, page_source, re.DOTALL)
            for match in matches:
                try:
                    # Clean and parse the data
                    data_str = f"[{match}]"
                    data_str = re.sub(r'\s+', '', data_str)
                    
                    # Try to parse as JSON
                    data = json.loads(data_str)
                    
                    for item in data:
                        if isinstance(item, list) and len(item) >= 2:
                            timestamp_ms = item[0]
                            value = item[1]
                            
                            if isinstance(timestamp_ms, (int, float)) and timestamp_ms > 1000000000000:
                                # Timestamp is in milliseconds
                                timestamp = datetime.fromtimestamp(timestamp_ms / 1000)
                                if self.start_date <= timestamp <= self.end_date:
                                    data_points.append(SteamDataPoint(
                                        timestamp=timestamp,
                                        concurrent_users=int(value),
                                        source="steamdb_highcharts"
                                    ))
                except (json.JSONDecodeError, ValueError, TypeError) as e:
                    continue
        
        return data_points
    
    def _extract_from_svg(self, driver: ChromeDriver) -> List[SteamDataPoint]:
        """Extract data points from SVG chart elements."""
        data_points = []
        
        try:
            # Find the chart container
            chart = driver.find_element(By.CSS_SELECTOR, "#chart-concurrent")
            
            # Get chart dimensions for scaling
            chart_width = chart.size['width']
            chart_height = chart.size['height']
            
            # Find data points in the SVG
            path_elements = chart.find_elements(By.CSS_SELECTOR, ".highcharts-series path")
            
            for path in path_elements:
                d_attr = path.get_attribute('d')
                if d_attr:
                    # Parse SVG path data - this is complex and may require additional logic
                    # depending on the exact SVG structure
                    pass
                    
        except Exception as e:
            logger.debug(f"SVG extraction failed: {e}")
        
        return data_points
    
    def _extract_via_javascript(self, driver: ChromeDriver) -> List[SteamDataPoint]:
        """Execute JavaScript to extract chart data directly from Highcharts object."""
        data_points = []
        
        try:
            # Try to access Highcharts chart object
            js_scripts = [
                "return Highcharts.charts[0].series[0].data.map(p => [p.x, p.y]);",
                "return Highcharts.charts.find(c => c).series[0].data.map(p => [p.x, p.y]);",
                "return document.querySelector('#chart-concurrent').__highcharts__.series[0].data.map(p => [p.x, p.y]);",
            ]
            
            for script in js_scripts:
                try:
                    data = driver.execute_script(script)
                    if data and isinstance(data, list):
                        for item in data:
                            if isinstance(item, list) and len(item) >= 2:
                                timestamp = datetime.fromtimestamp(item[0] / 1000)
                                if self.start_date <= timestamp <= self.end_date:
                                    data_points.append(SteamDataPoint(
                                        timestamp=timestamp,
                                        concurrent_users=int(item[1]),
                                        source="steamdb_js"
                                    ))
                        if data_points:
                            break
                except Exception:
                    continue
                    
        except Exception as e:
            logger.debug(f"JavaScript extraction failed: {e}")
        
        return data_points
    
    def collect_from_steamdb_api(self) -> pd.DataFrame:
        """
        Attempt to collect data from SteamDB's internal API.
        
        Note: This endpoint may require authentication or may not be publicly accessible.
        Falls back to scraping if API is unavailable.
        
        Returns:
            DataFrame with timestamp and concurrent_users columns
        """
        logger.info("Attempting to collect from SteamDB API...")
        
        headers = {
            "User-Agent": self.browser_config.get("user_agent", "Mozilla/5.0"),
            "Accept": "application/json",
            "Referer": self.STEAMDB_CHART_URL,
        }
        
        data_points: List[SteamDataPoint] = []
        
        try:
            response = requests.get(
                self.STEAM_API_CHART_URL,
                headers=headers,
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                
                if "data" in data and isinstance(data["data"], dict):
                    for timestamp_str, value in data["data"].items():
                        timestamp = datetime.fromtimestamp(int(timestamp_str))
                        if self.start_date <= timestamp <= self.end_date:
                            data_points.append(SteamDataPoint(
                                timestamp=timestamp,
                                concurrent_users=int(value),
                                source="steamdb_api"
                            ))
                            
                logger.info(f"Collected {len(data_points)} data points from SteamDB API")
            else:
                logger.warning(f"SteamDB API returned status {response.status_code}")
                
        except Exception as e:
            logger.warning(f"SteamDB API collection failed: {e}")
        
        return self._to_dataframe(data_points)
    
    def collect_current_from_steam(self) -> Optional[SteamDataPoint]:
        """
        Collect current concurrent user count from Steam using multiple fallback methods.

        This is useful for building up historical data over time through polling.

        Returns:
            Single data point with current timestamp and user count, or None if all methods fail
        """
        logger.debug("Collecting current data from Steam...")

        # Try multiple methods in order of reliability
        methods = [
            self._try_steamcharts_scrape,
            self._try_steam_web_api,
            self._try_steam_stats_page,
        ]

        for method in methods:
            try:
                result = method()
                if result:
                    logger.info(f"Successfully collected data using {method.__name__}")
                    return result
            except Exception as e:
                logger.debug(f"{method.__name__} failed: {e}")
                continue

        logger.warning("All Steam collection methods failed")
        return None

    def _try_steam_web_api(self) -> Optional[SteamDataPoint]:
        """Try Steam Web API for concurrent users (app 753 is Steam itself)"""
        r = requests.get(
            'https://api.steampowered.com/ISteamUserStats/GetNumberOfCurrentPlayers/v1/?appid=753',
            timeout=10
        )
        if r.status_code == 200:
            data = r.json()
            if 'response' in data and 'player_count' in data['response']:
                # Note: App 753 players * multiplier to estimate total Steam users
                # Based on typical ratio, multiply by ~50,000
                count = data['response']['player_count'] * 50000
                logger.info(f"Steam Web API estimate: {count:,} users")
                return SteamDataPoint(
                    timestamp=datetime.now(),
                    concurrent_users=count,
                    source="steam_web_api_estimated"
                )
        return None

    def _try_steamcharts_scrape(self) -> Optional[SteamDataPoint]:
        """Try scraping SteamCharts for current online users"""
        r = requests.get('https://steamcharts.com/', timeout=10)
        if r.status_code == 200:
            soup = BeautifulSoup(r.text, 'lxml')
            # Look for "Online Now" in the page
            text = soup.get_text()
            match = re.search(r'(\d{1,3}(?:,\d{3})+)\s+Online', text)
            if match:
                count = int(match.group(1).replace(',', ''))
                logger.info(f"SteamCharts: {count:,} users online")
                return SteamDataPoint(
                    timestamp=datetime.now(),
                    concurrent_users=count,
                    source="steamcharts"
                )
        return None

    def _try_steam_stats_page(self) -> Optional[SteamDataPoint]:
        """Original method: Try Steam's official stats page"""
        headers = {
            "User-Agent": self.browser_config.get("user_agent", "Mozilla/5.0"),
            "Accept": "text/html,application/xhtml+xml",
        }

        response = requests.get(
            self.STEAM_STATS_URL,
            headers=headers,
            timeout=30
        )

        if response.status_code != 200:
            logger.warning(f"Steam stats page returned status {response.status_code}")
            return None

        soup = BeautifulSoup(response.text, 'lxml')

        # Find the concurrent users count
        stats_span = soup.find('span', class_='statsCountConcurrent')

        if not stats_span:
            # Alternative: look in the page text for the pattern
            text = soup.get_text()
            match = re.search(r'(\d{1,3}(?:,\d{3})*)\s*(?:users?|players?)\s*(?:online|playing)',
                              text, re.IGNORECASE)
            if match:
                count_str = match.group(1).replace(',', '')
                return SteamDataPoint(
                    timestamp=datetime.now(),
                    concurrent_users=int(count_str),
                    source="steam_stats_page"
                )
        else:
            count_str = stats_span.get_text().strip().replace(',', '')
            return SteamDataPoint(
                timestamp=datetime.now(),
                concurrent_users=int(count_str),
                source="steam_stats_page"
            )

        # Also try to extract from the JSON data embedded in the page
        script_tags = soup.find_all('script')
        for script in script_tags:
            script_text = script.string or ""
            match = re.search(r'"player_count":\s*(\d+)', script_text)
            if match:
                return SteamDataPoint(
                    timestamp=datetime.now(),
                    concurrent_users=int(match.group(1)),
                    source="steam_stats_json"
                )

        logger.warning("Could not find concurrent user count on Steam stats page")
        return None
    
    def start_continuous_collection(self, output_path: Path, interval_seconds: int = 3600):
        """
        Start continuous collection of Steam data by polling at regular intervals.
        
        This method will run indefinitely, collecting data points and saving to a file.
        Use Ctrl+C to stop.
        
        Args:
            output_path: Path to save collected data
            interval_seconds: Seconds between collection attempts (default: 1 hour)
        """
        logger.info(f"Starting continuous collection with {interval_seconds}s interval...")
        
        # Load existing data if file exists
        if output_path.exists():
            existing_df = pd.read_parquet(output_path)
            data_points = [
                SteamDataPoint(
                    timestamp=row['timestamp'],
                    concurrent_users=row['concurrent_users'],
                    source=row.get('source', 'existing')
                )
                for _, row in existing_df.iterrows()
            ]
            logger.info(f"Loaded {len(data_points)} existing data points")
        else:
            data_points = []
        
        try:
            while True:
                data_point = self.collect_current_from_steam()
                
                if data_point:
                    data_points.append(data_point)
                    df = self._to_dataframe(data_points)
                    df.to_parquet(output_path)
                    logger.info(f"Collected: {data_point.timestamp} - {data_point.concurrent_users:,} users")
                else:
                    logger.warning("Failed to collect data point")
                
                time.sleep(interval_seconds)
                
        except KeyboardInterrupt:
            logger.info("Continuous collection stopped by user")
            if data_points:
                df = self._to_dataframe(data_points)
                df.to_parquet(output_path)
                logger.info(f"Saved {len(data_points)} total data points")
    
    def collect(self, method: str = "auto") -> pd.DataFrame:
        """
        Collect Steam concurrent user data using the specified method.
        
        Args:
            method: Collection method - "steamdb_selenium", "steamdb_api", "steam_current", or "auto"
        
        Returns:
            DataFrame with timestamp and concurrent_users columns
        """
        if method == "auto":
            # Try methods in order of preference
            methods = ["steamdb_api", "steamdb_selenium"]
            
            for m in methods:
                try:
                    df = self.collect(method=m)
                    if not df.empty:
                        return df
                except Exception as e:
                    logger.warning(f"Method {m} failed: {e}")
                    continue
            
            logger.error("All collection methods failed")
            return pd.DataFrame(columns=['timestamp', 'concurrent_users', 'source'])
        
        elif method == "steamdb_selenium":
            return self.collect_from_steamdb_selenium()
        
        elif method == "steamdb_api":
            return self.collect_from_steamdb_api()
        
        elif method == "steam_current":
            data_point = self.collect_current_from_steam()
            if data_point:
                return self._to_dataframe([data_point])
            return pd.DataFrame(columns=['timestamp', 'concurrent_users', 'source'])
        
        else:
            raise ValueError(f"Unknown collection method: {method}")
    
    def _to_dataframe(self, data_points: List[SteamDataPoint]) -> pd.DataFrame:
        """Convert list of data points to DataFrame."""
        if not data_points:
            return pd.DataFrame(columns=['timestamp', 'concurrent_users', 'source'])
        
        df = pd.DataFrame([
            {
                'timestamp': dp.timestamp,
                'concurrent_users': dp.concurrent_users,
                'source': dp.source
            }
            for dp in data_points
        ])
        
        # Ensure timestamp is datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Sort by timestamp
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # Remove duplicates (keep first)
        df = df.drop_duplicates(subset=['timestamp'], keep='first')
        
        return df
    
    def load_from_file(self, filepath: Path) -> pd.DataFrame:
        """Load previously collected data from file."""
        if not filepath.exists():
            logger.warning(f"Data file not found: {filepath}")
            return pd.DataFrame(columns=['timestamp', 'concurrent_users', 'source'])
        
        if filepath.suffix == '.parquet':
            return pd.read_parquet(filepath)
        elif filepath.suffix == '.csv':
            df = pd.read_csv(filepath, parse_dates=['timestamp'])
            return df
        else:
            raise ValueError(f"Unsupported file format: {filepath.suffix}")
    
    def save_to_file(self, df: pd.DataFrame, filepath: Path, format: str = "parquet"):
        """Save collected data to file."""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        if format == "parquet":
            df.to_parquet(filepath.with_suffix('.parquet'))
        elif format == "csv":
            df.to_csv(filepath.with_suffix('.csv'), index=False)
        elif format == "both":
            df.to_parquet(filepath.with_suffix('.parquet'))
            df.to_csv(filepath.with_suffix('.csv'), index=False)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logger.info(f"Saved {len(df)} records to {filepath}")


# Example usage and testing
if __name__ == "__main__":
    import yaml
    
    # Load config
    config_path = Path(__file__).parent.parent.parent / "config" / "config.yaml"
    if config_path.exists():
        with open(config_path) as f:
            config = yaml.safe_load(f)
    else:
        config = {}
    
    collector = SteamDataCollector(config)
    
    # Test current data collection
    print("Testing current data collection...")
    data_point = collector.collect_current_from_steam()
    if data_point:
        print(f"Current users: {data_point.concurrent_users:,}")
    else:
        print("Failed to collect current data")
