"""
Simple Steam data collector for GitHub Actions.
No complex dependencies - just requests + pandas.
"""
import os
import sys
import json
import requests
import pandas as pd
from datetime import datetime, timezone
from pathlib import Path

DATA_DIR = Path("data/raw")
DATA_DIR.mkdir(parents=True, exist_ok=True)

STEAM_FILE = DATA_DIR / "steam_hourly.parquet"

def get_steam_concurrent_users():
    """Get current Steam concurrent users via Steam Web API."""
    url = "https://api.steampowered.com/ISteamUserStats/GetNumberOfCurrentPlayers/v1/?appid=753"
    print(f"Fetching Steam API: {url}")

    r = requests.get(url, timeout=15)
    print(f"Steam API response status: {r.status_code}")
    print(f"Steam API response body: {r.text[:500]}")

    if r.status_code != 200:
        print(f"ERROR: Steam API returned {r.status_code}")
        sys.exit(1)

    data = r.json()
    player_count = data.get("response", {}).get("player_count", 0)
    print(f"App 753 player count: {player_count}")

    # Estimate total concurrent users (app 753 is a small fraction of total)
    estimated_total = player_count * 50000
    print(f"Estimated total concurrent users: {estimated_total:,}")

    return estimated_total

def collect_fred_data():
    """Collect FRED unemployment data."""
    api_key = os.environ.get("FRED_API_KEY", "")
    if not api_key or api_key == "NO_KEY":
        print("No FRED API key set, skipping FRED collection")
        return

    print(f"Fetching FRED data with API key: {api_key[:4]}...")

    fred_file = DATA_DIR / "unemployment.parquet"
    series_ids = ["UNRATE", "ICSA"]
    all_data = []

    for series_id in series_ids:
        url = f"https://api.stlouisfed.org/fred/series/observations?series_id={series_id}&api_key={api_key}&file_type=json&observation_start=2020-03-01"
        print(f"Fetching FRED series: {series_id}")

        r = requests.get(url, timeout=30)
        print(f"FRED API response status for {series_id}: {r.status_code}")

        if r.status_code != 200:
            print(f"ERROR: FRED API returned {r.status_code} for {series_id}")
            print(f"Response: {r.text[:500]}")
            continue

        data = r.json()
        observations = data.get("observations", [])
        print(f"Got {len(observations)} observations for {series_id}")

        for obs in observations:
            if obs["value"] != ".":
                all_data.append({
                    "date": obs["date"],
                    "series_id": series_id,
                    "value": float(obs["value"]),
                })

    if all_data:
        df = pd.DataFrame(all_data)
        df["date"] = pd.to_datetime(df["date"])
        df.to_parquet(fred_file, index=False)
        print(f"Saved {len(df)} FRED records to {fred_file}")
    else:
        print("No FRED data collected")

def main():
    print(f"=== Steam Data Collector ===")
    print(f"Time: {datetime.now(timezone.utc).isoformat()}")
    print(f"Working directory: {os.getcwd()}")
    print(f"Data directory: {DATA_DIR.absolute()}")
    print()

    # Get current Steam users
    concurrent_users = get_steam_concurrent_users()

    # Create new data point
    new_row = {
        "timestamp": datetime.now(timezone.utc),
        "concurrent_users": concurrent_users,
        "source": "steam_web_api",
    }
    print(f"\nNew data point: {new_row}")

    # Load existing data if it exists
    if STEAM_FILE.exists():
        existing = pd.read_parquet(STEAM_FILE)
        print(f"Loaded {len(existing)} existing records")
        df = pd.concat([existing, pd.DataFrame([new_row])], ignore_index=True)
    else:
        print("No existing data file, creating new one")
        df = pd.DataFrame([new_row])

    # Save
    df.to_parquet(STEAM_FILE, index=False)
    print(f"\nSaved {len(df)} total records to {STEAM_FILE}")

    # Verify file exists
    if STEAM_FILE.exists():
        print(f"File size: {STEAM_FILE.stat().st_size} bytes")
        print("SUCCESS: Data file created!")
    else:
        print("FAILURE: Data file was NOT created!")
        sys.exit(1)

    # Collect FRED data if requested
    if os.environ.get("COLLECT_FRED", "false").lower() == "true":
        print("\n=== FRED Data Collection ===")
        collect_fred_data()

    print("\n=== Done ===")

if __name__ == "__main__":
    main()
