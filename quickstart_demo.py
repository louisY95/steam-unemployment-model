#!/usr/bin/env python3
"""
Quick Start Demo

Demonstrates the analysis pipeline using simulated data.
Use this to understand the model structure before collecting real data.

Run: python quickstart_demo.py
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from loguru import logger

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.processors import DataProcessor
from src.analysis import StatisticalAnalysis


def generate_simulated_data(n_months: int = 36) -> tuple:
    """
    Generate simulated Steam and unemployment data with a lagged relationship.
    
    The simulation creates:
    - Hourly Steam concurrent user data
    - Monthly unemployment rate data
    - A lagged correlation between Steam working hours activity and unemployment
    
    Returns:
        Tuple of (steam_hourly_df, unemployment_df)
    """
    logger.info(f"Generating {n_months} months of simulated data...")
    
    np.random.seed(42)
    
    # Generate monthly unemployment data first
    start_date = datetime(2022, 1, 1)
    unemployment_dates = pd.date_range(start=start_date, periods=n_months, freq='MS')
    
    # Base unemployment with trend and seasonality
    base = 4.5  # Base unemployment rate
    trend = np.linspace(0, 0.5, n_months)  # Slight upward trend
    seasonality = 0.3 * np.sin(np.arange(n_months) * 2 * np.pi / 12)  # Annual cycle
    noise = np.random.randn(n_months) * 0.2
    
    unemployment = base + trend + seasonality + noise
    unemployment = np.maximum(unemployment, 3.0)  # Floor at 3%
    
    unemployment_df = pd.DataFrame({
        'date': unemployment_dates,
        'UNRATE': unemployment
    }).set_index('date')
    
    # Generate hourly Steam data
    # Total hours in the period
    hours_in_period = n_months * 30 * 24  # Approximate
    steam_dates = pd.date_range(start=start_date, periods=hours_in_period, freq='H')
    
    # Base concurrent users: ~25-35 million
    base_users = 28_000_000
    
    # Daily pattern: peak in evening US time (around 8-10 PM EST = 1-3 AM UTC)
    hour_of_day = steam_dates.hour
    daily_pattern = 5_000_000 * np.sin((hour_of_day - 2) * np.pi / 12)  # Peak around 2 AM UTC
    
    # Weekly pattern: higher on weekends
    day_of_week = steam_dates.dayofweek
    weekend_boost = np.where(day_of_week >= 5, 3_000_000, 0)
    
    # Monthly unemployment relationship (lagged)
    # Higher unemployment -> more Steam users during work hours
    month_indices = (steam_dates.to_period('M') - start_date.strftime('%Y-%m')).astype(int)
    month_indices = np.clip(month_indices, 0, n_months - 1)
    
    # Relationship: 500K more users per 1% unemployment (with 2-month lag)
    lagged_unemployment = np.roll(unemployment, 2)
    lagged_unemployment[:2] = unemployment[:2]  # Handle edge
    
    unemployment_effect = lagged_unemployment[month_indices] * 500_000
    
    # Random noise
    noise = np.random.randn(len(steam_dates)) * 1_000_000
    
    # Combine all effects
    concurrent_users = (
        base_users + 
        daily_pattern + 
        weekend_boost + 
        unemployment_effect + 
        noise
    ).astype(int)
    
    # Ensure reasonable bounds
    concurrent_users = np.clip(concurrent_users, 15_000_000, 45_000_000)
    
    steam_df = pd.DataFrame({
        'timestamp': steam_dates,
        'concurrent_users': concurrent_users,
        'source': 'simulated'
    })
    
    logger.info(f"Generated {len(steam_df):,} hourly Steam records")
    logger.info(f"Generated {len(unemployment_df)} monthly unemployment records")
    
    return steam_df, unemployment_df


def run_demo():
    """Run the full analysis pipeline with simulated data."""
    
    print("\n" + "="*70)
    print("🎮 STEAM-UNEMPLOYMENT MODEL - QUICK START DEMO")
    print("="*70)
    print("\nThis demo uses simulated data to demonstrate the analysis pipeline.")
    print("For real analysis, collect actual data using: python main.py collect")
    
    # Generate simulated data
    steam_df, unemployment_df = generate_simulated_data(n_months=36)
    
    print(f"\n📊 Generated Data:")
    print(f"   Steam hourly records: {len(steam_df):,}")
    print(f"   Date range: {steam_df['timestamp'].min()} to {steam_df['timestamp'].max()}")
    print(f"   Unemployment records: {len(unemployment_df)}")
    
    # Initialize processor
    config = {
        "working_hours": {
            "east_coast": {"start_hour": 8, "end_hour": 17},
            "west_coast": {"start_hour": 8, "end_hour": 17},
            "include_weekends": False
        }
    }
    
    processor = DataProcessor(config)
    
    # Process data
    print("\n🔧 Processing data...")
    print("   Filtering to US working hours (8am-5pm EST/PST, weekdays)...")
    
    wh_data, total_data = processor.prepare_analysis_dataset(
        steam_df,
        unemployment_df,
        steam_timestamp_col='timestamp',
        steam_value_col='concurrent_users'
    )
    
    print(f"\n   Working hours dataset: {len(wh_data)} monthly observations")
    print(f"   Total dataset: {len(total_data)} monthly observations")
    
    # Run analysis
    print("\n📈 Running statistical analysis...")
    
    analyzer = StatisticalAnalysis(config)
    
    steam_col = 'concurrent_users_mean'
    unemployment_col = 'UNRATE'
    
    # 1. Stationarity tests
    print("\n1️⃣ STATIONARITY TESTS (ADF)")
    print("-" * 40)
    
    for name, data in [('Working Hours', wh_data), ('Total', total_data)]:
        steam_stat = analyzer.test_stationarity(data[steam_col].dropna())
        unemp_stat = analyzer.test_stationarity(data[unemployment_col].dropna())
        
        print(f"\n   {name} Data:")
        print(f"     Steam: {steam_stat['interpretation']} (p={steam_stat['p_value']:.4f})")
        print(f"     Unemployment: {unemp_stat['interpretation']} (p={unemp_stat['p_value']:.4f})")
    
    # 2. Cross-correlation
    print("\n2️⃣ CROSS-CORRELATION ANALYSIS")
    print("-" * 40)
    
    for name, data in [('Working Hours', wh_data), ('Total', total_data)]:
        cc_results = analyzer.cross_correlation(
            data[steam_col].dropna(),
            data[unemployment_col].dropna(),
            max_lag=12
        )
        
        # Find peak correlation
        best = max(cc_results, key=lambda r: abs(r.correlation))
        optimal = analyzer.find_optimal_lag(cc_results)
        
        print(f"\n   {name} Data:")
        print(f"     Peak correlation: r={best.correlation:+.4f} at lag {best.lag}")
        print(f"     Optimal significant lag: {optimal}")
        
        # Interpretation
        if best.lag < 0:
            print(f"     → Steam leads unemployment by {abs(best.lag)} months")
        elif best.lag > 0:
            print(f"     → Unemployment leads Steam by {best.lag} months")
        else:
            print(f"     → Concurrent relationship (no lead/lag)")
    
    # 3. Granger causality
    print("\n3️⃣ GRANGER CAUSALITY TESTS")
    print("-" * 40)
    print("   Testing: Does Steam activity Granger-cause unemployment?")
    
    for name, data in [('Working Hours', wh_data), ('Total', total_data)]:
        granger_results = analyzer.granger_causality_test(
            data, steam_col, unemployment_col, max_lag=6
        )
        
        print(f"\n   {name} Data:")
        significant_lags = [r for r in granger_results if r.is_significant]
        
        if significant_lags:
            print(f"     ✓ SIGNIFICANT at lags: {[r.lag for r in significant_lags]}")
            best = min(significant_lags, key=lambda r: r.p_value)
            print(f"     Best: lag {best.lag}, F={best.f_statistic:.3f}, p={best.p_value:.4f}")
        else:
            print(f"     ✗ Not significant at any lag tested")
            best = min(granger_results, key=lambda r: r.p_value) if granger_results else None
            if best:
                print(f"     Lowest p-value: lag {best.lag}, p={best.p_value:.4f}")
    
    # 4. Regression
    print("\n4️⃣ PREDICTIVE REGRESSION")
    print("-" * 40)
    
    feature_cols = [f'{steam_col}_lag{lag}' for lag in [1, 2, 3]
                   if f'{steam_col}_lag{lag}' in wh_data.columns]
    
    if feature_cols:
        for name, data in [('Working Hours', wh_data), ('Total', total_data)]:
            try:
                result = analyzer.fit_ols_regression(
                    data, unemployment_col, feature_cols
                )
                
                cv_result = analyzer.cross_validate_prediction(
                    data, unemployment_col, feature_cols
                )
                
                print(f"\n   {name} Data:")
                print(f"     In-sample R²: {result.r_squared:.4f}")
                print(f"     CV R² (mean): {cv_result['r2_mean']:.4f} ± {cv_result['r2_std']:.4f}")
                print(f"     CV RMSE: {cv_result['rmse_mean']:.4f}")
                
            except Exception as e:
                print(f"\n   {name} Data: Error - {e}")
    
    # 5. Comparison
    print("\n5️⃣ WORKING HOURS vs TOTAL DATA COMPARISON")
    print("-" * 40)
    
    comparison = analyzer.compare_datasets(
        wh_data, total_data, unemployment_col, steam_col
    )
    
    for r in comparison:
        symbol = "✓" if r.better_model == "Working Hours" else "○"
        print(f"\n   {symbol} {r.metric}")
        print(f"     Working Hours: {r.working_hours_value:.4f}")
        print(f"     Total Data: {r.total_value:.4f}")
        print(f"     Better: {r.better_model}")
    
    # Summary
    print("\n" + "="*70)
    print("📋 SUMMARY")
    print("="*70)
    
    wh_wins = sum(1 for r in comparison if r.better_model == "Working Hours")
    total_wins = sum(1 for r in comparison if r.better_model == "Total")
    
    print(f"\n   Working Hours better in: {wh_wins}/{len(comparison)} metrics")
    print(f"   Total Data better in: {total_wins}/{len(comparison)} metrics")
    
    if wh_wins > total_wins:
        print("\n   🎯 CONCLUSION: Working hours filtered Steam data shows BETTER")
        print("      predictive capability for unemployment in this simulation.")
    elif total_wins > wh_wins:
        print("\n   📊 CONCLUSION: Total Steam data shows BETTER predictive")
        print("      capability for unemployment in this simulation.")
    else:
        print("\n   ⚖️ CONCLUSION: Results are MIXED - no clear winner between")
        print("      working hours and total data approaches.")
    
    print("\n" + "="*70)
    print("🚀 NEXT STEPS")
    print("="*70)
    print("""
   1. Set up your FRED API key:
      - Get free key: https://fred.stlouisfed.org/docs/api/api_key.html
      - Add to config/config.yaml
      
   2. Collect real data:
      python main.py collect
      
   3. Process data:
      python main.py process
      
   4. Run analysis:
      python main.py analyze
      
   5. Generate report:
      python main.py report
    """)


if __name__ == "__main__":
    run_demo()
