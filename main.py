#!/usr/bin/env python3
"""
Steam User Activity & US Unemployment Predictive Model

Main entry point for data collection, processing, and analysis.

Usage:
    python main.py collect              # Collect all data
    python main.py collect --source steam   # Collect Steam data only
    python main.py collect --source fred    # Collect FRED data only
    python main.py analyze              # Run full analysis
    python main.py analyze --type granger   # Run specific analysis
    python main.py report               # Generate report
    python main.py poll                 # Start continuous Steam data collection
"""

import sys
from pathlib import Path
from datetime import datetime
from typing import Optional

import click
from loguru import logger

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from src import RAW_DATA_DIR, PROCESSED_DATA_DIR, RESULTS_DIR
from src.utils import load_config, setup_logging, ensure_directories
from src.collectors import SteamDataCollector, FREDDataCollector
from src.processors import DataProcessor
from src.analysis import StatisticalAnalysis


@click.group()
@click.option('--config', '-c', type=click.Path(exists=True), help='Path to config file')
@click.pass_context
def cli(ctx, config):
    """Steam User Activity & US Unemployment Predictive Model."""
    ctx.ensure_object(dict)
    
    # Load configuration
    config_path = Path(config) if config else None
    ctx.obj['config'] = load_config(config_path)
    
    # Setup logging
    setup_logging(ctx.obj['config'])
    
    # Ensure directories exist
    ctx.obj['dirs'] = ensure_directories(ctx.obj['config'])
    
    logger.info("Steam-Unemployment Model initialized")


@cli.command()
@click.option('--source', '-s', type=click.Choice(['all', 'steam', 'fred']), default='all',
              help='Data source to collect')
@click.option('--method', '-m', type=click.Choice(['auto', 'steamdb_selenium', 'steamdb_api', 'steam_current']),
              default='auto', help='Steam collection method')
@click.pass_context
def collect(ctx, source, method):
    """Collect data from Steam and/or FRED."""
    config = ctx.obj['config']
    
    if source in ['all', 'steam']:
        logger.info("Collecting Steam data...")
        try:
            steam_collector = SteamDataCollector(config)
            steam_df = steam_collector.collect(method=method)
            
            if not steam_df.empty:
                output_path = RAW_DATA_DIR / "steam_hourly.parquet"
                steam_collector.save_to_file(steam_df, output_path)
                logger.info(f"Steam data saved: {len(steam_df)} records")
            else:
                logger.warning("No Steam data collected")
                
        except Exception as e:
            logger.error(f"Steam collection failed: {e}")
            if source == 'steam':
                raise
    
    if source in ['all', 'fred']:
        logger.info("Collecting FRED data...")
        try:
            fred_collector = FREDDataCollector(config)
            
            # Collect unemployment data
            unemployment_df = fred_collector.collect(['UNRATE', 'ICSA'])
            
            if not unemployment_df.empty:
                output_path = RAW_DATA_DIR / "unemployment.parquet"
                fred_collector.save_to_file(unemployment_df, output_path)
                logger.info(f"FRED data saved: {len(unemployment_df)} records")
            else:
                logger.warning("No FRED data collected")
                
        except Exception as e:
            logger.error(f"FRED collection failed: {e}")
            raise
    
    logger.info("Data collection complete")


@cli.command()
@click.option('--interval', '-i', type=int, default=3600,
              help='Polling interval in seconds (default: 3600 = 1 hour)')
@click.pass_context
def poll(ctx, interval):
    """Start continuous Steam data collection (polls Steam stats page)."""
    config = ctx.obj['config']
    
    logger.info(f"Starting continuous collection with {interval}s interval")
    logger.info("Press Ctrl+C to stop")
    
    try:
        steam_collector = SteamDataCollector(config)
        output_path = RAW_DATA_DIR / "steam_hourly_live.parquet"
        steam_collector.start_continuous_collection(output_path, interval)
        
    except KeyboardInterrupt:
        logger.info("Collection stopped by user")
    except Exception as e:
        logger.error(f"Collection failed: {e}")
        raise


@cli.command()
@click.pass_context
def process(ctx):
    """Process raw data and prepare for analysis."""
    config = ctx.obj['config']
    
    logger.info("Processing data...")
    
    # Load raw data
    steam_path = RAW_DATA_DIR / "steam_hourly.parquet"
    unemployment_path = RAW_DATA_DIR / "unemployment.parquet"
    
    if not steam_path.exists():
        logger.error(f"Steam data not found: {steam_path}")
        logger.info("Run 'python main.py collect' first")
        return
    
    if not unemployment_path.exists():
        logger.error(f"Unemployment data not found: {unemployment_path}")
        logger.info("Run 'python main.py collect' first")
        return
    
    import pandas as pd
    
    steam_df = pd.read_parquet(steam_path)
    unemployment_df = pd.read_parquet(unemployment_path)
    
    processor = DataProcessor(config)
    
    # Prepare analysis datasets
    wh_data, total_data = processor.prepare_analysis_dataset(
        steam_df, 
        unemployment_df,
        steam_timestamp_col='timestamp',
        steam_value_col='concurrent_users'
    )
    
    # Save processed data
    processor.save_processed_data(wh_data, PROCESSED_DATA_DIR / "working_hours_data.parquet")
    processor.save_processed_data(total_data, PROCESSED_DATA_DIR / "total_data.parquet")
    
    logger.info("Data processing complete")
    logger.info(f"Working hours dataset: {len(wh_data)} observations")
    logger.info(f"Total dataset: {len(total_data)} observations")


@cli.command()
@click.option('--type', '-t', 'analysis_type',
              type=click.Choice(['all', 'granger', 'correlation', 'regression', 'compare']),
              default='all', help='Type of analysis to run')
@click.pass_context
def analyze(ctx, analysis_type):
    """Run statistical analysis."""
    config = ctx.obj['config']
    
    logger.info(f"Running analysis: {analysis_type}")
    
    # Load processed data
    wh_path = PROCESSED_DATA_DIR / "working_hours_data.parquet"
    total_path = PROCESSED_DATA_DIR / "total_data.parquet"
    
    if not wh_path.exists() or not total_path.exists():
        logger.error("Processed data not found. Run 'python main.py process' first")
        return
    
    import pandas as pd
    
    wh_data = pd.read_parquet(wh_path)
    total_data = pd.read_parquet(total_path)
    
    analyzer = StatisticalAnalysis(config)
    
    # Determine column names
    steam_col = [c for c in wh_data.columns if 'concurrent_users' in c and 'mean' in c][0]
    unemployment_col = 'UNRATE'
    
    if analysis_type == 'all':
        # Run full analysis
        results = analyzer.run_full_analysis(
            wh_data, total_data,
            steam_col, unemployment_col
        )
        
        # Save results
        output_path = RESULTS_DIR / f"analysis_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        analyzer.save_results(results, output_path)
        
        # Print summary
        print("\n" + "="*60)
        print("ANALYSIS RESULTS SUMMARY")
        print("="*60)
        
        for conclusion in results.get('conclusions', []):
            print(f"\n• {conclusion}")
        
        print(f"\n\nFull results saved to: {output_path}")
        
    elif analysis_type == 'granger':
        print("\nGRANGER CAUSALITY TEST RESULTS")
        print("-"*40)
        
        for name, data in [('Working Hours', wh_data), ('Total', total_data)]:
            print(f"\n{name} Data:")
            results = analyzer.granger_causality_test(data, steam_col, unemployment_col)
            for r in results:
                sig = "***" if r.is_significant else ""
                print(f"  Lag {r.lag}: F={r.f_statistic:.3f}, p={r.p_value:.4f} {sig}")
    
    elif analysis_type == 'correlation':
        print("\nCROSS-CORRELATION ANALYSIS")
        print("-"*40)
        
        for name, data in [('Working Hours', wh_data), ('Total', total_data)]:
            print(f"\n{name} Data:")
            results = analyzer.cross_correlation(
                data[steam_col].dropna(),
                data[unemployment_col].dropna()
            )
            
            # Show top correlations
            sorted_results = sorted(results, key=lambda r: abs(r.correlation), reverse=True)[:5]
            for r in sorted_results:
                sig = "*" if r.is_significant else ""
                print(f"  Lag {r.lag:+3d}: r={r.correlation:+.4f}, p={r.p_value:.4f} {sig}")
            
            optimal = analyzer.find_optimal_lag(results)
            print(f"  Optimal lag: {optimal}")
    
    elif analysis_type == 'regression':
        print("\nREGRESSION ANALYSIS")
        print("-"*40)
        
        feature_cols = [f'{steam_col}_lag{lag}' for lag in [1, 2, 3]
                       if f'{steam_col}_lag{lag}' in wh_data.columns]
        
        if not feature_cols:
            print("No lag features available. Run 'python main.py process' first.")
            return
        
        for name, data in [('Working Hours', wh_data), ('Total', total_data)]:
            print(f"\n{name} Data:")
            try:
                result = analyzer.fit_ols_regression(data, unemployment_col, feature_cols)
                print(f"  R²: {result.r_squared:.4f}")
                print(f"  Adj R²: {result.adj_r_squared:.4f}")
                print(f"  RMSE: {result.rmse:.4f}")
                print(f"  AIC: {result.aic:.2f}")
            except Exception as e:
                print(f"  Error: {e}")
    
    elif analysis_type == 'compare':
        print("\nDATASET COMPARISON")
        print("-"*40)
        
        results = analyzer.compare_datasets(
            wh_data, total_data,
            unemployment_col, steam_col
        )
        
        for r in results:
            print(f"\n{r.metric}:")
            print(f"  Working Hours: {r.working_hours_value:.4f}")
            print(f"  Total: {r.total_value:.4f}")
            print(f"  Better: {r.better_model}")
            print(f"  {r.interpretation}")
    
    logger.info("Analysis complete")


@cli.command()
@click.pass_context
def report(ctx):
    """Generate analysis report."""
    config = ctx.obj['config']
    
    logger.info("Generating report...")
    
    # Find most recent results file
    results_files = list(RESULTS_DIR.glob("analysis_results_*.json"))
    
    if not results_files:
        logger.error("No analysis results found. Run 'python main.py analyze' first")
        return
    
    latest_results = max(results_files, key=lambda p: p.stat().st_mtime)
    
    import json
    
    with open(latest_results) as f:
        results = json.load(f)
    
    # Generate HTML report
    html_report = generate_html_report(results)
    
    report_path = RESULTS_DIR / f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
    with open(report_path, 'w') as f:
        f.write(html_report)
    
    logger.info(f"Report saved to: {report_path}")
    print(f"\nReport generated: {report_path}")


def generate_html_report(results: dict) -> str:
    """Generate HTML report from analysis results."""
    
    # Extract key findings
    conclusions = results.get('conclusions', [])
    comparison = results.get('comparison', [])
    
    wh_analysis = results.get('working_hours_analysis', {})
    total_analysis = results.get('total_analysis', {})
    
    html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Steam-Unemployment Analysis Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; }}
        h1 {{ color: #2c3e50; }}
        h2 {{ color: #34495e; border-bottom: 2px solid #3498db; padding-bottom: 10px; }}
        .summary {{ background: #ecf0f1; padding: 20px; border-radius: 5px; margin: 20px 0; }}
        .conclusion {{ background: #e8f6f3; padding: 15px; margin: 10px 0; border-left: 4px solid #1abc9c; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #bdc3c7; padding: 12px; text-align: left; }}
        th {{ background: #3498db; color: white; }}
        tr:nth-child(even) {{ background: #f2f2f2; }}
        .better {{ color: #27ae60; font-weight: bold; }}
        .worse {{ color: #e74c3c; }}
        .metric {{ font-family: monospace; }}
    </style>
</head>
<body>
    <h1>Steam User Activity &amp; US Unemployment Analysis Report</h1>
    
    <div class="summary">
        <h2>Executive Summary</h2>
        <p><strong>Analysis Date:</strong> {results.get('timestamp', 'Unknown')}</p>
        <p><strong>Hypothesis:</strong> Active Steam user data during US working hours (8am-5pm EST/PST) 
        has predictive capability for US unemployment rates.</p>
    </div>
    
    <h2>Key Findings</h2>
    {''.join(f'<div class="conclusion">• {c}</div>' for c in conclusions)}
    
    <h2>Working Hours vs Total Data Comparison</h2>
    <table>
        <tr>
            <th>Metric</th>
            <th>Working Hours</th>
            <th>Total Data</th>
            <th>Difference</th>
            <th>Better Model</th>
        </tr>
        {''.join(f'''
        <tr>
            <td>{c.get('metric', 'N/A')}</td>
            <td class="metric">{c.get('working_hours_value', 0):.4f}</td>
            <td class="metric">{c.get('total_value', 0):.4f}</td>
            <td class="metric">{c.get('difference', 0):.4f}</td>
            <td class="{'better' if c.get('better_model') == 'Working Hours' else ''}">{c.get('better_model', 'N/A')}</td>
        </tr>
        ''' for c in comparison)}
    </table>
    
    <h2>Granger Causality Results</h2>
    <h3>Working Hours Data</h3>
    <table>
        <tr><th>Lag</th><th>F-Statistic</th><th>P-Value</th><th>Significant?</th></tr>
        {''.join(f'''
        <tr>
            <td>{g.get('lag', 'N/A')}</td>
            <td class="metric">{g.get('f_statistic', 0):.3f}</td>
            <td class="metric">{g.get('p_value', 0):.4f}</td>
            <td class="{'better' if g.get('is_significant') else 'worse'}">{'Yes' if g.get('is_significant') else 'No'}</td>
        </tr>
        ''' for g in wh_analysis.get('granger_causality', []))}
    </table>
    
    <h3>Total Data</h3>
    <table>
        <tr><th>Lag</th><th>F-Statistic</th><th>P-Value</th><th>Significant?</th></tr>
        {''.join(f'''
        <tr>
            <td>{g.get('lag', 'N/A')}</td>
            <td class="metric">{g.get('f_statistic', 0):.3f}</td>
            <td class="metric">{g.get('p_value', 0):.4f}</td>
            <td class="{'better' if g.get('is_significant') else 'worse'}">{'Yes' if g.get('is_significant') else 'No'}</td>
        </tr>
        ''' for g in total_analysis.get('granger_causality', []))}
    </table>
    
    <h2>Cross-Validation Results</h2>
    <table>
        <tr>
            <th>Dataset</th>
            <th>R² (mean)</th>
            <th>R² (std)</th>
            <th>RMSE (mean)</th>
        </tr>
        <tr>
            <td>Working Hours</td>
            <td class="metric">{wh_analysis.get('cross_validation', {}).get('r2_mean', 0):.4f}</td>
            <td class="metric">{wh_analysis.get('cross_validation', {}).get('r2_std', 0):.4f}</td>
            <td class="metric">{wh_analysis.get('cross_validation', {}).get('rmse_mean', 0):.4f}</td>
        </tr>
        <tr>
            <td>Total</td>
            <td class="metric">{total_analysis.get('cross_validation', {}).get('r2_mean', 0):.4f}</td>
            <td class="metric">{total_analysis.get('cross_validation', {}).get('r2_std', 0):.4f}</td>
            <td class="metric">{total_analysis.get('cross_validation', {}).get('rmse_mean', 0):.4f}</td>
        </tr>
    </table>
    
    <h2>Methodology</h2>
    <ul>
        <li><strong>Steam Data:</strong> Total concurrent Steam platform users</li>
        <li><strong>Working Hours Filter:</strong> 8am-5pm EST and PST, weekdays only</li>
        <li><strong>Unemployment Data:</strong> FRED UNRATE (monthly, seasonally adjusted)</li>
        <li><strong>Analysis Period:</strong> Post-COVID (March 2020 onwards)</li>
        <li><strong>Statistical Tests:</strong> Granger causality, cross-correlation, OLS/Ridge regression</li>
    </ul>
    
    <h2>Limitations</h2>
    <ul>
        <li>Steam is a global platform; working hours filter may not capture US-only users</li>
        <li>Historical hourly data availability depends on scraping success</li>
        <li>Monthly unemployment data limits granularity of predictions</li>
        <li>Correlation does not imply causation</li>
        <li>Confounding variables (game releases, holidays) not controlled</li>
    </ul>
    
    <footer>
        <p><em>Report generated by Steam-Unemployment Predictive Model</em></p>
    </footer>
</body>
</html>
    """
    
    return html


@cli.command()
@click.pass_context
def status(ctx):
    """Show status of data files and configuration."""
    config = ctx.obj['config']
    
    print("\n" + "="*60)
    print("STEAM-UNEMPLOYMENT MODEL STATUS")
    print("="*60)
    
    # Check configuration
    print("\nCONFIGURATION")
    api_keys = config.get("api_keys", {})
    fred_key = api_keys.get("fred_api_key", "")

    if fred_key and fred_key != "YOUR_FRED_API_KEY_HERE":
        print("  [OK] FRED API key configured")
    else:
        print("  [!] FRED API key NOT configured")
        print("      Set in config/config.yaml or FRED_API_KEY environment variable")

    # Check data files
    print("\nDATA FILES")
    
    files = {
        'Steam Raw Data': RAW_DATA_DIR / "steam_hourly.parquet",
        'Steam Live Data': RAW_DATA_DIR / "steam_hourly_live.parquet",
        'Unemployment Data': RAW_DATA_DIR / "unemployment.parquet",
        'Working Hours Processed': PROCESSED_DATA_DIR / "working_hours_data.parquet",
        'Total Processed': PROCESSED_DATA_DIR / "total_data.parquet",
    }
    
    for name, path in files.items():
        if path.exists():
            import pandas as pd
            try:
                df = pd.read_parquet(path)
                print(f"  [OK] {name}: {len(df):,} records")
            except Exception:
                print(f"  [OK] {name}: exists (couldn't read)")
        else:
            print(f"  [!] {name}: not found")

    # Check results
    print("\nANALYSIS RESULTS")
    results_files = list(RESULTS_DIR.glob("analysis_results_*.json"))
    
    if results_files:
        latest = max(results_files, key=lambda p: p.stat().st_mtime)
        print(f"  [OK] {len(results_files)} result file(s)")
        print(f"       Latest: {latest.name}")
    else:
        print("  [!] No analysis results yet")
        print("      Run 'python main.py analyze' after collecting data")

    # Check reports
    report_files = list(RESULTS_DIR.glob("report_*.html"))
    if report_files:
        print(f"  [OK] {len(report_files)} report(s) generated")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    cli()
