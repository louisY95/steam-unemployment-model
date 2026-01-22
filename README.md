# Steam User Activity & US Unemployment Predictive Model

A comprehensive statistical model to test whether Steam platform user activity during US working hours (8am-5pm EST/PST) has predictive capability for US unemployment rates.

## Hypothesis

Active Steam user data during US East and West coast working hours has a strong predictive ability for unemployment rates in the USA. The theory is that increased gaming activity during traditional work hours may signal labor market changes before they appear in official statistics.

## Project Structure

```
steam_unemployment_model/
├── config/
│   └── config.yaml           # Configuration file (API keys, settings)
├── src/
│   ├── __init__.py
│   ├── collectors/
│   │   ├── __init__.py
│   │   ├── steam_collector.py    # Steam data collection
│   │   └── fred_collector.py     # FRED unemployment data collection
│   ├── processors/
│   │   ├── __init__.py
│   │   └── data_processor.py     # Data filtering and transformation
│   ├── analysis/
│   │   ├── __init__.py
│   │   └── statistical_analysis.py  # Statistical tests and models
│   └── utils/
│       ├── __init__.py
│       └── helpers.py            # Utility functions
├── data/
│   ├── raw/                  # Raw collected data
│   ├── processed/            # Processed data ready for analysis
│   └── results/              # Analysis results
├── notebooks/                # Jupyter notebooks for exploration
├── tests/                    # Unit tests
├── requirements.txt          # Python dependencies
├── setup.py                  # Package setup
├── main.py                   # Main entry point
└── README.md
```

## Data Sources

### Steam Concurrent Users
- **Source**: SteamDB (https://steamdb.info/app/753/charts/) for historical data
- **Backup**: Steam Stats page (https://store.steampowered.com/stats/) for real-time collection
- **Metric**: Total concurrent Steam users (platform-wide, not individual games)
- **Resolution**: Hourly data

### US Unemployment Rate
- **Source**: Federal Reserve Economic Data (FRED)
- **Series ID**: `UNRATE` (Seasonally Adjusted) and `ICSA` (Weekly Initial Claims)
- **Resolution**: Monthly (UNRATE) and Weekly (ICSA)

## Installation

### Prerequisites
- Python 3.9+
- Chrome browser (for Selenium)
- ChromeDriver (auto-installed via webdriver-manager)

### Setup

1. Clone or download this project

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Configure API keys:
```bash
cp config/config.yaml.example config/config.yaml
# Edit config/config.yaml with your API keys
```

### Required API Keys

1. **FRED API Key** (Required):
   - Register at: https://fred.stlouisfed.org/docs/api/api_key.html
   - Free registration required

2. **Steam Web API Key** (Optional, for additional data):
   - Register at: https://steamcommunity.com/dev/apikey

## Usage

### Collect Data
```bash
# Collect all data (Steam + FRED)
python main.py collect

# Collect only Steam data
python main.py collect --source steam

# Collect only FRED data
python main.py collect --source fred
```

### Run Analysis
```bash
# Run full analysis
python main.py analyze

# Run specific analysis
python main.py analyze --type granger      # Granger causality test
python main.py analyze --type correlation  # Cross-correlation analysis
python main.py analyze --type regression   # Regression models
python main.py analyze --type compare      # Compare working hours vs total data
```

### Generate Report
```bash
python main.py report
```

## Analysis Methods

### 1. Working Hours Filter
- **US East Coast**: 8:00 AM - 5:00 PM EST/EDT
- **US West Coast**: 8:00 AM - 5:00 PM PST/PDT
- Overlap period: 11:00 AM - 5:00 PM EST (8:00 AM - 2:00 PM PST)

### 2. Statistical Tests

#### Granger Causality Test
Tests whether lagged Steam activity improves prediction of unemployment beyond unemployment's own lags.

#### Cross-Correlation Analysis
Identifies lead/lag relationships between Steam activity and unemployment.

#### Vector Autoregression (VAR)
Models the dynamic relationship between Steam activity and unemployment.

#### Regression Models
- OLS regression with lagged predictors
- Ridge/Lasso regularization for variable selection
- ARIMAX models incorporating Steam data as exogenous variables

### 3. Comparison Metrics
- **Working Hours Data**: Steam activity during 8am-5pm EST and PST
- **Total Data**: All hourly Steam activity
- Evaluation: Which dataset provides better out-of-sample prediction?

## Output

### Data Files
- `data/processed/steam_hourly.parquet`: Processed Steam data
- `data/processed/steam_working_hours.parquet`: Working hours filtered data
- `data/processed/unemployment.parquet`: FRED unemployment data
- `data/processed/merged_data.parquet`: Combined dataset for analysis

### Results
- `data/results/granger_results.json`: Granger causality test results
- `data/results/correlation_results.json`: Cross-correlation results
- `data/results/regression_results.json`: Regression model results
- `data/results/comparison_report.json`: Working hours vs total data comparison
- `data/results/analysis_report.html`: Full HTML report with visualizations

## Key Metrics

The model evaluates:
1. **Statistical Significance**: p-values from Granger causality tests
2. **Predictive Power**: R², RMSE, MAE for regression models
3. **Forecast Accuracy**: Out-of-sample prediction performance
4. **Lead Time**: How far in advance Steam data predicts unemployment changes

## Limitations

1. **Data Availability**: Historical hourly Steam data requires scraping and may be incomplete
2. **Confounding Variables**: Other factors affect both gaming and unemployment
3. **Global Platform**: Steam is global; US-specific working hours may not capture US-only users
4. **Frequency Mismatch**: Hourly Steam data vs monthly/weekly unemployment data requires aggregation

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## License

MIT License - See LICENSE file for details

## References

- FRED API Documentation: https://fred.stlouisfed.org/docs/api/fred/
- SteamDB: https://steamdb.info/
- Steam Web API: https://developer.valvesoftware.com/wiki/Steam_Web_API
