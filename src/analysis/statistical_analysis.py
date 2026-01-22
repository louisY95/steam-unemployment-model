"""
Statistical Analysis Module

Provides statistical tests and models to analyze the relationship between
Steam user activity and US unemployment rates.

Tests include:
- Granger Causality
- Cross-Correlation
- Vector Autoregression (VAR)
- OLS and Regularized Regression
- Out-of-sample forecasting evaluation
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
from scipy import stats
from loguru import logger

try:
    import statsmodels.api as sm
    from statsmodels.tsa.stattools import grangercausalitytests, adfuller, ccf
    from statsmodels.tsa.api import VAR
    from statsmodels.regression.linear_model import OLS
    from statsmodels.stats.diagnostic import acorr_ljungbox
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    logger.warning("statsmodels not available. Some analyses will be limited.")

try:
    from sklearn.model_selection import TimeSeriesSplit, cross_val_score
    from sklearn.linear_model import Ridge, Lasso, ElasticNet
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("scikit-learn not available. Some analyses will be limited.")


@dataclass
class GrangerResult:
    """Results from Granger causality test."""
    lag: int
    f_statistic: float
    p_value: float
    is_significant: bool
    test_type: str = "ssr_ftest"


@dataclass
class CorrelationResult:
    """Results from cross-correlation analysis."""
    lag: int
    correlation: float
    p_value: float
    is_significant: bool


@dataclass
class RegressionResult:
    """Results from regression analysis."""
    model_name: str
    r_squared: float
    adj_r_squared: float
    rmse: float
    mae: float
    coefficients: Dict[str, float]
    p_values: Dict[str, float]
    aic: Optional[float] = None
    bic: Optional[float] = None


@dataclass
class ComparisonResult:
    """Comparison between working hours and total data models."""
    metric: str
    working_hours_value: float
    total_value: float
    difference: float
    better_model: str
    interpretation: str


class StatisticalAnalysis:
    """
    Comprehensive statistical analysis for Steam-Unemployment relationship.
    
    Provides methods for:
    1. Stationarity testing (ADF test)
    2. Granger causality testing
    3. Cross-correlation analysis
    4. VAR modeling
    5. Regression modeling (OLS, Ridge, Lasso)
    6. Out-of-sample forecasting
    7. Comparison of working hours vs total data predictive power
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize statistical analysis.
        
        Args:
            config: Configuration dictionary with analysis settings
        """
        self.config = config
        
        analysis_config = config.get("analysis", {})
        
        # Granger settings
        granger_config = analysis_config.get("granger", {})
        self.granger_max_lag = granger_config.get("max_lag", 12)
        self.significance_level = granger_config.get("significance_level", 0.05)
        
        # Cross-correlation settings
        cc_config = analysis_config.get("cross_correlation", {})
        self.cc_max_lag = cc_config.get("max_lag", 24)
        
        # Regression settings
        reg_config = analysis_config.get("regression", {})
        self.test_size = reg_config.get("test_size", 0.2)
        self.cv_folds = reg_config.get("cv_folds", 5)
        
        # VAR settings
        var_config = analysis_config.get("var", {})
        self.var_max_lag = var_config.get("max_lag", 12)
        self.ic_criterion = var_config.get("information_criterion", "aic")
    
    def test_stationarity(
        self,
        series: pd.Series,
        significance: float = 0.05
    ) -> Dict[str, Any]:
        """
        Test for stationarity using Augmented Dickey-Fuller test.
        
        Args:
            series: Time series to test
            significance: Significance level
            
        Returns:
            Dictionary with test results
        """
        if not STATSMODELS_AVAILABLE:
            raise RuntimeError("statsmodels required for stationarity test")
        
        # Remove NaN values
        series = series.dropna()
        
        result = adfuller(series, autolag='AIC')
        
        return {
            'test_statistic': result[0],
            'p_value': result[1],
            'lags_used': result[2],
            'n_observations': result[3],
            'critical_values': result[4],
            'is_stationary': result[1] < significance,
            'interpretation': 'Stationary' if result[1] < significance else 'Non-stationary'
        }
    
    def granger_causality_test(
        self,
        data: pd.DataFrame,
        cause_col: str,
        effect_col: str,
        max_lag: Optional[int] = None
    ) -> List[GrangerResult]:
        """
        Perform Granger causality test.
        
        Tests whether the 'cause' variable Granger-causes the 'effect' variable.
        
        Args:
            data: DataFrame with both variables
            cause_col: Column name of potential causal variable
            effect_col: Column name of effect variable
            max_lag: Maximum lag to test (default from config)
            
        Returns:
            List of GrangerResult for each lag
        """
        if not STATSMODELS_AVAILABLE:
            raise RuntimeError("statsmodels required for Granger causality test")
        
        max_lag = max_lag or self.granger_max_lag
        
        # Prepare data (effect column first, cause column second for grangercausalitytests)
        test_data = data[[effect_col, cause_col]].dropna()
        
        if len(test_data) < max_lag + 2:
            logger.warning(f"Insufficient data for Granger test. Need {max_lag + 2}, have {len(test_data)}")
            max_lag = max(1, len(test_data) - 2)
        
        logger.info(f"Running Granger causality test: {cause_col} -> {effect_col}, max_lag={max_lag}")
        
        results = []
        
        try:
            gc_results = grangercausalitytests(test_data, maxlag=max_lag, verbose=False)
            
            for lag in range(1, max_lag + 1):
                if lag in gc_results:
                    lag_result = gc_results[lag]
                    
                    # Get F-test results (most common test)
                    f_test = lag_result[0]['ssr_ftest']
                    f_stat = f_test[0]
                    p_value = f_test[1]
                    
                    results.append(GrangerResult(
                        lag=lag,
                        f_statistic=f_stat,
                        p_value=p_value,
                        is_significant=p_value < self.significance_level
                    ))
                    
        except Exception as e:
            logger.error(f"Granger test failed: {e}")
        
        return results
    
    def cross_correlation(
        self,
        x: pd.Series,
        y: pd.Series,
        max_lag: Optional[int] = None
    ) -> List[CorrelationResult]:
        """
        Compute cross-correlation at various lags.
        
        Args:
            x: First time series
            y: Second time series
            max_lag: Maximum lag to compute
            
        Returns:
            List of CorrelationResult for each lag
        """
        max_lag = max_lag or self.cc_max_lag
        
        # Align and clean data
        combined = pd.DataFrame({'x': x, 'y': y}).dropna()
        x_clean = combined['x'].values
        y_clean = combined['y'].values
        
        n = len(x_clean)
        
        if n < max_lag + 2:
            max_lag = max(1, n - 2)
        
        results = []
        
        for lag in range(-max_lag, max_lag + 1):
            if lag < 0:
                # x leads y
                corr, p_value = stats.pearsonr(x_clean[:lag], y_clean[-lag:])
            elif lag > 0:
                # y leads x
                corr, p_value = stats.pearsonr(x_clean[lag:], y_clean[:-lag])
            else:
                # No lag
                corr, p_value = stats.pearsonr(x_clean, y_clean)
            
            results.append(CorrelationResult(
                lag=lag,
                correlation=corr,
                p_value=p_value,
                is_significant=p_value < self.significance_level
            ))
        
        return results
    
    def find_optimal_lag(
        self,
        correlations: List[CorrelationResult],
        criterion: str = 'max_significant'
    ) -> Optional[int]:
        """
        Find the optimal lag based on cross-correlation results.
        
        Args:
            correlations: List of correlation results
            criterion: 'max_significant' or 'max_absolute'
            
        Returns:
            Optimal lag or None if no significant correlations
        """
        if criterion == 'max_significant':
            significant = [c for c in correlations if c.is_significant]
            if not significant:
                return None
            best = max(significant, key=lambda c: abs(c.correlation))
            return best.lag
        
        elif criterion == 'max_absolute':
            best = max(correlations, key=lambda c: abs(c.correlation))
            return best.lag
        
        return None
    
    def fit_var_model(
        self,
        data: pd.DataFrame,
        columns: List[str],
        max_lag: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Fit a Vector Autoregression (VAR) model.
        
        Args:
            data: DataFrame with time series variables
            columns: Columns to include in VAR
            max_lag: Maximum lag for lag order selection
            
        Returns:
            Dictionary with model results
        """
        if not STATSMODELS_AVAILABLE:
            raise RuntimeError("statsmodels required for VAR model")
        
        max_lag = max_lag or self.var_max_lag
        
        # Prepare data
        var_data = data[columns].dropna()
        
        if len(var_data) < max_lag + 2:
            raise ValueError(f"Insufficient data for VAR. Need {max_lag + 2}, have {len(var_data)}")
        
        logger.info(f"Fitting VAR model with columns: {columns}")
        
        # Fit VAR model
        model = VAR(var_data)
        
        # Select optimal lag order
        lag_selection = model.select_order(max_lag)
        optimal_lag = getattr(lag_selection, self.ic_criterion)
        
        # Fit with optimal lag
        fitted = model.fit(optimal_lag)
        
        return {
            'optimal_lag': optimal_lag,
            'lag_selection': {
                'aic': lag_selection.aic,
                'bic': lag_selection.bic,
                'hqic': lag_selection.hqic,
                'fpe': lag_selection.fpe
            },
            'summary': str(fitted.summary()),
            'aic': fitted.aic,
            'bic': fitted.bic,
            'fpe': fitted.fpe,
            'coefficients': {
                col: fitted.coefs[:, i, :].tolist()
                for i, col in enumerate(columns)
            }
        }
    
    def fit_ols_regression(
        self,
        data: pd.DataFrame,
        target_col: str,
        feature_cols: List[str],
        add_constant: bool = True
    ) -> RegressionResult:
        """
        Fit an OLS regression model.
        
        Args:
            data: DataFrame with features and target
            target_col: Name of target variable
            feature_cols: Names of feature variables
            add_constant: Whether to add intercept
            
        Returns:
            RegressionResult with model metrics
        """
        if not STATSMODELS_AVAILABLE:
            raise RuntimeError("statsmodels required for OLS regression")
        
        # Prepare data
        model_data = data[[target_col] + feature_cols].dropna()
        y = model_data[target_col]
        X = model_data[feature_cols]
        
        if add_constant:
            X = sm.add_constant(X)
        
        # Fit model
        model = OLS(y, X).fit()
        
        # Calculate predictions and errors
        predictions = model.predict(X)
        rmse = np.sqrt(mean_squared_error(y, predictions))
        mae = mean_absolute_error(y, predictions)
        
        # Extract coefficients and p-values
        coefficients = dict(zip(X.columns, model.params))
        p_values = dict(zip(X.columns, model.pvalues))
        
        return RegressionResult(
            model_name='OLS',
            r_squared=model.rsquared,
            adj_r_squared=model.rsquared_adj,
            rmse=rmse,
            mae=mae,
            coefficients=coefficients,
            p_values=p_values,
            aic=model.aic,
            bic=model.bic
        )
    
    def fit_regularized_regression(
        self,
        data: pd.DataFrame,
        target_col: str,
        feature_cols: List[str],
        model_type: str = 'ridge',
        alpha: float = 1.0
    ) -> RegressionResult:
        """
        Fit regularized regression (Ridge, Lasso, or ElasticNet).
        
        Args:
            data: DataFrame with features and target
            target_col: Name of target variable
            feature_cols: Names of feature variables
            model_type: 'ridge', 'lasso', or 'elasticnet'
            alpha: Regularization strength
            
        Returns:
            RegressionResult with model metrics
        """
        if not SKLEARN_AVAILABLE:
            raise RuntimeError("scikit-learn required for regularized regression")
        
        # Prepare data
        model_data = data[[target_col] + feature_cols].dropna()
        y = model_data[target_col].values
        X = model_data[feature_cols].values
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Select model
        if model_type == 'ridge':
            model = Ridge(alpha=alpha)
        elif model_type == 'lasso':
            model = Lasso(alpha=alpha)
        elif model_type == 'elasticnet':
            model = ElasticNet(alpha=alpha, l1_ratio=0.5)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Fit model
        model.fit(X_scaled, y)
        
        # Predictions
        predictions = model.predict(X_scaled)
        
        # Metrics
        r2 = r2_score(y, predictions)
        rmse = np.sqrt(mean_squared_error(y, predictions))
        mae = mean_absolute_error(y, predictions)
        
        # Coefficients (descaled)
        coefficients = dict(zip(feature_cols, model.coef_))
        
        return RegressionResult(
            model_name=model_type.title(),
            r_squared=r2,
            adj_r_squared=1 - (1 - r2) * (len(y) - 1) / (len(y) - len(feature_cols) - 1),
            rmse=rmse,
            mae=mae,
            coefficients=coefficients,
            p_values={}  # Not available for regularized models
        )
    
    def cross_validate_prediction(
        self,
        data: pd.DataFrame,
        target_col: str,
        feature_cols: List[str],
        model_type: str = 'ridge',
        n_splits: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Perform time series cross-validation.
        
        Args:
            data: DataFrame with features and target
            target_col: Name of target variable
            feature_cols: Names of feature variables
            model_type: Type of model to use
            n_splits: Number of CV splits
            
        Returns:
            Dictionary with CV results
        """
        if not SKLEARN_AVAILABLE:
            raise RuntimeError("scikit-learn required for cross-validation")
        
        n_splits = n_splits or self.cv_folds
        
        # Prepare data
        model_data = data[[target_col] + feature_cols].dropna()
        y = model_data[target_col].values
        X = model_data[feature_cols].values
        
        if len(y) < n_splits + 2:
            n_splits = max(2, len(y) - 1)
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Select model
        if model_type == 'ridge':
            model = Ridge(alpha=1.0)
        elif model_type == 'lasso':
            model = Lasso(alpha=1.0)
        elif model_type == 'ols':
            from sklearn.linear_model import LinearRegression
            model = LinearRegression()
        else:
            model = Ridge(alpha=1.0)
        
        # Time series split
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        # Cross-validation scores
        r2_scores = []
        rmse_scores = []
        mae_scores = []
        
        for train_idx, test_idx in tscv.split(X_scaled):
            X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            
            r2_scores.append(r2_score(y_test, predictions))
            rmse_scores.append(np.sqrt(mean_squared_error(y_test, predictions)))
            mae_scores.append(mean_absolute_error(y_test, predictions))
        
        return {
            'n_splits': n_splits,
            'r2_mean': np.mean(r2_scores),
            'r2_std': np.std(r2_scores),
            'rmse_mean': np.mean(rmse_scores),
            'rmse_std': np.std(rmse_scores),
            'mae_mean': np.mean(mae_scores),
            'mae_std': np.std(mae_scores),
            'r2_scores': r2_scores,
            'rmse_scores': rmse_scores
        }
    
    def compare_datasets(
        self,
        working_hours_data: pd.DataFrame,
        total_data: pd.DataFrame,
        target_col: str,
        steam_col: str,
        lags: List[int] = [1, 2, 3]
    ) -> List[ComparisonResult]:
        """
        Compare predictive power between working hours and total Steam data.
        
        Args:
            working_hours_data: Dataset with working hours filtered Steam data
            total_data: Dataset with total Steam data
            target_col: Unemployment variable name
            steam_col: Steam variable name
            lags: Lag periods to include as features
            
        Returns:
            List of comparison results
        """
        results = []
        
        # Build feature columns
        feature_cols = [f'{steam_col}_lag{lag}' for lag in lags]
        feature_cols = [c for c in feature_cols if c in working_hours_data.columns and c in total_data.columns]
        
        if not feature_cols:
            logger.warning("No lag features found for comparison")
            return results
        
        # Cross-validation comparison
        try:
            wh_cv = self.cross_validate_prediction(
                working_hours_data, target_col, feature_cols, 'ridge'
            )
            total_cv = self.cross_validate_prediction(
                total_data, target_col, feature_cols, 'ridge'
            )
            
            # R² comparison
            r2_diff = wh_cv['r2_mean'] - total_cv['r2_mean']
            results.append(ComparisonResult(
                metric='R² (Cross-Validated)',
                working_hours_value=wh_cv['r2_mean'],
                total_value=total_cv['r2_mean'],
                difference=r2_diff,
                better_model='Working Hours' if r2_diff > 0 else 'Total',
                interpretation=f"Working hours data explains {abs(r2_diff)*100:.1f}% "
                              f"{'more' if r2_diff > 0 else 'less'} variance"
            ))
            
            # RMSE comparison
            rmse_diff = total_cv['rmse_mean'] - wh_cv['rmse_mean']  # Lower is better
            results.append(ComparisonResult(
                metric='RMSE (Cross-Validated)',
                working_hours_value=wh_cv['rmse_mean'],
                total_value=total_cv['rmse_mean'],
                difference=rmse_diff,
                better_model='Working Hours' if rmse_diff > 0 else 'Total',
                interpretation=f"Working hours data has {abs(rmse_diff):.4f} "
                              f"{'lower' if rmse_diff > 0 else 'higher'} RMSE"
            ))
            
        except Exception as e:
            logger.error(f"CV comparison failed: {e}")
        
        # Granger causality comparison
        try:
            wh_granger = self.granger_causality_test(
                working_hours_data, steam_col, target_col
            )
            total_granger = self.granger_causality_test(
                total_data, steam_col, target_col
            )
            
            # Find best significant lag
            wh_sig = [g for g in wh_granger if g.is_significant]
            total_sig = [g for g in total_granger if g.is_significant]
            
            wh_min_p = min(wh_granger, key=lambda g: g.p_value).p_value if wh_granger else 1.0
            total_min_p = min(total_granger, key=lambda g: g.p_value).p_value if total_granger else 1.0
            
            results.append(ComparisonResult(
                metric='Granger Causality (min p-value)',
                working_hours_value=wh_min_p,
                total_value=total_min_p,
                difference=total_min_p - wh_min_p,  # Lower p is better
                better_model='Working Hours' if wh_min_p < total_min_p else 'Total',
                interpretation=f"{'Working Hours' if wh_min_p < total_min_p else 'Total'} "
                              f"shows stronger Granger causality"
            ))
            
            results.append(ComparisonResult(
                metric='Significant Lags (Granger)',
                working_hours_value=len(wh_sig),
                total_value=len(total_sig),
                difference=len(wh_sig) - len(total_sig),
                better_model='Working Hours' if len(wh_sig) > len(total_sig) else 'Total',
                interpretation=f"Working hours: {len(wh_sig)} significant lags, "
                              f"Total: {len(total_sig)} significant lags"
            ))
            
        except Exception as e:
            logger.error(f"Granger comparison failed: {e}")
        
        return results
    
    def run_full_analysis(
        self,
        working_hours_data: pd.DataFrame,
        total_data: pd.DataFrame,
        steam_col: str,
        unemployment_col: str = 'UNRATE'
    ) -> Dict[str, Any]:
        """
        Run complete analysis pipeline.
        
        Args:
            working_hours_data: Working hours filtered dataset
            total_data: Total dataset
            steam_col: Steam variable column name
            unemployment_col: Unemployment column name
            
        Returns:
            Dictionary with all analysis results
        """
        results = {
            'timestamp': datetime.now().isoformat(),
            'working_hours_analysis': {},
            'total_analysis': {},
            'comparison': {},
            'conclusions': []
        }
        
        # 1. Stationarity tests
        logger.info("Running stationarity tests...")
        for name, data in [('working_hours', working_hours_data), ('total', total_data)]:
            results[f'{name}_analysis']['stationarity'] = {
                'steam': self.test_stationarity(data[steam_col].dropna()),
                'unemployment': self.test_stationarity(data[unemployment_col].dropna())
            }
        
        # 2. Cross-correlation
        logger.info("Running cross-correlation analysis...")
        for name, data in [('working_hours', working_hours_data), ('total', total_data)]:
            cc_results = self.cross_correlation(
                data[steam_col].dropna(),
                data[unemployment_col].dropna()
            )
            results[f'{name}_analysis']['cross_correlation'] = [asdict(r) for r in cc_results]
            results[f'{name}_analysis']['optimal_lag'] = self.find_optimal_lag(cc_results)
        
        # 3. Granger causality
        logger.info("Running Granger causality tests...")
        for name, data in [('working_hours', working_hours_data), ('total', total_data)]:
            granger_results = self.granger_causality_test(data, steam_col, unemployment_col)
            results[f'{name}_analysis']['granger_causality'] = [asdict(r) for r in granger_results]
        
        # 4. Regression analysis
        logger.info("Running regression analysis...")
        lags = [1, 2, 3]
        feature_cols = [f'{steam_col}_lag{lag}' for lag in lags 
                       if f'{steam_col}_lag{lag}' in working_hours_data.columns]
        
        if feature_cols:
            for name, data in [('working_hours', working_hours_data), ('total', total_data)]:
                try:
                    ols_result = self.fit_ols_regression(data, unemployment_col, feature_cols)
                    results[f'{name}_analysis']['ols_regression'] = asdict(ols_result)
                    
                    ridge_result = self.fit_regularized_regression(
                        data, unemployment_col, feature_cols, 'ridge'
                    )
                    results[f'{name}_analysis']['ridge_regression'] = asdict(ridge_result)
                    
                    cv_result = self.cross_validate_prediction(
                        data, unemployment_col, feature_cols
                    )
                    results[f'{name}_analysis']['cross_validation'] = cv_result
                except Exception as e:
                    logger.error(f"Regression failed for {name}: {e}")
        
        # 5. Dataset comparison
        logger.info("Comparing working hours vs total data...")
        comparison_results = self.compare_datasets(
            working_hours_data, total_data, unemployment_col, steam_col
        )
        results['comparison'] = [asdict(r) for r in comparison_results]
        
        # 6. Generate conclusions
        logger.info("Generating conclusions...")
        results['conclusions'] = self._generate_conclusions(results)
        
        return results
    
    def _generate_conclusions(self, results: Dict[str, Any]) -> List[str]:
        """Generate human-readable conclusions from analysis results."""
        conclusions = []
        
        # Check Granger causality significance
        wh_granger = results.get('working_hours_analysis', {}).get('granger_causality', [])
        total_granger = results.get('total_analysis', {}).get('granger_causality', [])
        
        wh_sig = sum(1 for g in wh_granger if g.get('is_significant', False))
        total_sig = sum(1 for g in total_granger if g.get('is_significant', False))
        
        if wh_sig > 0 or total_sig > 0:
            conclusions.append(
                f"Granger causality detected: Working hours data has {wh_sig} significant lags, "
                f"total data has {total_sig} significant lags."
            )
        else:
            conclusions.append(
                "No significant Granger causality detected at conventional significance levels."
            )
        
        # Check R² values
        wh_cv = results.get('working_hours_analysis', {}).get('cross_validation', {})
        total_cv = results.get('total_analysis', {}).get('cross_validation', {})
        
        if wh_cv and total_cv:
            wh_r2 = wh_cv.get('r2_mean', 0)
            total_r2 = total_cv.get('r2_mean', 0)
            
            if wh_r2 > total_r2:
                conclusions.append(
                    f"Working hours Steam data shows better predictive power "
                    f"(R²: {wh_r2:.3f} vs {total_r2:.3f})."
                )
            else:
                conclusions.append(
                    f"Total Steam data shows better predictive power "
                    f"(R²: {total_r2:.3f} vs {wh_r2:.3f})."
                )
        
        # Overall assessment
        comparison = results.get('comparison', [])
        wh_better = sum(1 for c in comparison if c.get('better_model') == 'Working Hours')
        total_better = sum(1 for c in comparison if c.get('better_model') == 'Total')
        
        if wh_better > total_better:
            conclusions.append(
                "OVERALL: Working hours filtered data appears to have stronger "
                "predictive capability for unemployment."
            )
        elif total_better > wh_better:
            conclusions.append(
                "OVERALL: Total Steam data appears to have stronger "
                "predictive capability for unemployment."
            )
        else:
            conclusions.append(
                "OVERALL: Results are mixed - neither dataset shows consistently "
                "superior predictive power."
            )
        
        return conclusions
    
    def save_results(self, results: Dict[str, Any], filepath: Path):
        """Save analysis results to JSON file."""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert numpy types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(i) for i in obj]
            return obj
        
        results = convert_numpy(results)
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Saved analysis results to {filepath}")


# Example usage
if __name__ == "__main__":
    import yaml
    
    # Create sample data
    np.random.seed(42)
    n = 36  # 3 years of monthly data
    
    dates = pd.date_range(start='2021-01-01', periods=n, freq='MS')
    
    # Simulated data with some correlation
    unemployment = 5 + np.cumsum(np.random.randn(n) * 0.2)
    steam_users = 25_000_000 + unemployment * 500_000 + np.random.randn(n) * 1_000_000
    
    data = pd.DataFrame({
        'month': dates,
        'concurrent_users_mean': steam_users,
        'UNRATE': unemployment
    })
    
    # Add lagged features
    for lag in [1, 2, 3]:
        data[f'concurrent_users_mean_lag{lag}'] = data['concurrent_users_mean'].shift(lag)
    
    config = {}
    analyzer = StatisticalAnalysis(config)
    
    # Test stationarity
    print("Testing stationarity...")
    stat_result = analyzer.test_stationarity(data['UNRATE'])
    print(f"Unemployment stationarity: {stat_result['interpretation']}")
    
    # Test Granger causality
    print("\nTesting Granger causality...")
    granger_results = analyzer.granger_causality_test(
        data, 'concurrent_users_mean', 'UNRATE', max_lag=4
    )
    for r in granger_results:
        print(f"  Lag {r.lag}: F={r.f_statistic:.2f}, p={r.p_value:.4f}, significant={r.is_significant}")
