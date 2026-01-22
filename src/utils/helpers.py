"""
Utility Functions

Helper functions for configuration loading, logging setup, and common operations.
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional

import yaml
from loguru import logger


def load_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to config file. If None, searches default locations.
        
    Returns:
        Configuration dictionary
    """
    # Default search paths
    search_paths = [
        config_path,
        Path("config/config.yaml"),
        Path("config.yaml"),
        Path(__file__).parent.parent.parent / "config" / "config.yaml",
    ]
    
    for path in search_paths:
        if path and path.exists():
            logger.info(f"Loading config from: {path}")
            with open(path) as f:
                return yaml.safe_load(f) or {}
    
    # Check for example config
    example_path = Path(__file__).parent.parent.parent / "config" / "config.yaml.example"
    if example_path.exists():
        logger.warning(
            f"Config file not found. Please copy {example_path} to config/config.yaml "
            "and fill in your API keys."
        )
    
    logger.warning("No config file found, using defaults")
    return {}


def setup_logging(config: Dict[str, Any]) -> None:
    """
    Configure logging based on config settings.
    
    Args:
        config: Configuration dictionary
    """
    log_config = config.get("logging", {})
    
    level = log_config.get("level", "INFO")
    log_to_file = log_config.get("log_to_file", True)
    log_file = log_config.get("log_file", "logs/model.log")
    
    # Remove default handler
    logger.remove()
    
    # Add console handler
    logger.add(
        sys.stderr,
        level=level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
               "<level>{level: <8}</level> | "
               "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
               "<level>{message}</level>"
    )
    
    # Add file handler if configured
    if log_to_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.add(
            log_path,
            level=level,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
            rotation="10 MB",
            retention="7 days"
        )
        
        logger.info(f"Logging to file: {log_path}")


def ensure_directories(config: Dict[str, Any]) -> Dict[str, Path]:
    """
    Ensure required directories exist.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Dictionary of directory paths
    """
    base_dir = Path(__file__).parent.parent.parent
    
    directories = {
        'data': base_dir / "data",
        'raw': base_dir / "data" / "raw",
        'processed': base_dir / "data" / "processed",
        'results': base_dir / "data" / "results",
        'logs': base_dir / "logs",
        'config': base_dir / "config",
    }
    
    for name, path in directories.items():
        path.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Directory ensured: {path}")
    
    return directories


def validate_config(config: Dict[str, Any]) -> bool:
    """
    Validate configuration has required values.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        True if valid, raises ValueError if not
    """
    # Check for FRED API key
    api_keys = config.get("api_keys", {})
    fred_key = api_keys.get("fred_api_key", "")
    
    if not fred_key or fred_key == "YOUR_FRED_API_KEY_HERE":
        # Check environment variable
        if not os.environ.get("FRED_API_KEY"):
            raise ValueError(
                "FRED API key required. Set in config.yaml or FRED_API_KEY environment variable.\n"
                "Get a free key at: https://fred.stlouisfed.org/docs/api/api_key.html"
            )
    
    return True


def format_number(n: float, decimal_places: int = 2) -> str:
    """Format large numbers with commas."""
    if abs(n) >= 1_000_000:
        return f"{n/1_000_000:,.{decimal_places}f}M"
    elif abs(n) >= 1_000:
        return f"{n/1_000:,.{decimal_places}f}K"
    else:
        return f"{n:,.{decimal_places}f}"


def safe_divide(a: float, b: float, default: float = 0.0) -> float:
    """Safely divide two numbers, returning default if division by zero."""
    if b == 0:
        return default
    return a / b
