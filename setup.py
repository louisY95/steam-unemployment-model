"""Setup script for Steam-Unemployment Predictive Model."""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text() if readme_path.exists() else ""

setup(
    name="steam-unemployment-model",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Statistical model testing Steam user activity as predictor of US unemployment",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/steam-unemployment-model",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
    python_requires=">=3.9",
    install_requires=[
        "pandas>=2.1.0",
        "numpy>=1.24.0",
        "scipy>=1.11.0",
        "statsmodels>=0.14.0",
        "scikit-learn>=1.3.0",
        "requests>=2.31.0",
        "beautifulsoup4>=4.12.0",
        "lxml>=5.0.0",
        "fredapi>=0.5.1",
        "pyarrow>=14.0.0",
        "pytz>=2023.3",
        "pyyaml>=6.0.1",
        "loguru>=0.7.2",
        "click>=8.1.7",
        "tqdm>=4.66.0",
    ],
    extras_require={
        "selenium": [
            "selenium>=4.15.0",
            "webdriver-manager>=4.0.1",
        ],
        "visualization": [
            "matplotlib>=3.8.0",
            "seaborn>=0.13.0",
            "plotly>=5.18.0",
        ],
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "jupyter>=1.0.0",
        ],
        "all": [
            "selenium>=4.15.0",
            "webdriver-manager>=4.0.1",
            "matplotlib>=3.8.0",
            "seaborn>=0.13.0",
            "plotly>=5.18.0",
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "jupyter>=1.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "steam-unemployment=main:cli",
        ],
    },
)
