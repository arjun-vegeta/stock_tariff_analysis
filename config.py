"""
Configuration settings for stock tariff analysis.
Loads API keys from environment variables and defines application constants.
"""

import os
from dataclasses import dataclass
from typing import List
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

@dataclass
class Config:
    """Configuration settings for stock tariff analysis."""
    
    # API Keys
    NEWS_API_KEY = os.getenv('NEWS_API_KEY')
    ALPHA_VANTAGE_KEY = os.getenv('ALPHA_VANTAGE_KEY')
    
    # Stock symbols by sector
    STOCK_SYMBOLS = {
        'Technology': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA', 'NFLX'],
        'Manufacturing': ['GE', 'CAT', 'BA', 'MMM', 'HON', 'LMT', 'NOC'],
        'Automotive': ['F', 'GM', 'TSLA', 'HMC', 'TM'],
        'Energy': ['XOM', 'CVX', 'COP', 'SLB', 'EOG', 'PXD', 'MPC', 'VLO'],
        'Materials': ['DOW', 'DD', 'APD', 'ECL', 'PPG', 'SHW', 'NEM', 'FCX'],
        'Consumer_Goods': ['PG', 'KO', 'PEP', 'WMT', 'HD', 'NKE', 'MCD', 'SBUX'],
        'Financial': ['JPM', 'BAC', 'WFC', 'C', 'GS', 'MS', 'AXP', 'V']
    }
    
    # Tariff-related keywords
    TARIFF_KEYWORDS = [
        'tariff', 'import tax', 'customs duty', 'dumping duties', 'countervailing duties',
        'trade war', 'trade policy', 'trade sanctions', 'trade barriers',
        'import restrictions', 'export controls', 'import ban', 'export ban',
        'protectionism', 'free trade', 'trade deal', 'trade agreement',
        'anti-dumping', 'safeguard measures', 'quota', 'embargo'
    ]
    
    # Time settings
    ANALYSIS_PERIOD_DAYS = 365  # 1 year
    NEWS_LOOKBACK_DAYS = 30     # 30 days
    
    # Streamlit UI settings
    PAGE_TITLE = "Stock Price & Tariff Impact Analysis"
    PAGE_ICON = "📈"
    
    # Cache settings
    CACHE_TTL_SECONDS = 3600  # 1 hour

# Create global configuration instance
# This allows other modules to import and use the configuration easily
config = Config()

def validate_config():
    """Check if required API keys are configured."""
    missing_keys = []
    
    if not config.NEWS_API_KEY:
        missing_keys.append('NEWS_API_KEY')
    
    if not config.ALPHA_VANTAGE_KEY:
        missing_keys.append('ALPHA_VANTAGE_KEY')
    
    if missing_keys:
        print(f"⚠️  Missing API keys: {', '.join(missing_keys)}")
        print("Please set these in your .env file")
        print("\nTo get API keys:")
        print("- NewsAPI: https://newsapi.org/register")
        print("- Alpha Vantage: https://www.alphavantage.co/support/#api-key")
        return False
    
    print("✅ All API keys are configured correctly!")
    return True

def get_all_stock_symbols():
    """Get list of all stock symbols across sectors."""
    all_symbols = []
    for sector_symbols in config.STOCK_SYMBOLS.values():
        all_symbols.extend(sector_symbols)
    
    # Remove duplicates (some stocks may appear in multiple sectors)
    return list(set(all_symbols))

def get_sector_by_symbol(symbol: str) -> str:
    """Find sector for a given stock symbol."""
    for sector, symbols in config.STOCK_SYMBOLS.items():
        if symbol in symbols:
            return sector
    return 'Unknown'

# Main execution for configuration testing
if __name__ == "__main__":
    print("Stock Tariff Analysis - Configuration Module")
    print("=" * 50)
    
    # Validate configuration
    validate_config()
    
    # Display configuration summary
    print(f"\nConfiguration Summary:")
    print(f"- Total sectors: {len(config.STOCK_SYMBOLS)}")
    print(f"- Total stocks: {len(get_all_stock_symbols())}")
    print(f"- Tariff keywords: {len(config.TARIFF_KEYWORDS)}")
    print(f"- Analysis period: {config.ANALYSIS_PERIOD_DAYS} days")
    print(f"- News lookback: {config.NEWS_LOOKBACK_DAYS} days")
    
    # Display sectors and their stock counts
    print(f"\nSectors and stock counts:")
    for sector, symbols in config.STOCK_SYMBOLS.items():
        print(f"- {sector}: {len(symbols)} stocks")