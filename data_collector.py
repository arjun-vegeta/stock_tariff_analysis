"""
Data collection module for stock tariff analysis.
Handles stock price and news data collection from various APIs.
"""

import requests
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import time
from typing import Dict, List, Optional, Tuple
import json
import streamlit as st
from config import config

class StockDataCollector:
    """Collects stock price data using free APIs."""
    
    def __init__(self):
        """Initialize stock data collector."""
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    @st.cache_data(ttl=config.CACHE_TTL_SECONDS)
    def get_stock_data(_self, symbols: List[str], period: str = "1y") -> pd.DataFrame:
        """Fetch stock data using yfinance."""
        try:
            # Download data for all symbols
            data = yf.download(symbols, period=period, group_by='ticker', progress=False)
            
            # Handle single stock case
            if len(symbols) == 1:
                data.columns = pd.MultiIndex.from_product([[symbols[0]], data.columns])
            
            # Process data for each symbol
            stock_data = []
            
            for symbol in symbols:
                try:
                    symbol_data = data[symbol].copy()
                    symbol_data = symbol_data.dropna()
                    
                    if not symbol_data.empty:
                        symbol_data['Symbol'] = symbol
                        symbol_data['Date'] = symbol_data.index
                        symbol_data = symbol_data.reset_index(drop=True)
                        
                        # Calculate metrics
                        symbol_data['Daily_Return'] = symbol_data['Close'].pct_change()
                        symbol_data['Volatility'] = symbol_data['Daily_Return'].rolling(window=20).std()
                        symbol_data['SMA_20'] = symbol_data['Close'].rolling(window=20).mean()
                        symbol_data['SMA_50'] = symbol_data['Close'].rolling(window=50).mean()
                        
                        stock_data.append(symbol_data)
                
                except Exception as e:
                    st.warning(f"Could not fetch data for {symbol}: {str(e)}")
                    continue
            
            if stock_data:
                return pd.concat(stock_data, ignore_index=True)
            else:
                return pd.DataFrame()
                
        except Exception as e:
            st.error(f"Error fetching stock data: {str(e)}")
            return pd.DataFrame()
    
    def get_sector_data(self, sector_dict: Dict[str, List[str]]) -> Dict[str, pd.DataFrame]:
        """Fetch stock data for multiple sectors."""
        sector_data = {}
        
        progress_bar = st.progress(0)
        total_sectors = len(sector_dict)
        
        for i, (sector, symbols) in enumerate(sector_dict.items()):
            st.text(f"Fetching data for {sector} sector...")
            
            data = self.get_stock_data(symbols)
            if not data.empty:
                data['Sector'] = sector
                sector_data[sector] = data
            
            progress_bar.progress((i + 1) / total_sectors)
        
        st.text("Stock data collection complete!")
        return sector_data

class NewsDataCollector:
    """Collects tariff-related news articles from NewsAPI."""
    
    def __init__(self):
        """Initialize news data collector."""
        self.session = requests.Session()
        self.base_url = "https://newsapi.org/v2"
        self.session.headers.update({
            'User-Agent': 'Stock-Tariff-Analysis/1.0',
            'Accept': 'application/json'
        })
    
    @st.cache_data(ttl=config.CACHE_TTL_SECONDS)
    def get_tariff_news(_self, days_back: int = 30) -> pd.DataFrame:
        """Fetch news articles related to tariffs."""
        if config.NEWS_API_KEY == 'your_newsapi_key_here':
            st.warning("NewsAPI key not configured. Using sample data.")
            return _self._get_sample_news_data()
        
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)
            
            all_articles = []
            
            for keyword in config.TARIFF_KEYWORDS[:3]:
                url = f"{_self.base_url}/everything"
                
                params = {
                    'q': keyword,
                    'from': start_date.strftime('%Y-%m-%d'),
                    'to': end_date.strftime('%Y-%m-%d'),
                    'sortBy': 'publishedAt',
                    'language': 'en',
                    'apiKey': config.NEWS_API_KEY,
                    'pageSize': 20
                }
                
                response = _self.session.get(url, params=params)
                
                if response.status_code == 200:
                    data = response.json()
                    articles = data.get('articles', [])
                    
                    for article in articles:
                        all_articles.append({
                            'title': article.get('title', ''),
                            'description': article.get('description', ''),
                            'content': article.get('content', ''),
                            'url': article.get('url', ''),
                            'published_at': article.get('publishedAt', ''),
                            'source': article.get('source', {}).get('name', ''),
                            'keyword': keyword
                        })
                
                time.sleep(0.1)
            
            if all_articles:
                df = pd.DataFrame(all_articles)
                df['published_at'] = pd.to_datetime(df['published_at'])
                df = df.sort_values('published_at', ascending=False)
                df = df.drop_duplicates(subset=['title', 'url'])
                return df
            else:
                return _self._get_sample_news_data()
                
        except requests.exceptions.RequestException as e:
            st.warning(f"Error fetching news data: {str(e)}. Using sample data.")
            return _self._get_sample_news_data()
        except Exception as e:
            st.warning(f"Unexpected error: {str(e)}. Using sample data.")
            return _self._get_sample_news_data()
    
    def _get_sample_news_data(self) -> pd.DataFrame:
        """Generate sample news data for development."""
        sample_articles = [
            {
                'title': 'US Announces New Tariffs on Steel Imports',
                'description': 'The administration imposed additional tariffs on steel imports affecting manufacturing sector.',
                'content': 'Sample content about steel tariffs and their impact on the manufacturing industry...',
                'url': 'https://example.com/news1',
                'published_at': datetime.now() - timedelta(days=5),
                'source': 'Sample News',
                'keyword': 'tariff'
            },
            {
                'title': 'Trade War Escalation Impacts Tech Stocks',
                'description': 'Technology companies face uncertainty as trade tensions rise.',
                'content': 'Sample content about trade war impacts on technology sector...',
                'url': 'https://example.com/news2',
                'published_at': datetime.now() - timedelta(days=10),
                'source': 'Sample News',
                'keyword': 'trade war'
            },
            {
                'title': 'Import Restrictions Affect Automotive Industry',
                'description': 'New import policies create challenges for car manufacturers.',
                'content': 'Sample content about import restrictions and automotive industry...',
                'url': 'https://example.com/news3',
                'published_at': datetime.now() - timedelta(days=15),
                'source': 'Sample News',
                'keyword': 'import restrictions'
            },
            {
                'title': 'Trade Deal Negotiations Show Progress',
                'description': 'Positive developments in trade negotiations boost market sentiment.',
                'content': 'Sample content about positive trade deal developments...',
                'url': 'https://example.com/news4',
                'published_at': datetime.now() - timedelta(days=20),
                'source': 'Sample News',
                'keyword': 'trade deal'
            },
            {
                'title': 'Anti-Dumping Duties Impact Energy Sector',
                'description': 'New anti-dumping measures affect renewable energy imports.',
                'content': 'Sample content about anti-dumping duties in energy sector...',
                'url': 'https://example.com/news5',
                'published_at': datetime.now() - timedelta(days=25),
                'source': 'Sample News',
                'keyword': 'anti-dumping'
            }
        ]
        
        return pd.DataFrame(sample_articles)

class AlphaVantageCollector:
    """Alternative stock data collector using Alpha Vantage API."""
    
    def __init__(self):
        """Initialize Alpha Vantage collector."""
        self.base_url = "https://www.alphavantage.co/query"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Stock-Tariff-Analysis/1.0'
        })
    
    def get_stock_data(self, symbol: str) -> pd.DataFrame:
        """Fetch stock data from Alpha Vantage API."""
        if config.ALPHA_VANTAGE_KEY == 'your_alpha_vantage_key_here':
            return pd.DataFrame()
        
        try:
            params = {
                'function': 'TIME_SERIES_DAILY',
                'symbol': symbol,
                'apikey': config.ALPHA_VANTAGE_KEY,
                'outputsize': 'full'
            }
            
            response = self.session.get(self.base_url, params=params)
            data = response.json()
            
            if 'Time Series (Daily)' in data:
                time_series = data['Time Series (Daily)']
                
                df_data = []
                for date, values in time_series.items():
                    df_data.append({
                        'Date': pd.to_datetime(date),
                        'Open': float(values['1. open']),
                        'High': float(values['2. high']),
                        'Low': float(values['3. low']),
                        'Close': float(values['4. close']),
                        'Adj Close': float(values['4. close']),
                        'Volume': int(values['5. volume']),
                        'Symbol': symbol
                    })
                
                df = pd.DataFrame(df_data)
                df = df.sort_values('Date')
                return df
            
            else:
                if 'Error Message' in data:
                    st.warning(f"Alpha Vantage error for {symbol}: {data['Error Message']}")
                elif 'Note' in data:
                    st.warning(f"Alpha Vantage rate limit reached: {data['Note']}")
                return pd.DataFrame()
                
        except Exception as e:
            st.warning(f"Error fetching Alpha Vantage data for {symbol}: {str(e)}")
            return pd.DataFrame()

def clean_stock_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and validate stock data."""
    if df.empty:
        return df
    
    df = df.dropna(subset=['Close', 'Volume'])
    df = df[df['Close'] > 0]
    df = df[df['Volume'] >= 0]
    df = df.sort_values(['Symbol', 'Date'])
    
    return df

def merge_stock_news_data(stock_data: pd.DataFrame, news_data: pd.DataFrame) -> pd.DataFrame:
    """Merge stock and news data for correlation analysis."""
    if stock_data.empty or news_data.empty:
        return stock_data
    
    # Create copies to avoid modifying original DataFrames
    stock_df = stock_data.copy()
    news_df = news_data.copy()
    
    # Ensure Date columns are datetime
    stock_df['Date'] = pd.to_datetime(stock_df['Date'])
    news_df['Date'] = pd.to_datetime(news_df['published_at']).dt.date
    stock_df['Date_key'] = stock_df['Date'].dt.date
    
    # Aggregate news by date
    daily_news = news_df.groupby('Date').agg({
        'final_score': 'mean',
        'title': 'count'
    }).rename(columns={'title': 'news_count'}).reset_index()
    
    # Merge stock and news data
    merged = stock_df.merge(daily_news, left_on='Date_key', right_on='Date', how='left', suffixes=('', '_news'))
    
    # Drop duplicate date column and temporary key
    merged = merged.drop(columns=['Date_key', 'Date_news'], errors='ignore')
    
    # Fill missing sentiment scores with 0 (days with no news)
    merged['final_score'] = merged['final_score'].fillna(0)
    merged['news_count'] = merged['news_count'].fillna(0)
    
    return merged