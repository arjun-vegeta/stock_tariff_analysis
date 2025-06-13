"""
Utility functions for stock tariff analysis.
Includes technical indicators, statistical analysis, and data processing tools.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union
import streamlit as st
from scipy import stats
from scipy.stats import pearsonr, spearmanr
import io
import warnings
import logging
warnings.filterwarnings('ignore')

# Configure logging for debugging and monitoring
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataProcessor:
    """Data processing and analysis utilities."""
    
    @staticmethod
    def calculate_stock_metrics(stock_data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate additional stock metrics
        
        Args:
            stock_data: DataFrame with stock data
            
        Returns:
            DataFrame with additional metrics
        """
        if stock_data.empty:
            return stock_data
        
        result_data = []
        
        for symbol in stock_data['Symbol'].unique():
            symbol_data = stock_data[stock_data['Symbol'] == symbol].copy()
            symbol_data = symbol_data.sort_values('Date')
            
            # Calculate returns
            symbol_data['Daily_Return'] = symbol_data['Close'].pct_change()
            symbol_data['Log_Return'] = np.log(symbol_data['Close'] / symbol_data['Close'].shift(1))
            
            # Calculate moving averages
            symbol_data['SMA_20'] = symbol_data['Close'].rolling(window=20).mean()
            symbol_data['SMA_50'] = symbol_data['Close'].rolling(window=50).mean()
            symbol_data['EMA_12'] = symbol_data['Close'].ewm(span=12).mean()
            symbol_data['EMA_26'] = symbol_data['Close'].ewm(span=26).mean()
            
            # Calculate technical indicators
            symbol_data['RSI'] = DataProcessor._calculate_rsi(symbol_data['Close'])
            symbol_data['MACD'] = symbol_data['EMA_12'] - symbol_data['EMA_26']
            symbol_data['MACD_Signal'] = symbol_data['MACD'].ewm(span=9).mean()
            
            # Calculate volatility
            symbol_data['Volatility_20'] = symbol_data['Daily_Return'].rolling(window=20).std()
            symbol_data['Volatility_50'] = symbol_data['Daily_Return'].rolling(window=50).std()
            
            # Calculate Bollinger Bands
            symbol_data['BB_Upper'] = symbol_data['SMA_20'] + (symbol_data['Close'].rolling(window=20).std() * 2)
            symbol_data['BB_Lower'] = symbol_data['SMA_20'] - (symbol_data['Close'].rolling(window=20).std() * 2)
            
            # Calculate price momentum
            symbol_data['Momentum_5'] = symbol_data['Close'].pct_change(periods=5)
            symbol_data['Momentum_10'] = symbol_data['Close'].pct_change(periods=10)
            
            result_data.append(symbol_data)
        
        return pd.concat(result_data, ignore_index=True)
    
    @staticmethod
    def _calculate_rsi(prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate Relative Strength Index (RSI)"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    @staticmethod
    def calculate_correlation_matrix(stock_data: pd.DataFrame, 
                                   sentiment_data: pd.DataFrame) -> pd.DataFrame:
        """Calculate correlation between stocks and sentiment."""
        if stock_data.empty or sentiment_data.empty:
            return pd.DataFrame()
        
        # Get daily returns for each stock
        daily_returns = stock_data.pivot_table(
            values='Daily_Return', 
            index='Date', 
            columns='Symbol'
        )
        
        # Merge with sentiment data
        sentiment_data['Date'] = pd.to_datetime(sentiment_data['date'])
        merged_data = daily_returns.merge(
            sentiment_data[['Date', 'avg_sentiment']], 
            on='Date', 
            how='inner'
        )
        
        if merged_data.empty:
            return pd.DataFrame()
        
        return merged_data.corr()
    
    @staticmethod
    def detect_anomalies(data: pd.Series, method: str = 'zscore', threshold: float = 3.0) -> pd.DataFrame:
        """
        Detect anomalies in time series data
        
        Args:
            data: Time series data
            method: Anomaly detection method ('zscore', 'iqr')
            threshold: Threshold for anomaly detection
            
        Returns:
            DataFrame with anomaly information
        """
        anomalies = []
        
        if method == 'zscore':
            clean_data = data.dropna()
            if len(clean_data) < 3:  # Need at least 3 points for z-score
                return pd.DataFrame()
                
            try:
                z_scores = np.abs(stats.zscore(clean_data.values))
                anomaly_mask = z_scores > threshold
                
                for i, is_anomaly in enumerate(anomaly_mask):
                    if is_anomaly and i < len(clean_data):
                        anomalies.append({
                            'index': clean_data.index[i],
                            'value': clean_data.iloc[i],
                            'z_score': z_scores[i],
                            'method': 'zscore'
                        })
            except Exception as e:
                # If z-score calculation fails, return empty DataFrame
                return pd.DataFrame()
                
        elif method == 'iqr':
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            anomaly_mask = (data < lower_bound) | (data > upper_bound)
            anomaly_indices = data[anomaly_mask].index
            
            for idx in anomaly_indices:
                anomalies.append({
                    'index': idx,
                    'value': data.loc[idx],
                    'bounds': f"[{lower_bound:.3f}, {upper_bound:.3f}]",
                    'method': 'iqr'
                })
        
        return pd.DataFrame(anomalies)
    
    @staticmethod
    def calculate_sector_impact_scores(sector_data: Dict[str, pd.DataFrame], 
                                     sentiment_data: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate impact scores for each sector based on sentiment
        
        Args:
            sector_data: Dictionary mapping sector names to DataFrames
            sentiment_data: DataFrame with sentiment data
            
        Returns:
            Dictionary with sector impact scores
        """
        impact_scores = {}
        
        if sentiment_data.empty:
            return impact_scores
        
        avg_sentiment = sentiment_data['avg_sentiment'].mean()
        sentiment_volatility = sentiment_data['avg_sentiment'].std()
        
        for sector, data in sector_data.items():
            if data.empty:
                impact_scores[sector] = 0.0
                continue
            
            # Calculate sector metrics
            sector_returns = data.groupby('Date')['Daily_Return'].mean()
            sector_volatility = sector_returns.std()
            
            # Merge with sentiment data for correlation
            sentiment_data['Date'] = pd.to_datetime(sentiment_data['date'])
            merged = pd.merge(
                sector_returns.reset_index(),
                sentiment_data[['Date', 'avg_sentiment']],
                on='Date',
                how='inner'
            )
            
            if len(merged) > 10:  # Need sufficient data points
                correlation = merged['Daily_Return'].corr(merged['avg_sentiment'])
                
                # Calculate impact score combining correlation and volatility
                impact_score = abs(correlation) * sector_volatility * sentiment_volatility
                impact_scores[sector] = impact_score
            else:
                impact_scores[sector] = 0.0
        
        return impact_scores

class StatisticalAnalyzer:
    """Statistical analysis utilities"""
    
    @staticmethod
    def perform_granger_causality_test(x: pd.Series, y: pd.Series, 
                                     max_lags: int = 5) -> Dict[str, float]:
        """
        Perform Granger causality test
        Note: This is a simplified implementation
        
        Args:
            x: First time series
            y: Second time series
            max_lags: Maximum number of lags to test
            
        Returns:
            Dictionary with test results
        """
        try:
            from statsmodels.tsa.stattools import grangercausalitytests
            
            # Combine series
            data = pd.concat([y, x], axis=1).dropna()
            
            if len(data) < 2 * max_lags:
                return {'p_value': 1.0, 'causality': False}
            
            # Perform test
            result = grangercausalitytests(data, max_lags, verbose=False)
            
            # Get minimum p-value across all lags
            p_values = [result[lag][0]['ssr_ftest'][1] for lag in range(1, max_lags + 1)]
            min_p_value = min(p_values)
            
            return {
                'p_value': min_p_value,
                'causality': min_p_value < 0.05,
                'optimal_lag': p_values.index(min(p_values)) + 1
            }
            
        except Exception as e:
            st.warning(f"Granger causality test failed: {e}")
            return {'p_value': 1.0, 'causality': False}
    
    @staticmethod
    def calculate_var_at_risk(returns: pd.Series, confidence_level: float = 0.05) -> float:
        """
        Calculate Value at Risk (VaR)
        
        Args:
            returns: Return series
            confidence_level: Confidence level (e.g., 0.05 for 95% VaR)
            
        Returns:
            VaR value
        """
        if returns.empty:
            return 0.0
        
        return np.percentile(returns.dropna(), confidence_level * 100)
    
    @staticmethod
    def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """
        Calculate Sharpe ratio
        
        Args:
            returns: Return series
            risk_free_rate: Risk-free rate (annual)
            
        Returns:
            Sharpe ratio
        """
        if returns.empty or returns.std() == 0:
            return 0.0
        
        excess_returns = returns.mean() - (risk_free_rate / 252)  # Daily risk-free rate
        return excess_returns / returns.std()

class ReportGenerator:
    """Generate analysis reports"""
    
    @staticmethod
    def generate_stock_summary(stock_data: pd.DataFrame) -> Dict[str, any]:
        """Generate summary statistics for stocks"""
        if stock_data.empty:
            return {}
        
        summary = {
            'total_stocks': stock_data['Symbol'].nunique(),
            'date_range': {
                'start': stock_data['Date'].min(),
                'end': stock_data['Date'].max()
            },
            'avg_return': stock_data['Daily_Return'].mean(),
            'avg_volatility': stock_data['Daily_Return'].std(),
            'total_trading_days': len(stock_data['Date'].unique()),
            'best_performing_stock': None,
            'worst_performing_stock': None
        }
        
        # Calculate stock performance
        stock_performance = []
        for symbol in stock_data['Symbol'].unique():
            symbol_data = stock_data[stock_data['Symbol'] == symbol].sort_values('Date')
            if len(symbol_data) > 1:
                first_price = symbol_data['Close'].iloc[0]
                last_price = symbol_data['Close'].iloc[-1]
                total_return = (last_price - first_price) / first_price
                stock_performance.append({
                    'symbol': symbol,
                    'total_return': total_return
                })
        
        if stock_performance:
            best_stock = max(stock_performance, key=lambda x: x['total_return'])
            worst_stock = min(stock_performance, key=lambda x: x['total_return'])
            
            summary['best_performing_stock'] = best_stock
            summary['worst_performing_stock'] = worst_stock
        
        return summary
    
    @staticmethod
    def generate_sentiment_summary(news_data: pd.DataFrame, 
                                 daily_sentiment: pd.DataFrame) -> Dict[str, any]:
        """Generate sentiment analysis summary"""
        if news_data.empty:
            return {}
        
        summary = {
            'total_articles': len(news_data),
            'date_range': {
                'start': news_data['published_at'].min() if 'published_at' in news_data.columns else None,
                'end': news_data['published_at'].max() if 'published_at' in news_data.columns else None
            },
            'avg_sentiment': news_data['final_score'].mean() if 'final_score' in news_data.columns else 0,
            'sentiment_volatility': news_data['final_score'].std() if 'final_score' in news_data.columns else 0,
            'sentiment_distribution': {},
            'most_positive_article': None,
            'most_negative_article': None
        }
        
        # Sentiment distribution
        if 'final_sentiment' in news_data.columns:
            sentiment_counts = news_data['final_sentiment'].value_counts()
            summary['sentiment_distribution'] = sentiment_counts.to_dict()
        
        # Extreme articles
        if 'final_score' in news_data.columns:
            most_positive_idx = news_data['final_score'].idxmax()
            most_negative_idx = news_data['final_score'].idxmin()
            
            summary['most_positive_article'] = {
                'title': news_data.loc[most_positive_idx, 'title'] if 'title' in news_data.columns else '',
                'score': news_data.loc[most_positive_idx, 'final_score']
            }
            
            summary['most_negative_article'] = {
                'title': news_data.loc[most_negative_idx, 'title'] if 'title' in news_data.columns else '',
                'score': news_data.loc[most_negative_idx, 'final_score']
            }
        
        return summary
    
    @staticmethod
    def generate_correlation_report(correlation_matrix: pd.DataFrame) -> Dict[str, any]:
        """Generate correlation analysis report"""
        if correlation_matrix.empty:
            return {}
        
        # Find strongest correlations with sentiment
        if 'avg_sentiment' in correlation_matrix.columns:
            sentiment_correlations = correlation_matrix['avg_sentiment'].drop('avg_sentiment').sort_values(key=abs, ascending=False)
            
            return {
                'strongest_positive_correlation': {
                    'stock': sentiment_correlations.idxmax(),
                    'correlation': sentiment_correlations.max()
                },
                'strongest_negative_correlation': {
                    'stock': sentiment_correlations.idxmin(),
                    'correlation': sentiment_correlations.min()
                },
                'all_correlations': sentiment_correlations.to_dict()
            }
        
        return {}

class DataValidator:
    """Validate and clean data"""
    
    @staticmethod
    def validate_stock_data(stock_data: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Validate stock data quality
        
        Args:
            stock_data: DataFrame with stock data
            
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        
        if stock_data.empty:
            issues.append("Stock data is empty")
            return False, issues
        
        required_columns = ['Symbol', 'Date', 'Close', 'Volume']
        missing_columns = [col for col in required_columns if col not in stock_data.columns]
        if missing_columns:
            issues.append(f"Missing required columns: {missing_columns}")
        
        # Check for negative prices
        if 'Close' in stock_data.columns:
            negative_prices = stock_data[stock_data['Close'] <= 0]
            if not negative_prices.empty:
                issues.append(f"Found {len(negative_prices)} rows with negative/zero prices")
        
        # Check for missing dates
        if 'Date' in stock_data.columns:
            null_dates = stock_data['Date'].isnull().sum()
            if null_dates > 0:
                issues.append(f"Found {null_dates} rows with missing dates")
        
        # Check for duplicate entries
        if 'Symbol' in stock_data.columns and 'Date' in stock_data.columns:
            duplicates = stock_data.duplicated(subset=['Symbol', 'Date']).sum()
            if duplicates > 0:
                issues.append(f"Found {duplicates} duplicate Symbol-Date combinations")
        
        return len(issues) == 0, issues
    
    @staticmethod
    def validate_news_data(news_data: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Validate news data quality
        
        Args:
            news_data: DataFrame with news data
            
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        
        if news_data.empty:
            issues.append("News data is empty")
            return False, issues
        
        required_columns = ['title', 'published_at']
        missing_columns = [col for col in required_columns if col not in news_data.columns]
        if missing_columns:
            issues.append(f"Missing required columns: {missing_columns}")
        
        # Check for empty titles
        if 'title' in news_data.columns:
            empty_titles = news_data['title'].isnull().sum()
            if empty_titles > 0:
                issues.append(f"Found {empty_titles} articles with missing titles")
        
        # Check for invalid dates
        if 'published_at' in news_data.columns:
            try:
                pd.to_datetime(news_data['published_at'])
            except:
                issues.append("Invalid date format in published_at column")
        
        return len(issues) == 0, issues

def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safely divide two numbers, returning default if denominator is zero"""
    if denominator == 0 or pd.isna(denominator) or pd.isna(numerator):
        return default
    return numerator / denominator

def calculate_percentage_change(old_value: float, new_value: float) -> float:
    """Calculate percentage change between two values"""
    if old_value == 0 or pd.isna(old_value) or pd.isna(new_value):
        return 0.0
    return ((new_value - old_value) / old_value) * 100

def format_currency(value: float) -> str:
    """Format value as currency"""
    if pd.isna(value):
        return "N/A"
    return f"${value:,.2f}"

def format_percentage(value: float) -> str:
    """Format value as percentage"""
    if pd.isna(value):
        return "N/A"
    return f"{value:.2f}%"

def get_date_range_string(start_date: datetime, end_date: datetime) -> str:
    """Get formatted date range string"""
    return f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}"

def calculate_trading_days(start_date: datetime, end_date: datetime) -> int:
    """Calculate number of trading days between two dates (approximation)"""
    total_days = (end_date - start_date).days
    # Approximate: exclude weekends (rough estimate)
    trading_days = total_days * 5 / 7
    return int(trading_days)

class PerformanceTracker:
    """Track performance metrics for the application"""
    
    def __init__(self):
        self.start_time = None
        self.metrics = {}
    
    def start_timing(self, operation: str):
        """Start timing an operation"""
        self.start_time = datetime.now()
        self.metrics[operation] = {'start': self.start_time}
    
    def end_timing(self, operation: str):
        """End timing an operation"""
        if operation in self.metrics and self.start_time:
            end_time = datetime.now()
            duration = (end_time - self.start_time).total_seconds()
            self.metrics[operation]['end'] = end_time
            self.metrics[operation]['duration'] = duration
            return duration
        return 0
    
    def get_summary(self) -> Dict[str, float]:
        """Get performance summary"""
        summary = {}
        for operation, data in self.metrics.items():
            if 'duration' in data:
                summary[operation] = data['duration']
        return summary

# Global performance tracker instance
performance_tracker = PerformanceTracker()

def log_performance(operation_name: str):
    """Decorator to log performance of functions"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            performance_tracker.start_timing(operation_name)
            try:
                result = func(*args, **kwargs)
                duration = performance_tracker.end_timing(operation_name)
                if duration > 1:  # Only log if operation takes more than 1 second
                    st.sidebar.text(f"{operation_name}: {duration:.2f}s")
                return result
            except Exception as e:
                performance_tracker.end_timing(operation_name)
                raise e
        return wrapper
    return decorator

def calculate_technical_indicators(stock_data: pd.DataFrame, 
                                 periods: Dict[str, int] = None) -> pd.DataFrame:
    """
    Calculate comprehensive technical indicators for stock price analysis.
    
    This function computes various technical analysis indicators that are
    commonly used in financial analysis. These indicators help identify
    trends, momentum, and potential trading signals.
    
    Args:
        stock_data (pd.DataFrame): Stock data with columns:
            - Symbol: Stock symbol identifier
            - Date: Trading date
            - Close: Closing price
            - High: Daily high price
            - Low: Daily low price
            - Volume: Trading volume
        periods (Dict[str, int], optional): Custom periods for indicators.
            Default: {'sma_short': 20, 'sma_long': 50, 'rsi': 14, 'bb': 20}
            
    Returns:
        pd.DataFrame: Original data enhanced with technical indicators:
            - SMA_20, SMA_50: Simple moving averages
            - EMA_12, EMA_26: Exponential moving averages
            - RSI: Relative Strength Index
            - BB_Upper, BB_Lower, BB_Middle: Bollinger Bands
            - MACD, MACD_Signal, MACD_Histogram: MACD indicators
            - Volatility: Rolling volatility (standard deviation)
    """
    if stock_data.empty:
        logger.warning("Empty stock data provided to calculate_technical_indicators")
        return stock_data
    
    # Set default periods if not provided
    if periods is None:
        periods = {
            'sma_short': 20,    # Short-term moving average period
            'sma_long': 50,     # Long-term moving average period
            'rsi': 14,          # RSI calculation period
            'bb': 20,           # Bollinger Bands period
            'ema_fast': 12,     # Fast EMA for MACD
            'ema_slow': 26,     # Slow EMA for MACD
            'macd_signal': 9    # MACD signal line period
        }
    
    # Create a copy to avoid modifying original data
    data = stock_data.copy()
    
    # Sort data by symbol and date for proper calculation
    data = data.sort_values(['Symbol', 'Date'])
    
    # Group by symbol to calculate indicators for each stock separately
    for symbol in data['Symbol'].unique():
        symbol_mask = data['Symbol'] == symbol
        symbol_data = data[symbol_mask].copy()
        
        # Skip if insufficient data for calculations
        if len(symbol_data) < max(periods.values()):
            logger.warning(f"Insufficient data for {symbol} - skipping technical indicators")
            continue
        
        # 1. Simple Moving Averages (SMA)
        # SMA smooths price data to identify trends
        data.loc[symbol_mask, 'SMA_20'] = symbol_data['Close'].rolling(
            window=periods['sma_short'], min_periods=1
        ).mean()
        
        data.loc[symbol_mask, 'SMA_50'] = symbol_data['Close'].rolling(
            window=periods['sma_long'], min_periods=1
        ).mean()
        
        # 2. Exponential Moving Averages (EMA)
        # EMA gives more weight to recent prices
        data.loc[symbol_mask, 'EMA_12'] = symbol_data['Close'].ewm(
            span=periods['ema_fast'], adjust=False
        ).mean()
        
        data.loc[symbol_mask, 'EMA_26'] = symbol_data['Close'].ewm(
            span=periods['ema_slow'], adjust=False
        ).mean()
        
        # 3. Relative Strength Index (RSI)
        # RSI measures the speed and change of price movements (0-100 scale)
        rsi_values = _calculate_rsi(symbol_data['Close'], periods['rsi'])
        data.loc[symbol_mask, 'RSI'] = rsi_values
        
        # 4. Bollinger Bands
        # BB shows volatility and potential support/resistance levels
        bb_upper, bb_middle, bb_lower = _calculate_bollinger_bands(
            symbol_data['Close'], periods['bb']
        )
        data.loc[symbol_mask, 'BB_Upper'] = bb_upper
        data.loc[symbol_mask, 'BB_Middle'] = bb_middle
        data.loc[symbol_mask, 'BB_Lower'] = bb_lower
        
        # 5. MACD (Moving Average Convergence Divergence)
        # MACD shows the relationship between two moving averages
        macd, macd_signal, macd_histogram = _calculate_macd(
            symbol_data['Close'], 
            periods['ema_fast'], 
            periods['ema_slow'], 
            periods['macd_signal']
        )
        data.loc[symbol_mask, 'MACD'] = macd
        data.loc[symbol_mask, 'MACD_Signal'] = macd_signal
        data.loc[symbol_mask, 'MACD_Histogram'] = macd_histogram
        
        # 6. Volatility (Rolling Standard Deviation)
        # Measures price volatility over a rolling window
        data.loc[symbol_mask, 'Volatility'] = symbol_data['Close'].rolling(
            window=periods['sma_short'], min_periods=1
        ).std()
    
    logger.info(f"Technical indicators calculated for {data['Symbol'].nunique()} symbols")
    return data

def _calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """
    Calculate Relative Strength Index (RSI) for price momentum analysis.
    
    RSI is a momentum oscillator that measures the speed and change of price
    movements. It oscillates between 0 and 100, with values above 70 typically
    considered overbought and values below 30 considered oversold.
    
    Args:
        prices (pd.Series): Series of closing prices
        period (int): Number of periods for RSI calculation (default: 14)
        
    Returns:
        pd.Series: RSI values
    """
    # Calculate price changes
    delta = prices.diff()
    
    # Separate gains and losses
    gains = delta.where(delta > 0, 0)  # Positive changes only
    losses = -delta.where(delta < 0, 0)  # Negative changes (made positive)
    
    # Calculate average gains and losses using exponential moving average
    avg_gains = gains.ewm(alpha=1/period, adjust=False).mean()
    avg_losses = losses.ewm(alpha=1/period, adjust=False).mean()
    
    # Calculate Relative Strength (RS) and RSI
    rs = avg_gains / avg_losses
    rsi = 100 - (100 / (1 + rs))
    
    return rsi

def _calculate_bollinger_bands(prices: pd.Series, period: int = 20, 
                             std_dev: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calculate Bollinger Bands for volatility analysis and support/resistance levels.
    
    Bollinger Bands consist of a moving average (middle band) and two standard
    deviation bands (upper and lower) that expand and contract based on volatility.
    
    Args:
        prices (pd.Series): Series of closing prices
        period (int): Number of periods for moving average (default: 20)
        std_dev (float): Number of standard deviations for bands (default: 2.0)
        
    Returns:
        Tuple[pd.Series, pd.Series, pd.Series]: Upper band, middle band, lower band
    """
    # Calculate middle band (simple moving average)
    middle_band = prices.rolling(window=period, min_periods=1).mean()
    
    # Calculate rolling standard deviation
    rolling_std = prices.rolling(window=period, min_periods=1).std()
    
    # Calculate upper and lower bands
    upper_band = middle_band + (rolling_std * std_dev)
    lower_band = middle_band - (rolling_std * std_dev)
    
    return upper_band, middle_band, lower_band

def _calculate_macd(prices: pd.Series, fast_period: int = 12, 
                   slow_period: int = 26, signal_period: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Calculate MACD indicators."""
    # Calculate EMAs
    ema_fast = prices.ewm(span=fast_period, adjust=False).mean()
    ema_slow = prices.ewm(span=slow_period, adjust=False).mean()
    
    # Calculate MACD line
    macd_line = ema_fast - ema_slow
    
    # Calculate signal line
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    
    # Calculate histogram
    histogram = macd_line - signal_line
    
    return macd_line, signal_line, histogram

def calculate_daily_returns(stock_data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate daily returns for stock price analysis and risk assessment.
    
    Daily returns measure the percentage change in stock price from one day
    to the next, which is essential for volatility and correlation analysis.
    
    Args:
        stock_data (pd.DataFrame): Stock data with Symbol, Date, and Close columns
        
    Returns:
        pd.DataFrame: Original data with added Daily_Return column
    """
    if stock_data.empty or 'Close' not in stock_data.columns:
        logger.warning("Invalid stock data for daily returns calculation")
        return stock_data
    
    data = stock_data.copy()
    data = data.sort_values(['Symbol', 'Date'])
    
    # Calculate daily returns for each stock symbol
    data['Daily_Return'] = data.groupby('Symbol')['Close'].pct_change()
    
    # Fill first day returns with 0 (no previous day to compare)
    data['Daily_Return'] = data['Daily_Return'].fillna(0)
    
    logger.info(f"Daily returns calculated for {data['Symbol'].nunique()} symbols")
    return data

def calculate_stock_sentiment_correlation(stock_data: pd.DataFrame, 
                                        sentiment_data: pd.DataFrame) -> pd.DataFrame:
    """Calculate correlation between stock returns and sentiment scores."""
    if stock_data.empty or sentiment_data.empty:
        logger.warning("Empty data provided for correlation analysis")
        return pd.DataFrame()
    
    try:
        # Ensure both datasets have date columns in proper format
        stock_data['Date'] = pd.to_datetime(stock_data['Date'])
        sentiment_data['date'] = pd.to_datetime(sentiment_data['date'])
        
        # Merge stock and sentiment data on date
        merged_data = pd.merge(
            stock_data[['Date', 'Symbol', 'Daily_Return']],
            sentiment_data[['date', 'avg_sentiment']],
            left_on='Date',
            right_on='date',
            how='inner'
        )
        
        if merged_data.empty:
            logger.warning("No overlapping dates found for correlation analysis")
            return pd.DataFrame()
        
        # Pivot stock data to have symbols as columns
        stock_returns_pivot = merged_data.pivot_table(
            index='Date',
            columns='Symbol',
            values='Daily_Return',
            aggfunc='mean'
        )
        
        # Get sentiment data indexed by date
        sentiment_indexed = merged_data[['Date', 'avg_sentiment']].drop_duplicates()
        sentiment_indexed = sentiment_indexed.set_index('Date')['avg_sentiment']
        
        # Combine stock returns and sentiment data
        combined_data = pd.concat([stock_returns_pivot, sentiment_indexed], axis=1)
        combined_data.columns = list(stock_returns_pivot.columns) + ['Sentiment']
        
        # Calculate correlation matrix
        correlation_matrix = combined_data.corr()
        
        logger.info(f"Correlation matrix calculated for {len(stock_returns_pivot.columns)} stocks")
        return correlation_matrix
        
    except Exception as e:
        logger.error(f"Error in correlation calculation: {str(e)}")
        return pd.DataFrame()

def detect_anomalies(data: pd.DataFrame, column: str = 'Close', 
                    method: str = 'iqr', threshold: float = 1.5) -> pd.DataFrame:
    """
    Detect anomalies (outliers) in stock price or sentiment data.
    
    Anomaly detection helps identify unusual market events, data errors,
    or significant news events that cause abnormal price movements.
    
    Args:
        data (pd.DataFrame): Data to analyze for anomalies
        column (str): Column name to analyze (default: 'Close')
        method (str): Detection method - 'iqr', 'zscore', or 'isolation' (default: 'iqr')
        threshold (float): Threshold for anomaly detection (default: 1.5)
        
    Returns:
        pd.DataFrame: Original data with added 'is_anomaly' column
    """
    if data.empty or column not in data.columns:
        logger.warning(f"Invalid data or column '{column}' not found for anomaly detection")
        return data
    
    data_copy = data.copy()
    data_copy['is_anomaly'] = False
    
    try:
        if method == 'iqr':
            # Interquartile Range (IQR) method
            # Identifies outliers beyond Q1 - 1.5*IQR and Q3 + 1.5*IQR
            Q1 = data_copy[column].quantile(0.25)
            Q3 = data_copy[column].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            
            anomalies = (data_copy[column] < lower_bound) | (data_copy[column] > upper_bound)
            
        elif method == 'zscore':
            # Z-score method (standard score)
            # Identifies outliers beyond threshold standard deviations from mean
            z_scores = np.abs(stats.zscore(data_copy[column].dropna()))
            anomalies = pd.Series(False, index=data_copy.index)
            anomalies.iloc[data_copy[column].dropna().index] = z_scores > threshold
            
        elif method == 'isolation':
            # Isolation Forest method (for multivariate anomaly detection)
            from sklearn.ensemble import IsolationForest
            
            # Reshape data for sklearn
            values = data_copy[column].values.reshape(-1, 1)
            
            # Fit isolation forest
            iso_forest = IsolationForest(contamination=0.1, random_state=42)
            anomalies = iso_forest.fit_predict(values) == -1
            anomalies = pd.Series(anomalies, index=data_copy.index)
            
        else:
            logger.warning(f"Unknown anomaly detection method: {method}")
            return data_copy
        
        # Apply anomaly flags
        data_copy['is_anomaly'] = anomalies
        
        anomaly_count = anomalies.sum()
        logger.info(f"Detected {anomaly_count} anomalies using {method} method")
        
        return data_copy
        
    except Exception as e:
        logger.error(f"Error in anomaly detection: {str(e)}")
        return data_copy

def calculate_portfolio_metrics(stock_data: pd.DataFrame, 
                               weights: Dict[str, float] = None) -> Dict[str, float]:
    """
    Calculate portfolio-level risk and return metrics.
    
    This function computes comprehensive portfolio statistics including
    returns, volatility, Sharpe ratio, and other risk metrics.
    
    Args:
        stock_data (pd.DataFrame): Stock data with daily returns
        weights (Dict[str, float], optional): Portfolio weights by symbol.
            If None, uses equal weights.
            
    Returns:
        Dict[str, float]: Portfolio metrics including:
            - total_return: Cumulative portfolio return
            - annualized_return: Annualized return percentage
            - volatility: Portfolio volatility (standard deviation)
            - sharpe_ratio: Risk-adjusted return measure
            - max_drawdown: Maximum portfolio decline
            - var_95: Value at Risk (95% confidence)
            - cvar_95: Conditional Value at Risk (95% confidence)
    """
    if stock_data.empty or 'Daily_Return' not in stock_data.columns:
        logger.warning("Invalid stock data for portfolio metrics calculation")
        return {}
    
    try:
        # Pivot data to have symbols as columns
        returns_pivot = stock_data.pivot_table(
            index='Date',
            columns='Symbol',
            values='Daily_Return',
            aggfunc='mean'
        ).fillna(0)
        
        if returns_pivot.empty:
            return {}
        
        # Set equal weights if none provided
        if weights is None:
            symbols = returns_pivot.columns
            weights = {symbol: 1.0/len(symbols) for symbol in symbols}
        
        # Filter weights to only include available symbols
        available_symbols = returns_pivot.columns
        filtered_weights = {k: v for k, v in weights.items() if k in available_symbols}
        
        if not filtered_weights:
            logger.warning("No valid symbols found in weights")
            return {}
        
        # Normalize weights to sum to 1
        total_weight = sum(filtered_weights.values())
        if total_weight == 0:
            return {}
        
        normalized_weights = {k: v/total_weight for k, v in filtered_weights.items()}
        
        # Calculate portfolio returns
        portfolio_returns = pd.Series(0.0, index=returns_pivot.index)
        for symbol, weight in normalized_weights.items():
            if symbol in returns_pivot.columns:
                portfolio_returns += returns_pivot[symbol] * weight
        
        # Calculate portfolio metrics
        total_return = (1 + portfolio_returns).prod() - 1
        annualized_return = (1 + total_return) ** (252 / len(portfolio_returns)) - 1
        volatility = portfolio_returns.std() * np.sqrt(252)  # Annualized volatility
        
        # Sharpe ratio (assuming risk-free rate of 2%)
        risk_free_rate = 0.02
        sharpe_ratio = (annualized_return - risk_free_rate) / volatility if volatility > 0 else 0
        
        # Maximum drawdown calculation
        cumulative_returns = (1 + portfolio_returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Value at Risk (VaR) and Conditional VaR (CVaR) at 95% confidence
        var_95 = portfolio_returns.quantile(0.05)  # 5th percentile (95% VaR)
        cvar_95 = portfolio_returns[portfolio_returns <= var_95].mean()  # Expected loss beyond VaR
        
        metrics = {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'var_95': var_95,
            'cvar_95': cvar_95,
            'num_assets': len(normalized_weights),
            'data_points': len(portfolio_returns)
        }
        
        logger.info("Portfolio metrics calculated successfully")
        return metrics
        
    except Exception as e:
        logger.error(f"Error calculating portfolio metrics: {str(e)}")
        return {}

def create_analysis_summary(stock_data: pd.DataFrame, 
                          sentiment_data: pd.DataFrame,
                          correlation_data: pd.DataFrame) -> Dict[str, any]:
    """
    Create a comprehensive analysis summary combining all data sources.
    
    This function generates high-level insights and statistics from the complete
    analysis, providing key findings for executive summary and reporting.
    
    Args:
        stock_data (pd.DataFrame): Processed stock data with indicators
        sentiment_data (pd.DataFrame): Daily sentiment analysis results
        correlation_data (pd.DataFrame): Stock-sentiment correlation matrix
        
    Returns:
        Dict[str, any]: Comprehensive analysis summary with key findings
    """
    summary = {
        'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'data_coverage': {},
        'stock_analysis': {},
        'sentiment_analysis': {},
        'correlation_insights': {},
        'key_findings': []
    }
    
    try:
        # Data coverage metrics
        if not stock_data.empty:
            summary['data_coverage']['stock_symbols'] = stock_data['Symbol'].nunique()
            summary['data_coverage']['stock_date_range'] = {
                'start': stock_data['Date'].min().strftime('%Y-%m-%d'),
                'end': stock_data['Date'].max().strftime('%Y-%m-%d'),
                'days': (stock_data['Date'].max() - stock_data['Date'].min()).days
            }
            summary['data_coverage']['total_stock_observations'] = len(stock_data)
        
        if not sentiment_data.empty:
            summary['data_coverage']['sentiment_date_range'] = {
                'start': sentiment_data['date'].min().strftime('%Y-%m-%d'),
                'end': sentiment_data['date'].max().strftime('%Y-%m-%d'),
                'days': len(sentiment_data)
            }
            summary['data_coverage']['total_articles'] = sentiment_data['article_count'].sum()
        
        # Stock analysis summary
        if not stock_data.empty and 'Daily_Return' in stock_data.columns:
            daily_returns = stock_data['Daily_Return']
            
            summary['stock_analysis'] = {
                'avg_daily_return': daily_returns.mean(),
                'volatility': daily_returns.std(),
                'best_day': daily_returns.max(),
                'worst_day': daily_returns.min(),
                'positive_days_pct': (daily_returns > 0).mean() * 100,
                'extreme_moves': len(daily_returns[abs(daily_returns) > 0.05])  # Days with >5% moves
            }
            
            # Identify best and worst performing stocks
            stock_performance = stock_data.groupby('Symbol')['Daily_Return'].agg(['mean', 'std']).reset_index()
            if not stock_performance.empty:
                best_performer = stock_performance.loc[stock_performance['mean'].idxmax()]
                worst_performer = stock_performance.loc[stock_performance['mean'].idxmin()]
                
                summary['stock_analysis']['best_performer'] = {
                    'symbol': best_performer['Symbol'],
                    'avg_return': best_performer['mean']
                }
                summary['stock_analysis']['worst_performer'] = {
                    'symbol': worst_performer['Symbol'],
                    'avg_return': worst_performer['mean']
                }
        
        # Sentiment analysis summary
        if not sentiment_data.empty:
            summary['sentiment_analysis'] = {
                'avg_sentiment': sentiment_data['avg_sentiment'].mean(),
                'sentiment_volatility': sentiment_data['avg_sentiment'].std(),
                'most_positive_day': {
                    'date': sentiment_data.loc[sentiment_data['avg_sentiment'].idxmax(), 'date'].strftime('%Y-%m-%d'),
                    'score': sentiment_data['avg_sentiment'].max()
                },
                'most_negative_day': {
                    'date': sentiment_data.loc[sentiment_data['avg_sentiment'].idxmin(), 'date'].strftime('%Y-%m-%d'),
                    'score': sentiment_data['avg_sentiment'].min()
                },
                'avg_articles_per_day': sentiment_data['article_count'].mean()
            }
        
        # Correlation insights
        if not correlation_data.empty and 'Sentiment' in correlation_data.columns:
            sentiment_correlations = correlation_data['Sentiment'].drop('Sentiment')
            
            if not sentiment_correlations.empty:
                # Find stocks most correlated with sentiment
                positive_corr = sentiment_correlations[sentiment_correlations > 0]
                negative_corr = sentiment_correlations[sentiment_correlations < 0]
                
                summary['correlation_insights'] = {
                    'avg_sentiment_correlation': sentiment_correlations.mean(),
                    'strongest_positive_correlation': {
                        'symbol': positive_corr.idxmax() if not positive_corr.empty else None,
                        'correlation': positive_corr.max() if not positive_corr.empty else None
                    },
                    'strongest_negative_correlation': {
                        'symbol': negative_corr.idxmin() if not negative_corr.empty else None,
                        'correlation': negative_corr.min() if not negative_corr.empty else None
                    },
                    'high_correlation_count': len(sentiment_correlations[abs(sentiment_correlations) > 0.3])
                }
        
        # Generate key findings based on the analysis
        findings = []
        
        # Stock market findings
        if 'stock_analysis' in summary and summary['stock_analysis']:
            avg_return = summary['stock_analysis'].get('avg_daily_return', 0)
            if avg_return > 0.001:  # > 0.1% daily
                findings.append("Stock market showed positive momentum during the analysis period")
            elif avg_return < -0.001:  # < -0.1% daily
                findings.append("Stock market experienced downward pressure during the analysis period")
            
            volatility = summary['stock_analysis'].get('volatility', 0)
            if volatility > 0.02:  # > 2% daily volatility
                findings.append("High market volatility observed, indicating significant uncertainty")
            elif volatility < 0.01:  # < 1% daily volatility
                findings.append("Low market volatility suggests stable trading conditions")
        
        # Sentiment findings
        if 'sentiment_analysis' in summary and summary['sentiment_analysis']:
            avg_sentiment = summary['sentiment_analysis'].get('avg_sentiment', 0)
            if avg_sentiment > 0.1:
                findings.append("Overall news sentiment was positive regarding trade policies")
            elif avg_sentiment < -0.1:
                findings.append("Overall news sentiment was negative regarding trade policies")
            else:
                findings.append("News sentiment remained largely neutral on trade policies")
        
        # Correlation findings
        if 'correlation_insights' in summary and summary['correlation_insights']:
            avg_corr = summary['correlation_insights'].get('avg_sentiment_correlation', 0)
            if abs(avg_corr) > 0.2:
                findings.append(f"Moderate correlation found between news sentiment and stock prices")
            
            high_corr_count = summary['correlation_insights'].get('high_correlation_count', 0)
            total_stocks = summary['data_coverage'].get('stock_symbols', 1)
            if high_corr_count / total_stocks > 0.5:
                findings.append("Majority of stocks show significant sensitivity to news sentiment")
        
        summary['key_findings'] = findings
        
        logger.info("Analysis summary created successfully")
        return summary
        
    except Exception as e:
        logger.error(f"Error creating analysis summary: {str(e)}")
        return summary

def export_analysis_to_excel(data_dict: Dict[str, pd.DataFrame], 
                           filename: str = "stock_tariff_analysis.xlsx") -> io.BytesIO:
    """
    Export analysis results to Excel format with multiple sheets.
    
    This function creates a comprehensive Excel workbook containing all
    analysis results, formatted for easy sharing and further analysis.
    
    Args:
        data_dict (Dict[str, pd.DataFrame]): Dictionary mapping sheet names to DataFrames
        filename (str): Output filename (default: "stock_tariff_analysis.xlsx")
        
    Returns:
        io.BytesIO: Excel file as bytes buffer for download
    """
    try:
        # Create bytes buffer for Excel file
        buffer = io.BytesIO()
        
        # Create Excel writer object
        with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
            # Add each DataFrame as a separate sheet
            for sheet_name, df in data_dict.items():
                if not df.empty:
                    # Clean sheet name (Excel has restrictions)
                    clean_sheet_name = sheet_name[:31].replace('/', '_').replace('\\', '_')
                    
                    # Write DataFrame to sheet
                    df.to_excel(writer, sheet_name=clean_sheet_name, index=False)
                    
                    # Get workbook and worksheet objects for formatting
                    workbook = writer.book
                    worksheet = writer.sheets[clean_sheet_name]
                    
                    # Apply formatting
                    header_format = workbook.add_format({
                        'bold': True,
                        'text_wrap': True,
                        'valign': 'top',
                        'fg_color': '#D7E4BC',
                        'border': 1
                    })
                    
                    # Format header row
                    for col_num, value in enumerate(df.columns.values):
                        worksheet.write(0, col_num, value, header_format)
                        worksheet.set_column(col_num, col_num, 15)  # Set column width
            
            # Add summary sheet with metadata
            summary_data = {
                'Metric': [
                    'Export Date',
                    'Total Sheets',
                    'Analysis Period',
                    'Generated By'
                ],
                'Value': [
                    datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    len(data_dict),
                    'Variable by dataset',
                    'Stock Tariff Analysis Application'
                ]
            }
            
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
        
        buffer.seek(0)  # Reset buffer position
        logger.info(f"Excel export completed with {len(data_dict)} sheets")
        return buffer
        
    except Exception as e:
        logger.error(f"Error exporting to Excel: {str(e)}")
        # Return empty buffer on error
        buffer = io.BytesIO()
        return buffer

def export_analysis_to_csv(df: pd.DataFrame, filename: str = "analysis_data.csv") -> io.BytesIO:
    """
    Export a single DataFrame to CSV format.
    
    Args:
        df (pd.DataFrame): DataFrame to export
        filename (str): Output filename (default: "analysis_data.csv")
        
    Returns:
        io.BytesIO: CSV file as bytes buffer for download
    """
    try:
        buffer = io.BytesIO()
        df.to_csv(buffer, index=False, encoding='utf-8')
        buffer.seek(0)
        
        logger.info(f"CSV export completed for {len(df)} rows")
        return buffer
        
    except Exception as e:
        logger.error(f"Error exporting to CSV: {str(e)}")
        buffer = io.BytesIO()
        return buffer

def validate_data_quality(data: pd.DataFrame, 
                         required_columns: List[str],
                         data_type: str = "stock") -> Dict[str, any]:
    """
    Validate data quality and completeness for analysis.
    
    This function performs comprehensive data quality checks to ensure
    the reliability of analysis results and identify potential issues.
    
    Args:
        data (pd.DataFrame): Data to validate
        required_columns (List[str]): List of required column names
        data_type (str): Type of data being validated ("stock", "sentiment", etc.)
        
    Returns:
        Dict[str, any]: Data quality report with issues and recommendations
    """
    report = {
        'data_type': data_type,
        'validation_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'total_rows': len(data),
        'total_columns': len(data.columns),
        'issues': [],
        'warnings': [],
        'recommendations': [],
        'quality_score': 100  # Start with perfect score and deduct for issues
    }
    
    try:
        # Check if data is empty
        if data.empty:
            report['issues'].append("Dataset is empty")
            report['quality_score'] = 0
            return report
        
        # Check for required columns
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            report['issues'].append(f"Missing required columns: {missing_columns}")
            report['quality_score'] -= 30
        
        # Check for missing values
        missing_stats = data.isnull().sum()
        columns_with_missing = missing_stats[missing_stats > 0]
        
        if not columns_with_missing.empty:
            for col, missing_count in columns_with_missing.items():
                missing_pct = (missing_count / len(data)) * 100
                if missing_pct > 50:
                    report['issues'].append(f"Column '{col}' has {missing_pct:.1f}% missing values")
                    report['quality_score'] -= 15
                elif missing_pct > 10:
                    report['warnings'].append(f"Column '{col}' has {missing_pct:.1f}% missing values")
                    report['quality_score'] -= 5
        
        # Data type specific validations
        if data_type == "stock":
            # Validate stock-specific data quality
            if 'Close' in data.columns:
                # Check for negative prices
                negative_prices = (data['Close'] < 0).sum()
                if negative_prices > 0:
                    report['issues'].append(f"Found {negative_prices} negative stock prices")
                    report['quality_score'] -= 20
                
                # Check for extreme price changes (>50% in one day)
                if 'Daily_Return' in data.columns:
                    extreme_returns = (abs(data['Daily_Return']) > 0.5).sum()
                    if extreme_returns > 0:
                        report['warnings'].append(f"Found {extreme_returns} extreme daily returns (>50%)")
                        report['quality_score'] -= 5
            
            # Check date continuity
            if 'Date' in data.columns:
                date_gaps = _check_date_continuity(data, 'Date')
                if date_gaps > 0:
                    report['warnings'].append(f"Found {date_gaps} date gaps in the data")
                    report['quality_score'] -= 10
        
        elif data_type == "sentiment":
            # Validate sentiment-specific data quality
            if 'avg_sentiment' in data.columns:
                # Check for sentiment values outside expected range
                out_of_range = ((data['avg_sentiment'] < -1) | (data['avg_sentiment'] > 1)).sum()
                if out_of_range > 0:
                    report['warnings'].append(f"Found {out_of_range} sentiment values outside [-1, 1] range")
                    report['quality_score'] -= 10
        
        # Check for duplicate records
        if data.duplicated().sum() > 0:
            duplicate_count = data.duplicated().sum()
            report['warnings'].append(f"Found {duplicate_count} duplicate records")
            report['quality_score'] -= 5
        
        # Generate recommendations based on issues found
        if report['quality_score'] < 70:
            report['recommendations'].append("Data quality is below acceptable threshold - consider data cleaning")
        
        if missing_columns:
            report['recommendations'].append("Verify data source and collection process for missing columns")
        
        if len(data) < 30:
            report['recommendations'].append("Dataset is small - consider longer time period for more robust analysis")
        
        # Final quality assessment
        if report['quality_score'] >= 90:
            report['overall_quality'] = "Excellent"
        elif report['quality_score'] >= 70:
            report['overall_quality'] = "Good"
        elif report['quality_score'] >= 50:
            report['overall_quality'] = "Fair"
        else:
            report['overall_quality'] = "Poor"
        
        logger.info(f"Data quality validation completed - Score: {report['quality_score']}")
        return report
        
    except Exception as e:
        logger.error(f"Error in data validation: {str(e)}")
        report['issues'].append(f"Validation error: {str(e)}")
        report['quality_score'] = 0
        return report

def _check_date_continuity(data: pd.DataFrame, date_column: str) -> int:
    """
    Check for gaps in date sequence (helper function for data validation).
    
    Args:
        data (pd.DataFrame): Data with date column
        date_column (str): Name of the date column
        
    Returns:
        int: Number of date gaps found
    """
    try:
        # Convert to datetime if needed
        dates = pd.to_datetime(data[date_column])
        unique_dates = dates.drop_duplicates().sort_values()
        
        # Calculate expected number of business days
        if len(unique_dates) < 2:
            return 0
        
        date_range = pd.date_range(
            start=unique_dates.min(),
            end=unique_dates.max(),
            freq='B'  # Business days
        )
        
        # Count missing business days
        missing_dates = len(date_range) - len(unique_dates)
        
        # Only count significant gaps (more than 3 days)
        return max(0, missing_dates - 3)
        
    except Exception:
        return 0