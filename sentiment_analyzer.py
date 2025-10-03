"""
Sentiment analysis for stock tariff news articles.
Combines TextBlob, VADER, and custom financial sentiment analysis.
"""

import pandas as pd
import numpy as np
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import re
from typing import Dict, List, Tuple, Optional
import streamlit as st
from datetime import datetime, timedelta

class SentimentAnalyzer:
    """
    Sentiment analysis engine for financial news.
    Combines multiple analysis methods with trade-specific handling.
    """
    
    def __init__(self):
        """Initialize sentiment analyzer with VADER and custom word lists."""
        # Initialize VADER sentiment analyzer
        self.vader_analyzer = SentimentIntensityAnalyzer()
        
        # Custom positive word lists for financial/trade sentiment
        self.positive_words = [
            # Market performance terms
            'growth', 'profit', 'gain', 'rise', 'increase', 'boost', 'surge', 
            'rally', 'optimistic', 'bullish', 'recovery', 'expansion',
            
            # Trade and policy positive terms
            'deal', 'agreement', 'resolution', 'positive', 'strong', 'robust', 
            'stable', 'improvement', 'success', 'breakthrough', 'progress',
            
            # Economic positive indicators
            'prosperity', 'opportunity', 'benefit', 'advantage', 'competitive'
        ]
        
        # Custom negative word lists for financial/trade sentiment
        self.negative_words = [
            # Market decline terms
            'loss', 'decline', 'fall', 'drop', 'crash', 'plunge', 'recession',
            'crisis', 'downturn', 'collapse', 'tumble', 'sink',
            
            # Conflict and tension terms
            'conflict', 'war', 'tension', 'threat', 'risk', 'concern',
            'uncertainty', 'volatile', 'bearish', 'weak', 'struggle', 'problem',
            
            # Economic negative indicators
            'deficit', 'debt', 'bankruptcy', 'failure', 'dispute', 'retaliation'
        ]
        
        # Tariff and trade-specific terminology
        self.tariff_specific_words = {
            'negative': [
                'tariff', 'import tax', 'trade war', 'restriction', 'barrier', 
                'sanction', 'dumping', 'protectionism', 'retaliation', 'quota',
                'embargo', 'ban', 'penalty', 'dispute', 'escalation'
            ],
            'positive': [
                'trade deal', 'agreement', 'negotiation', 'resolution', 
                'cooperation', 'partnership', 'free trade', 'settlement',
                'accord', 'treaty', 'compromise', 'breakthrough'
            ]
        }
    
    def analyze_text_sentiment(self, text: str) -> Dict[str, float]:
        """
        Perform comprehensive sentiment analysis on a single text using multiple methods.
        
        This method combines three different sentiment analysis approaches:
        1. TextBlob: General sentiment with polarity and subjectivity
        2. VADER: Social media-tuned sentiment analysis
        3. Custom: Financial and trade-specific sentiment analysis
        
        Args:
            text (str): Text to analyze (article title, description, or content)
            
        Returns:
            Dict[str, float]: Comprehensive sentiment scores including:
                - textblob_polarity: TextBlob polarity score (-1 to 1)
                - textblob_subjectivity: TextBlob subjectivity score (0 to 1)
                - vader_compound: VADER compound score (-1 to 1)
                - vader_positive: VADER positive score (0 to 1)
                - vader_negative: VADER negative score (0 to 1)
                - vader_neutral: VADER neutral score (0 to 1)
                - custom_score: Custom financial sentiment score (-1 to 1)
                - final_sentiment: Overall sentiment classification ('positive', 'negative', 'neutral')
                - final_score: Combined final sentiment score (-1 to 1)
        """
        # Handle empty or invalid text input
        if not text or pd.isna(text):
            return {
                'textblob_polarity': 0.0,
                'textblob_subjectivity': 0.0,
                'vader_compound': 0.0,
                'vader_positive': 0.0,
                'vader_negative': 0.0,
                'vader_neutral': 0.0,
                'custom_score': 0.0,
                'final_sentiment': 'neutral',
                'final_score': 0.0
            }
        
        # Preprocess text for better analysis
        clean_text = self._clean_text(text)
        
        # 1. TextBlob sentiment analysis
        # TextBlob provides polarity (sentiment direction) and subjectivity (opinion vs fact)
        blob = TextBlob(clean_text)
        textblob_polarity = blob.sentiment.polarity      # -1 (negative) to 1 (positive)
        textblob_subjectivity = blob.sentiment.subjectivity  # 0 (objective) to 1 (subjective)
        
        # 2. VADER sentiment analysis
        # VADER is specifically tuned for social media text and handles context well
        vader_scores = self.vader_analyzer.polarity_scores(clean_text)
        
        # 3. Custom financial sentiment analysis
        # Our specialized approach using financial and trade-specific terminology
        custom_score = self._custom_sentiment_analysis(clean_text)
        
        # Combine all scores for final sentiment determination
        # Weight each method equally for balanced analysis
        final_score = (textblob_polarity + vader_scores['compound'] + custom_score) / 3
        
        # Classify final sentiment with threshold buffer to avoid neutral over-classification
        if final_score > 0.1:
            final_sentiment = 'positive'
        elif final_score < -0.1:
            final_sentiment = 'negative'
        else:
            final_sentiment = 'neutral'
        
        return {
            'textblob_polarity': textblob_polarity,
            'textblob_subjectivity': textblob_subjectivity,
            'vader_compound': vader_scores['compound'],
            'vader_positive': vader_scores['pos'],
            'vader_negative': vader_scores['neg'],
            'vader_neutral': vader_scores['neu'],
            'custom_score': custom_score,
            'final_sentiment': final_sentiment,
            'final_score': final_score
        }
    
    def _clean_text(self, text: str) -> str:
        """
        Clean and preprocess text for improved sentiment analysis accuracy.
        
        This function performs comprehensive text cleaning to remove noise and
        normalize the text for better sentiment analysis results.
        
        Args:
            text (str): Raw text to clean
            
        Returns:
            str: Cleaned and preprocessed text
        """
        if not text:
            return ""
        
        # Convert to lowercase for consistent processing
        text = text.lower()
        
        # Remove URLs (common in news articles)
        text = re.sub(r'http\S+', '', text)
        text = re.sub(r'www\S+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove special characters but preserve important punctuation for sentiment
        # Keep periods, commas, exclamation marks, question marks as they affect sentiment
        text = re.sub(r'[^\w\s!?.,;:]', ' ', text)
        
        # Remove extra whitespace and normalize spacing
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def _custom_sentiment_analysis(self, text: str) -> float:
        """
        Perform custom sentiment analysis focused on financial and trade context.
        
        This method uses specialized word lists and weighting schemes designed
        specifically for analyzing financial and trade-related news content.
        It gives higher weight to trade-specific terminology.
        
        Args:
            text (str): Cleaned text to analyze
            
        Returns:
            float: Custom sentiment score between -1 (very negative) and 1 (very positive)
        """
        # Split text into individual words for analysis
        words = text.split()
        positive_count = 0
        negative_count = 0
        
        # Analyze each word for sentiment indicators
        for word in words:
            # Check tariff-specific words first with higher weighting
            # These terms have specific implications in trade policy contexts
            if word in self.tariff_specific_words['positive']:
                positive_count += 2  # Double weight for trade-positive terms
            elif word in self.tariff_specific_words['negative']:
                negative_count += 2  # Double weight for trade-negative terms
            # Check against general positive financial terms
            elif word in self.positive_words:
                positive_count += 1
            # Check against general negative financial terms
            elif word in self.negative_words:
                negative_count += 1
        
        # Calculate normalized sentiment score
        total_words = len(words)
        if total_words == 0:
            return 0.0
        
        # Calculate ratios of positive and negative words
        positive_ratio = positive_count / total_words
        negative_ratio = negative_count / total_words
        
        # Use hyperbolic tangent to normalize the difference to [-1, 1] range
        # This provides a smooth, bounded sentiment score
        return np.tanh(positive_ratio - negative_ratio)
    
    def analyze_news_dataframe(self, news_df: pd.DataFrame) -> pd.DataFrame:
        """
        Analyze sentiment for an entire DataFrame of news articles.
        
        This method processes multiple news articles in batch, combining title,
        description, and content for comprehensive sentiment analysis. It provides
        progress tracking for user feedback during processing.
        
        Args:
            news_df (pd.DataFrame): DataFrame containing news articles with columns:
                - title: Article headlines
                - description: Article summaries
                - content: Article content (optional)
                
        Returns:
            pd.DataFrame: Original DataFrame enhanced with sentiment analysis columns:
                - combined_text: Concatenated text used for analysis
                - All sentiment scores from analyze_text_sentiment()
        """
        if news_df.empty:
            return news_df
        
        # Combine multiple text fields for comprehensive analysis
        # Use fillna('') to handle missing values gracefully
        news_df['combined_text'] = (
            news_df['title'].fillna('') + ' ' + 
            news_df['description'].fillna('') + ' ' + 
            news_df['content'].fillna('')
        )
        
        # Process each article with progress tracking
        sentiment_results = []
        
        # Create progress bar for user feedback
        progress_bar = st.progress(0)
        total_articles = len(news_df)
        
        # Analyze sentiment for each article
        for i, text in enumerate(news_df['combined_text']):
            # Perform comprehensive sentiment analysis
            sentiment = self.analyze_text_sentiment(text)
            sentiment_results.append(sentiment)
            
            # Update progress bar
            progress_bar.progress((i + 1) / total_articles)
        
        # Convert sentiment results to DataFrame and combine with original data
        sentiment_df = pd.DataFrame(sentiment_results)
        result_df = pd.concat([news_df, sentiment_df], axis=1)
        
        return result_df
    
    def get_daily_sentiment_scores(self, news_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate daily aggregate sentiment scores from multiple articles.
        
        This method consolidates multiple news articles per day into single
        daily sentiment metrics, providing cleaner data for time series analysis
        and correlation with stock price movements.
        
        Args:
            news_df (pd.DataFrame): DataFrame with sentiment analysis results
                
        Returns:
            pd.DataFrame: Daily aggregated sentiment data with columns:
                - date: Trading date
                - avg_sentiment: Average sentiment score for the day
                - sentiment_std: Standard deviation of sentiment scores
                - article_count: Number of articles analyzed for that day
                - avg_textblob: Average TextBlob polarity score
                - avg_vader: Average VADER compound score
                - avg_custom: Average custom sentiment score
                - dominant_sentiment: Most common sentiment classification
        """
        if news_df.empty or 'published_at' not in news_df.columns:
            return pd.DataFrame()
        
        # Extract date component from publication timestamp
        news_df['date'] = news_df['published_at'].dt.date
        
        # Group articles by date and calculate aggregation statistics
        daily_sentiment = news_df.groupby('date').agg({
            # Final sentiment scores
            'final_score': ['mean', 'std', 'count'],  # Average, std dev, and count
            
            # Individual engine scores for detailed analysis
            'textblob_polarity': 'mean',
            'vader_compound': 'mean',
            'custom_score': 'mean',
            
            # Dominant sentiment classification (most common)
            'final_sentiment': lambda x: x.mode().iloc[0] if len(x) > 0 else 'neutral'
        }).reset_index()
        
        # Flatten multi-level column names for easier access
        daily_sentiment.columns = [
            'date', 'avg_sentiment', 'sentiment_std', 'article_count',
            'avg_textblob', 'avg_vader', 'avg_custom', 'dominant_sentiment'
        ]
        
        # Handle missing standard deviation values (single article days)
        daily_sentiment['sentiment_std'] = daily_sentiment['sentiment_std'].fillna(0)
        
        return daily_sentiment
    
    def get_sentiment_by_keyword(self, news_df: pd.DataFrame) -> pd.DataFrame:
        """
        Analyze sentiment breakdown by tariff-related keywords.
        
        This method provides insights into how different types of trade news
        (tariffs vs trade deals vs sanctions, etc.) affect overall sentiment.
        
        Args:
            news_df (pd.DataFrame): DataFrame with sentiment analysis and keyword data
                
        Returns:
            pd.DataFrame: Keyword-based sentiment analysis with columns:
                - keyword: The tariff-related keyword
                - avg_sentiment: Average sentiment for articles matching this keyword
                - sentiment_std: Standard deviation of sentiment scores
                - article_count: Number of articles for this keyword
                - sentiment_distribution: Dictionary of sentiment classification counts
        """
        if news_df.empty or 'keyword' not in news_df.columns:
            return pd.DataFrame()
        
        # Group by keyword and calculate sentiment statistics
        keyword_sentiment = news_df.groupby('keyword').agg({
            'final_score': ['mean', 'std', 'count'],
            'final_sentiment': lambda x: x.value_counts().to_dict()
        }).reset_index()
        
        # Flatten column names for easier access
        keyword_sentiment.columns = [
            'keyword', 'avg_sentiment', 'sentiment_std', 'article_count', 'sentiment_distribution'
        ]
        
        return keyword_sentiment
    
    def detect_sentiment_events(self, daily_sentiment_df: pd.DataFrame, threshold: float = 2.0) -> pd.DataFrame:
        """
        Detect significant sentiment events (unusual spikes in positive/negative sentiment).
        
        This method identifies days where sentiment deviates significantly from the
        normal range, which may indicate important market-moving news events.
        
        Args:
            daily_sentiment_df (pd.DataFrame): Daily aggregated sentiment data
            threshold (float): Number of standard deviations for event detection (default: 2.0)
                
        Returns:
            pd.DataFrame: Detected sentiment events with columns:
                - date: Date of the event
                - sentiment_score: Sentiment score on that date
                - z_score: Standardized score (number of standard deviations from mean)
                - event_type: 'positive_spike' or 'negative_spike'
                - significance: 'high' or 'extreme' based on z_score magnitude
        """
        if daily_sentiment_df.empty or 'avg_sentiment' not in daily_sentiment_df.columns:
            return pd.DataFrame()
        
        # Calculate sentiment statistics for event detection
        sentiment_mean = daily_sentiment_df['avg_sentiment'].mean()
        sentiment_std = daily_sentiment_df['avg_sentiment'].std()
        
        # Handle case where standard deviation is zero (all same values)
        if sentiment_std == 0:
            return pd.DataFrame()
        
        # Calculate z-scores (standardized scores)
        daily_sentiment_df['z_score'] = (
            daily_sentiment_df['avg_sentiment'] - sentiment_mean
        ) / sentiment_std
        
        # Identify significant events based on z-score threshold
        events = []
        
        for _, row in daily_sentiment_df.iterrows():
            z_score = abs(row['z_score'])
            
            # Check if this qualifies as a significant event
            if z_score >= threshold:
                # Determine event type and significance
                event_type = 'positive_spike' if row['z_score'] > 0 else 'negative_spike'
                significance = 'extreme' if z_score >= 3.0 else 'high'
                
                events.append({
                    'date': row['date'],
                    'sentiment_score': row['avg_sentiment'],
                    'z_score': row['z_score'],
                    'event_type': event_type,
                    'significance': significance,
                    'article_count': row.get('article_count', 0)
                })
        
        return pd.DataFrame(events)

def create_sentiment_summary(news_df: pd.DataFrame, daily_sentiment_df: pd.DataFrame) -> Dict[str, any]:
    """
    Create a comprehensive summary of sentiment analysis results.
    
    This function generates high-level statistics and insights from the sentiment
    analysis, providing key metrics for dashboard display and reporting.
    
    Args:
        news_df (pd.DataFrame): News articles with sentiment analysis
        daily_sentiment_df (pd.DataFrame): Daily aggregated sentiment data
        
    Returns:
        Dict[str, any]: Comprehensive sentiment summary including:
            - total_articles: Total number of articles analyzed
            - date_range: Start and end dates of analysis
            - avg_sentiment: Overall average sentiment score
            - sentiment_distribution: Breakdown of positive/negative/neutral articles
            - most_positive_day: Date with highest sentiment
            - most_negative_day: Date with lowest sentiment
            - sentiment_volatility: Standard deviation of daily sentiment
            - trending_direction: Overall sentiment trend
    """
    summary = {}
    
    if not news_df.empty:
        # Basic article statistics
        summary['total_articles'] = len(news_df)
        summary['date_range'] = {
            'start': news_df['published_at'].min(),
            'end': news_df['published_at'].max()
        }
        
        # Overall sentiment metrics
        summary['avg_sentiment'] = news_df['final_score'].mean()
        
        # Sentiment distribution (percentage breakdown)
        sentiment_counts = news_df['final_sentiment'].value_counts()
        total_articles = len(news_df)
        summary['sentiment_distribution'] = {
            'positive': (sentiment_counts.get('positive', 0) / total_articles) * 100,
            'negative': (sentiment_counts.get('negative', 0) / total_articles) * 100,
            'neutral': (sentiment_counts.get('neutral', 0) / total_articles) * 100
        }
        
        # Most extreme sentiment articles
        if 'final_score' in news_df.columns:
            most_positive_idx = news_df['final_score'].idxmax()
            most_negative_idx = news_df['final_score'].idxmin()
            
            summary['most_positive_article'] = {
                'title': news_df.loc[most_positive_idx, 'title'] if 'title' in news_df.columns else '',
                'score': news_df.loc[most_positive_idx, 'final_score']
            }
            
            summary['most_negative_article'] = {
                'title': news_df.loc[most_negative_idx, 'title'] if 'title' in news_df.columns else '',
                'score': news_df.loc[most_negative_idx, 'final_score']
            }
    
    if not daily_sentiment_df.empty:
        # Daily sentiment analysis
        summary['most_positive_day'] = {
            'date': daily_sentiment_df.loc[daily_sentiment_df['avg_sentiment'].idxmax(), 'date'],
            'score': daily_sentiment_df['avg_sentiment'].max()
        }
        
        summary['most_negative_day'] = {
            'date': daily_sentiment_df.loc[daily_sentiment_df['avg_sentiment'].idxmin(), 'date'],
            'score': daily_sentiment_df['avg_sentiment'].min()
        }
        
        # Sentiment volatility (how much sentiment varies day to day)
        summary['sentiment_volatility'] = daily_sentiment_df['avg_sentiment'].std()
        
        # Trend analysis (simple linear trend)
        if len(daily_sentiment_df) > 1:
            # Calculate basic trend direction
            first_half_avg = daily_sentiment_df.head(len(daily_sentiment_df)//2)['avg_sentiment'].mean()
            second_half_avg = daily_sentiment_df.tail(len(daily_sentiment_df)//2)['avg_sentiment'].mean()
            
            if second_half_avg > first_half_avg + 0.05:
                summary['trending_direction'] = 'improving'
            elif second_half_avg < first_half_avg - 0.05:
                summary['trending_direction'] = 'declining'
            else:
                summary['trending_direction'] = 'stable'
    
    return summary