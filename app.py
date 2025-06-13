#!/usr/bin/env python3
"""
Stock Price & Tariff Impact Analysis Application
Analyzes tariff impacts on stock prices using real-time data and sentiment analysis.
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')  # Suppress warnings for cleaner output

# Import custom modules for different functionalities
from config import config, validate_config
from data_collector import StockDataCollector, NewsDataCollector
from sentiment_analyzer import SentimentAnalyzer
from visualizations import StockTariffVisualizer, create_summary_metrics_display, display_sector_metrics
from utils import DataProcessor, StatisticalAnalyzer, ReportGenerator, PerformanceTracker

# Configure Streamlit page settings
st.set_page_config(
    page_title=config.PAGE_TITLE,
    page_icon=config.PAGE_ICON,
    layout="wide",                      # Use wide layout for better data visualization
    initial_sidebar_state="expanded"    # Start with sidebar open for easy configuration
)

def main():
    """
    Main application function that orchestrates the entire Streamlit interface.
    
    This function:
    1. Sets up the UI layout and sidebar controls
    2. Initializes data collectors and analyzers
    3. Loads and processes stock and news data
    4. Creates interactive tabs for different analysis views
    5. Handles user interactions and data visualization
    """
    
    # Initialize performance tracking for monitoring app performance
    perf_tracker = PerformanceTracker()
    
    # Create main application header
    st.title("📈 Stock Price & Tariff Impact Analysis")
    st.markdown("""
    Analyze the impact of tariffs and trade policies on stock prices across different sectors.
    This application provides real-time data analysis, sentiment tracking, and correlation insights.
    """)
    
 
    # SIDEBAR CONFIGURATION
 
    st.sidebar.title("🔧 Configuration")
    
    # Validate API configuration and show status
    if not validate_config():
        st.sidebar.error("⚠️ API keys not configured properly!")
        st.sidebar.info("""
        To use this application with real data:
        1. Get API keys from NewsAPI and Alpha Vantage
        2. Set them as environment variables or update config.py
        
        Current configuration uses sample data.
        """)
    else:
        st.sidebar.success("✅ API keys configured correctly!")
    
    # Data collection settings section
    st.sidebar.subheader("📊 Data Settings")
    
    # Allow users to select which sectors to analyze
    selected_sectors = st.sidebar.multiselect(
        "Select Sectors to Analyze",
        options=list(config.STOCK_SYMBOLS.keys()),
        default=list(config.STOCK_SYMBOLS.keys())[:3],  # Default to first 3 sectors
        help="Choose which stock market sectors to include in the analysis"
    )
    
    # Time period selection for stock data
    analysis_period = st.sidebar.selectbox(
        "Analysis Period",
        options=["1mo", "3mo", "6mo", "1y", "2y"],
        index=3,  # Default to 1 year
        help="Select the time period for stock data analysis"
    )
    
    # News analysis period configuration
    news_days = st.sidebar.slider(
        "News Analysis Period (days)",
        min_value=7,
        max_value=90,
        value=30,
        help="Number of days to look back for news articles"
    )
    
    # Advanced analysis options
    with st.sidebar.expander("🔬 Advanced Options"):
        show_technical_indicators = st.checkbox("Show Technical Indicators", value=True,
                                               help="Display RSI, MACD, Bollinger Bands, etc.")
        show_correlation_analysis = st.checkbox("Show Correlation Analysis", value=True,
                                               help="Analyze correlations between stocks and sentiment")
        show_statistical_tests = st.checkbox("Show Statistical Tests", value=False,
                                            help="Perform Granger causality and other statistical tests")
        anomaly_detection = st.checkbox("Anomaly Detection", value=True,
                                       help="Detect unusual price movements and outliers")
    
 
    # MAIN CONTENT TABS
 
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Overview", 
        "📈 Stock Analysis", 
        "📰 News & Sentiment", 
        "🔗 Correlations", 
        "📋 Reports"
    ])
    
 
    # DATA COLLECTION AND PROCESSING
 
    
    # Initialize all data collectors and analyzers
    stock_collector = StockDataCollector()
    news_collector = NewsDataCollector()
    sentiment_analyzer = SentimentAnalyzer()
    visualizer = StockTariffVisualizer()
    
    # Load and process data with progress indication
    with st.spinner("Loading data..."):
        perf_tracker.start_timing("data_collection")
        
        # Collect stock data for selected sectors
        sector_stock_data = {}
        all_stock_symbols = []
        
        # Process each selected sector
        for sector in selected_sectors:
            symbols = config.STOCK_SYMBOLS[sector]
            all_stock_symbols.extend(symbols)
            
            # Fetch stock data for this sector
            sector_data = stock_collector.get_stock_data(symbols, period=analysis_period)
            if not sector_data.empty:
                sector_data['Sector'] = sector  # Add sector label
                sector_stock_data[sector] = sector_data
        
        # Combine all stock data into a single DataFrame
        if sector_stock_data:
            combined_stock_data = pd.concat(sector_stock_data.values(), ignore_index=True)
            # Calculate additional technical indicators and metrics
            combined_stock_data = DataProcessor.calculate_stock_metrics(combined_stock_data)
        else:
            combined_stock_data = pd.DataFrame()
        
        # Collect news data related to tariffs and trade
        news_data = news_collector.get_tariff_news(days_back=news_days)
        
        # Perform sentiment analysis on news articles
        if not news_data.empty:
            news_with_sentiment = sentiment_analyzer.analyze_news_dataframe(news_data)
            daily_sentiment = sentiment_analyzer.get_daily_sentiment_scores(news_with_sentiment)
            keyword_sentiment = sentiment_analyzer.get_sentiment_by_keyword(news_with_sentiment)
        else:
            # Initialize empty DataFrames if no news data available
            news_with_sentiment = pd.DataFrame()
            daily_sentiment = pd.DataFrame()
            keyword_sentiment = pd.DataFrame()
        
        perf_tracker.end_timing("data_collection")
    
 
    # TAB 1: OVERVIEW
 
    with tab1:
        st.header("📊 Market Overview")
        
        if not combined_stock_data.empty:
            # Create two-column layout for overview content
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Display sector performance comparison chart
                sector_performance_fig = visualizer.create_sector_performance_chart(sector_stock_data)
                st.plotly_chart(sector_performance_fig, use_container_width=True)
            
            with col2:
                # Generate and display key metrics
                stock_summary = ReportGenerator.generate_stock_summary(combined_stock_data)
                sentiment_summary = ReportGenerator.generate_sentiment_summary(
                    news_with_sentiment, daily_sentiment
                ) if not daily_sentiment.empty else {}
                
                # Display summary metrics in sidebar format
                create_summary_metrics_display(stock_summary, sentiment_summary)
        
        # Recent news highlights section
        st.subheader("📰 Recent News Highlights")
        if not news_with_sentiment.empty:
            # Show the 5 most recent news articles with sentiment indicators
            recent_news = news_with_sentiment.sort_values('published_at', ascending=False).head(5)
            
            for _, article in recent_news.iterrows():
                # Color-code sentiment with emoji indicators
                sentiment_color = {
                    'positive': '🟢',
                    'negative': '🔴',
                    'neutral': '🟡'
                }.get(article['final_sentiment'], '🟡')
                
                # Display article information
                st.write(f"{sentiment_color} **{article['title']}**")
                st.write(f"*{article['source']} - {article['published_at'].strftime('%Y-%m-%d %H:%M')}*")
                st.write(article['description'])
                st.write(f"Sentiment Score: {article['final_score']:.3f}")
                st.write("---")
        else:
            st.info("No recent news data available.")
    
 
    # TAB 2: STOCK ANALYSIS
 
    with tab2:
        st.header("📈 Stock Price Analysis")
        
        if not combined_stock_data.empty:
            # Stock selection interface for detailed analysis
            available_stocks = sorted(combined_stock_data['Symbol'].unique())
            selected_stocks = st.multiselect(
                "Select Stocks for Detailed Analysis",
                options=available_stocks,
                default=available_stocks[:5],  # Default to first 5 stocks
                help="Choose specific stocks to analyze in detail"
            )
            
            if selected_stocks:
                # Create main stock price chart
                price_chart = visualizer.create_stock_price_chart(
                    combined_stock_data, selected_stocks
                )
                st.plotly_chart(price_chart, use_container_width=True)
                
                # Technical indicators section (optional)
                if show_technical_indicators:
                    st.subheader("🔧 Technical Analysis")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Volatility analysis chart
                        volatility_chart = visualizer.create_volatility_chart(combined_stock_data)
                        st.plotly_chart(volatility_chart, use_container_width=True)
                    
                    with col2:
                        # Returns comparison with sentiment data
                        if not daily_sentiment.empty:
                            returns_chart = visualizer.create_returns_comparison(
                                combined_stock_data, daily_sentiment
                            )
                            st.plotly_chart(returns_chart, use_container_width=True)
                
                # Anomaly detection section (optional)
                if anomaly_detection:
                    st.subheader("🚨 Anomaly Detection")
                    
                    # Analyze anomalies for selected stocks (limit to 3 for performance)
                    for stock in selected_stocks[:3]:
                        try:
                            # Extract daily returns for the stock
                            stock_data = combined_stock_data[
                                combined_stock_data['Symbol'] == stock
                            ]['Daily_Return']
                            
                            # Perform anomaly detection with sufficient data
                            if not stock_data.empty and len(stock_data.dropna()) > 10:
                                anomalies = DataProcessor.detect_anomalies(stock_data)
                                
                                if not anomalies.empty:
                                    st.write(f"**{stock} - Detected Anomalies:**")
                                    st.dataframe(anomalies)
                                else:
                                    st.write(f"**{stock}**: No anomalies detected")
                            else:
                                st.write(f"**{stock}**: Insufficient data for anomaly detection")
                        except Exception as e:
                            st.write(f"**{stock}**: Error in anomaly detection: {str(e)}")
            
            # Sector performance metrics
            st.subheader("🏭 Sector Performance Metrics")
            display_sector_metrics(sector_stock_data)
        
        else:
            st.warning("No stock data available. Please check your configuration.")
    
 
    # TAB 3: NEWS & SENTIMENT ANALYSIS
 
    with tab3:
        st.header("📰 News & Sentiment Analysis")
        
        if not daily_sentiment.empty:
            # Sentiment timeline visualization
            sentiment_timeline = visualizer.create_sentiment_timeline(daily_sentiment)
            st.plotly_chart(sentiment_timeline, use_container_width=True)
            
            # Two-column layout for additional sentiment analysis
            col1, col2 = st.columns(2)
            
            with col1:
                # Sentiment distribution chart
                sentiment_dist = visualizer.create_sentiment_distribution(news_with_sentiment)
                st.plotly_chart(sentiment_dist, use_container_width=True)
            
            with col2:
                # Keyword-based sentiment analysis
                if not keyword_sentiment.empty:
                    keyword_chart = visualizer.create_keyword_sentiment_chart(keyword_sentiment)
                    st.plotly_chart(keyword_chart, use_container_width=True)
            
            # Significant sentiment events detection
            st.subheader("📅 Significant Sentiment Events")
            sentiment_events = sentiment_analyzer.detect_sentiment_events(daily_sentiment)
            
            if not sentiment_events.empty:
                st.dataframe(sentiment_events)
            else:
                st.info("No significant sentiment events detected in the selected period.")
            
            # Complete news articles table
            st.subheader("📋 News Articles")
            if not news_with_sentiment.empty:
                # Display filtered columns for better readability
                display_news = news_with_sentiment[[
                    'title', 'source', 'published_at', 'final_sentiment', 'final_score'
                ]].sort_values('published_at', ascending=False)
                
                st.dataframe(display_news, use_container_width=True)
        
        else:
            st.warning("No sentiment data available.")
    
 
    # TAB 4: CORRELATION ANALYSIS
 
    with tab4:
        st.header("🔗 Correlation Analysis")
        
        if show_correlation_analysis and not combined_stock_data.empty and not daily_sentiment.empty:
            # Calculate correlation matrix between stocks and sentiment
            correlation_matrix = DataProcessor.calculate_correlation_matrix(
                combined_stock_data, daily_sentiment
            )
            
            if not correlation_matrix.empty:
                # Display correlation heatmap
                correlation_heatmap = visualizer.create_correlation_heatmap(correlation_matrix)
                st.plotly_chart(correlation_heatmap, use_container_width=True)
                
                # Generate correlation analysis report
                correlation_report = ReportGenerator.generate_correlation_report(correlation_matrix)
                
                # Display strongest correlations
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("📊 Strongest Positive Correlation")
                    if correlation_report and 'strongest_positive_correlation' in correlation_report:
                        pos_corr = correlation_report['strongest_positive_correlation']
                        st.write(f"• **{pos_corr['stock']}**: {pos_corr['correlation']:.3f}")
                    else:
                        st.write("No positive correlations found.")
                
                with col2:
                    st.subheader("📊 Strongest Negative Correlation")
                    if correlation_report and 'strongest_negative_correlation' in correlation_report:
                        neg_corr = correlation_report['strongest_negative_correlation']
                        st.write(f"• **{neg_corr['stock']}**: {neg_corr['correlation']:.3f}")
                    else:
                        st.write("No negative correlations found.")
                
                # Advanced statistical tests (optional)
                if show_statistical_tests:
                    st.subheader("🧪 Statistical Tests")
                    
                    # Granger causality analysis
                    st.write("**Granger Causality Analysis:**")
                    
                    if 'avg_sentiment' in daily_sentiment.columns:
                        sentiment_series = daily_sentiment.set_index('date')['avg_sentiment']
                        
                        # Test causality for selected stocks (limited for performance)
                        for stock in selected_stocks[:3]:
                            stock_returns = combined_stock_data[
                                combined_stock_data['Symbol'] == stock
                            ].set_index('Date')['Daily_Return']
                            
                            # Align time series data
                            aligned_data = pd.concat([stock_returns, sentiment_series], axis=1).dropna()
                            
                            if len(aligned_data) > 20:
                                # Perform Granger causality test
                                causality_result = StatisticalAnalyzer.perform_granger_causality_test(
                                    aligned_data.iloc[:, 1], aligned_data.iloc[:, 0]
                                )
                                
                                st.write(f"**{stock}:**")
                                st.write(f"  - P-value: {causality_result['p_value']:.4f}")
                                st.write(f"  - Causality: {'Yes' if causality_result['causality'] else 'No'}")
                
                # Sector impact analysis
                st.subheader("🎯 Sector Impact Scores")
                impact_scores = DataProcessor.calculate_sector_impact_scores(
                    sector_stock_data, daily_sentiment
                )
                
                if impact_scores:
                    # Create and display impact scores table
                    impact_df = pd.DataFrame([
                        {'Sector': sector, 'Impact Score': score}
                        for sector, score in impact_scores.items()
                    ]).sort_values('Impact Score', ascending=False)
                    
                    st.dataframe(impact_df, use_container_width=True)
            
            else:
                st.warning("Unable to calculate correlations with available data.")
        
        else:
            st.info("Enable correlation analysis in the sidebar to view this section.")
    
 
    # TAB 5: COMPREHENSIVE REPORTS
 
    with tab5:
        st.header("📋 Analysis Reports")
        
        # Generate comprehensive analysis report
        if st.button("📄 Generate Full Report"):
            with st.spinner("Generating comprehensive report..."):
                
                # Stock analysis summary
                if not combined_stock_data.empty:
                    st.subheader("📈 Stock Analysis Summary")
                    stock_summary = ReportGenerator.generate_stock_summary(combined_stock_data)
                    
                    # Display all summary metrics
                    for key, value in stock_summary.items():
                        if isinstance(value, dict):
                            st.write(f"**{key.replace('_', ' ').title()}:**")
                            for subkey, subvalue in value.items():
                                st.write(f"  - {subkey}: {subvalue}")
                        else:
                            st.write(f"**{key.replace('_', ' ').title()}:** {value}")
                
                # Sentiment analysis summary
                if not daily_sentiment.empty:
                    st.subheader("📰 Sentiment Analysis Summary")
                    sentiment_summary = ReportGenerator.generate_sentiment_summary(
                        news_with_sentiment, daily_sentiment
                    )
                    
                    # Display sentiment metrics
                    for key, value in sentiment_summary.items():
                        if isinstance(value, dict):
                            st.write(f"**{key.replace('_', ' ').title()}:**")
                            for subkey, subvalue in value.items():
                                st.write(f"  - {subkey}: {subvalue}")
                        else:
                            st.write(f"**{key.replace('_', ' ').title()}:** {value}")
                
                # Application performance metrics
                st.subheader("⚡ Performance Metrics")
                perf_summary = perf_tracker.get_summary()
                for operation, duration in perf_summary.items():
                    st.write(f"**{operation.replace('_', ' ').title()}:** {duration:.2f} seconds")
        
        # Data download section
        st.subheader("💾 Download Data")
        
        # Three-column layout for download buttons
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if not combined_stock_data.empty:
                csv_stock = combined_stock_data.to_csv(index=False)
                st.download_button(
                    label="📊 Download Stock Data",
                    data=csv_stock,
                    file_name=f"stock_data_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv",
                    help="Download complete stock data with technical indicators"
                )
        
        with col2:
            if not news_with_sentiment.empty:
                csv_news = news_with_sentiment.to_csv(index=False)
                st.download_button(
                    label="📰 Download News Data",
                    data=csv_news,
                    file_name=f"news_sentiment_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv",
                    help="Download news articles with sentiment analysis"
                )
        
        with col3:
            if not daily_sentiment.empty:
                csv_sentiment = daily_sentiment.to_csv(index=False)
                st.download_button(
                    label="📈 Download Sentiment Data",
                    data=csv_sentiment,
                    file_name=f"daily_sentiment_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv",
                    help="Download daily aggregated sentiment scores"
                )
    
 
    # APPLICATION FOOTER
 
    st.markdown("---")
    st.markdown("""
    **Note:** This application uses free APIs and may have rate limits. 
    For production use, consider upgrading to paid API plans.
    
    *Built with Streamlit, yfinance, NewsAPI, Alpha Vantage, and various ML/NLP libraries.*
    
    **Features:**
    - Real-time stock data analysis
    - News sentiment analysis  
    - Interactive visualizations
    - Statistical correlation analysis
    - Technical indicators (RSI, MACD, Bollinger Bands)
    - Anomaly detection
    - Comprehensive reporting
    """)

if __name__ == "__main__":
    # Run the main application
    main()
