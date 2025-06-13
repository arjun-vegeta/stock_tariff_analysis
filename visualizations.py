"""
Interactive visualization tools for stock and tariff analysis.
Uses Plotly for dynamic charts and Streamlit for display.
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import seaborn as sns
import matplotlib.pyplot as plt

class StockTariffVisualizer:
    """Interactive visualization tools for stock and tariff analysis."""
    
    def __init__(self):
        """Initialize visualizer with color schemes for consistent styling."""
        # Color scheme for financial data visualization
        self.colors = {
            'positive': '#00cc96',    # Green
            'negative': '#ff6692',    # Red
            'neutral': '#636efa',     # Blue
            'primary': '#1f77b4',     # Primary blue
            'secondary': '#ff7f0e',   # Orange
            'background': '#f8f9fa'   # Light gray
        }
        
        # Sector-specific colors
        self.sector_colors = {
            'Technology': '#636efa',      # Blue - tech/innovation
            'Manufacturing': '#ef553b',   # Red - industrial/manufacturing
            'Automotive': '#00cc96',      # Green - automotive/mobility
            'Energy': '#ab63fa',          # Purple - energy/utilities
            'Materials': '#ffa15a',       # Orange - materials/commodities
            'Consumer_Goods': '#19d3f3',  # Cyan - consumer products
            'Financial': '#ff6692'        # Pink - financial services
        }
    
    def create_stock_price_chart(self, stock_data: pd.DataFrame, 
                               selected_stocks: List[str] = None) -> go.Figure:
        """Create interactive multi-stock price chart."""
        fig = go.Figure()
        
        # Handle empty data case
        if stock_data.empty:
            fig.add_annotation(
                text="No stock data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16, color="gray")
            )
            return fig
        
        # Filter data for selected stocks if specified
        if selected_stocks:
            stock_data = stock_data[stock_data['Symbol'].isin(selected_stocks)]
        
        # Create individual traces for each stock
        for symbol in stock_data['Symbol'].unique():
            # Extract and sort data for this specific stock
            symbol_data = stock_data[stock_data['Symbol'] == symbol].copy()
            symbol_data = symbol_data.sort_values('Date')
            
            # Add stock price line to the chart
            fig.add_trace(
                go.Scatter(
                    x=symbol_data['Date'],
                    y=symbol_data['Close'],
                    mode='lines',                    # Line chart (no markers for cleaner appearance)
                    name=symbol,                     # Legend name
                    line=dict(width=2),             # Line thickness
                    hovertemplate=(                  # Custom hover information
                        '%{fullData.name}<br>' +
                        'Date: %{x}<br>' +
                        'Price: $%{y:.2f}<extra></extra>'
                    )
                )
            )
        
        # Configure chart layout and styling
        fig.update_layout(
            title='Stock Price Trends',
            xaxis_title='Date',
            yaxis_title='Stock Price ($)',
            hovermode='x unified',              # Show all values at same x-position
            height=500,                         # Chart height in pixels
            showlegend=True,                    # Enable legend
            legend=dict(                        # Legend positioning
                orientation="h",                # Horizontal legend
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        return fig
    
    def create_sector_performance_chart(self, sector_data: Dict[str, pd.DataFrame]) -> go.Figure:
        """Create sector performance comparison chart."""
        fig = go.Figure()
        sector_performance = []
        
        # Calculate performance metrics for each sector
        for sector, data in sector_data.items():
            if not data.empty:
                # Find the date range for performance calculation
                latest_date = data['Date'].max()
                earliest_date = data['Date'].min()
                
                # Get average prices for start and end of period
                latest_data = data[data['Date'] == latest_date]
                earliest_data = data[data['Date'] == earliest_date]
                
                if not latest_data.empty and not earliest_data.empty:
                    # Calculate sector-wide average performance
                    latest_avg = latest_data['Close'].mean()
                    earliest_avg = earliest_data['Close'].mean()
                    
                    # Calculate percentage return
                    performance = ((latest_avg - earliest_avg) / earliest_avg) * 100
                    
                    sector_performance.append({
                        'Sector': sector,
                        'Performance': performance,
                        'Color': self.sector_colors.get(sector, '#636efa')
                    })
        
        # Create the performance chart if data is available
        if sector_performance:
            df = pd.DataFrame(sector_performance)
            
            # Create bar chart with color-coded performance
            fig = px.bar(
                df, 
                x='Sector', 
                y='Performance',
                color='Performance',
                color_continuous_scale=['red', 'yellow', 'green'],
                title='Sector Performance Comparison (%)',
                labels={'Performance': 'Return (%)'}
            )
            
            # Add zero reference line for better interpretation
            fig.add_hline(y=0, line_dash="dash", line_color="black", line_width=1)
            
            # Update layout for better presentation
            fig.update_layout(
                xaxis_title='Sector',
                yaxis_title='Performance (%)',
                height=400,
                showlegend=False                    # Hide color scale legend for cleaner look
            )
        
        return fig
    
    def create_sentiment_timeline(self, daily_sentiment: pd.DataFrame) -> go.Figure:
        """Create timeline chart of sentiment scores and article counts."""
        # Create subplot with secondary y-axis
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                           vertical_spacing=0.1,
                           row_heights=[0.7, 0.3])
        
        # Add sentiment score line chart
        fig.add_trace(
            go.Scatter(
                x=daily_sentiment['date'],
                y=daily_sentiment['avg_sentiment'],
                mode='lines+markers',
                name='Average Sentiment',
                line=dict(color=self.colors['primary'], width=2),
                marker=dict(size=4),
                fill='tonexty',
                fillcolor='rgba(31, 119, 180, 0.1)',
                hovertemplate=(
                    'Date: %{x}<br>' +
                    'Sentiment: %{y:.3f}<extra></extra>'
                )
            ),
            row=1, col=1
        )
        
        # Add zero reference line
        fig.add_hline(
            y=0, 
            line_dash="dash", 
            line_color="gray", 
            row=1, col=1,
            annotation_text="Neutral"
        )
        
        # Add article count bar chart
        fig.add_trace(
            go.Bar(
                x=daily_sentiment['date'],
                y=daily_sentiment['article_count'],
                name='Article Count',
                marker_color=self.colors['secondary'],
                hovertemplate=(
                    'Date: %{x}<br>' +
                    'Articles: %{y}<extra></extra>'
                )
            ),
            row=2, col=1
        )
        
        # Configure layout
        fig.update_layout(
            title='News Sentiment Analysis Over Time',
            height=600,
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        fig.update_xaxes(title_text="Date", row=2, col=1)
        fig.update_yaxes(title_text="Sentiment Score", row=1, col=1)
        
        return fig
    
    def create_correlation_heatmap(self, correlation_data: pd.DataFrame) -> go.Figure:
        """Create correlation heatmap for stock-sentiment relationships."""
        if correlation_data.empty:
            fig = go.Figure()
            fig.add_annotation(
                text="No correlation data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16, color="gray")
            )
            return fig
        
        # Create heatmap using Plotly
        fig = go.Figure(data=go.Heatmap(
            z=correlation_data.values,          # Correlation coefficient matrix
            x=correlation_data.columns,         # Column labels (variables)
            y=correlation_data.index,           # Row labels (variables)
            colorscale='RdBu',                  # Red-Blue color scale (red=negative, blue=positive)
            zmid=0,                             # Center colorscale at zero
            hoverongaps=False,                  # Don't show hover on missing data
            hovertemplate=(                     # Custom hover information
                '%{y} vs %{x}<br>' +
                'Correlation: %{z:.3f}<extra></extra>'
            ),
            colorbar=dict(                      # Colorbar configuration
                title="Correlation<br>Coefficient",
                titleside="right"
            )
        ))
        
        # Configure layout
        fig.update_layout(
            title='Stock Price vs Sentiment Correlation',
            height=500,
            xaxis=dict(side="bottom"),          # X-axis labels at bottom
            yaxis=dict(side="left")             # Y-axis labels at left
        )
        
        return fig
    
    def create_volatility_chart(self, stock_data: pd.DataFrame, 
                              sentiment_events: pd.DataFrame = None) -> go.Figure:
        """Create volatility analysis chart with sentiment event markers."""
        fig = go.Figure()
        
        # Handle empty data case
        if stock_data.empty:
            fig.add_annotation(
                text="No volatility data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16, color="gray")
            )
            return fig
        
        # Calculate average volatility across all stocks by date
        if 'Volatility' in stock_data.columns:
            daily_volatility = stock_data.groupby('Date')['Volatility'].mean().reset_index()
            
            # Add volatility line chart
            fig.add_trace(
                go.Scatter(
                    x=daily_volatility['Date'],
                    y=daily_volatility['Volatility'],
                    mode='lines',
                    name='Average Volatility',
                    line=dict(color=self.colors['primary'], width=2),
                    hovertemplate=(
                        'Date: %{x}<br>' +
                        'Volatility: %{y:.4f}<extra></extra>'
                    )
                )
            )
            
            # Add sentiment event markers if provided
            if sentiment_events is not None and not sentiment_events.empty:
                for _, event in sentiment_events.iterrows():
                    event_color = self.colors['positive'] if event['event_type'] == 'positive_spike' else self.colors['negative']
                    
                    fig.add_vline(
                        x=event['date'],
                        line_dash="dash",
                        line_color=event_color,
                        annotation_text=f"{event['event_type']}<br>({event['significance']})",
                        annotation_position="top"
                    )
        
        # Configure layout
        fig.update_layout(
            title='Stock Price Volatility Analysis',
            xaxis_title='Date',
            yaxis_title='Volatility (Standard Deviation)',
            height=400,
            showlegend=True
        )
        
        return fig
    
    def create_sentiment_distribution(self, news_data: pd.DataFrame) -> go.Figure:
        """Create sentiment distribution pie chart."""
        fig = go.Figure()
        
        # Handle empty data or missing sentiment column
        if news_data.empty or 'final_sentiment' not in news_data.columns:
            fig.add_annotation(
                text="No sentiment distribution data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16, color="gray")
            )
            return fig
        
        # Calculate sentiment distribution
        sentiment_counts = news_data['final_sentiment'].value_counts()
        
        # Create pie chart
        fig.add_trace(
            go.Pie(
                labels=sentiment_counts.index,
                values=sentiment_counts.values,
                hole=0.3,
                marker_colors=[
                    self.colors['positive'] if label == 'positive' 
                    else self.colors['negative'] if label == 'negative' 
                    else self.colors['neutral'] 
                    for label in sentiment_counts.index
                ],
                hovertemplate=(
                    '%{label}<br>' +
                    'Articles: %{value}<br>' +
                    'Percentage: %{percent}<extra></extra>'
                )
            )
        )
        
        # Configure layout
        fig.update_layout(
            title='Sentiment Distribution of News Articles',
            height=400,
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=-0.1, xanchor="center", x=0.5)
        )
        
        return fig
    
    def create_returns_comparison(self, stock_data: pd.DataFrame, 
                                sentiment_data: pd.DataFrame) -> go.Figure:
        """Create chart comparing stock returns with sentiment scores."""
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        if stock_data.empty or sentiment_data.empty:
            fig.add_annotation(
                text="Insufficient data for returns comparison",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16, color="gray")
            )
            return fig
        
        # Calculate daily average returns
        if 'Daily_Return' in stock_data.columns:
            daily_returns = stock_data.groupby('Date')['Daily_Return'].mean().reset_index()
            
            # Add returns line
            fig.add_trace(
                go.Scatter(
                    x=daily_returns['Date'],
                    y=daily_returns['Daily_Return'] * 100,
                    mode='lines',
                    name='Average Daily Return (%)',
                    line=dict(color=self.colors['primary'], width=2)
                ),
                secondary_y=False
            )
        
        # Add sentiment line (secondary y-axis)
        if 'avg_sentiment' in sentiment_data.columns:
            fig.add_trace(
                go.Scatter(
                    x=sentiment_data['date'],
                    y=sentiment_data['avg_sentiment'],
                    mode='lines',
                    name='Average Sentiment',
                    line=dict(color=self.colors['secondary'], width=2)
                ),
                secondary_y=True
            )
        
        # Configure axes
        fig.update_xaxes(title_text="Date")
        fig.update_yaxes(title_text="Daily Return (%)", secondary_y=False)
        fig.update_yaxes(title_text="Sentiment Score", secondary_y=True)
        
        # Configure layout
        fig.update_layout(
            title='Stock Returns vs News Sentiment Comparison',
            height=400,
            showlegend=True
        )
        
        return fig
    
    def create_keyword_sentiment_chart(self, keyword_sentiment: pd.DataFrame) -> go.Figure:
        """Create chart showing sentiment by tariff keywords."""
        fig = go.Figure()
        
        if keyword_sentiment.empty:
            fig.add_annotation(
                text="No keyword sentiment data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16, color="gray")
            )
            return fig
        
        # Create bar chart
        fig.add_trace(
            go.Bar(
                x=keyword_sentiment['keyword'],
                y=keyword_sentiment['avg_sentiment'],
                marker_color=[
                    self.colors['positive'] if sentiment > 0 
                    else self.colors['negative'] if sentiment < 0 
                    else self.colors['neutral']
                    for sentiment in keyword_sentiment['avg_sentiment']
                ],
                hovertemplate=(
                    'Keyword: %{x}<br>' +
                    'Avg Sentiment: %{y:.3f}<br>' +
                    'Article Count: %{customdata}<extra></extra>'
                ),
                customdata=keyword_sentiment['article_count']
            )
        )
        
        fig.add_hline(y=0, line_dash="dash", line_color="gray")
        
        # Configure layout
        fig.update_layout(
            title='Sentiment Analysis by Tariff Keywords',
            xaxis_title='Keywords',
            yaxis_title='Average Sentiment Score',
            height=400,
            showlegend=False
        )
        
        # Rotate x-axis labels for better readability
        fig.update_xaxes(tickangle=45)
        
        return fig

# Additional utility functions for creating summary displays

def create_summary_metrics_display(stock_summary: Dict, sentiment_summary: Dict) -> None:
    """Display key performance indicators in Streamlit sidebar."""
    st.subheader("📊 Key Metrics")
    
    # Stock metrics section
    if stock_summary:
        st.write("**Stock Analysis:**")
        if 'total_stocks' in stock_summary:
            st.metric("Stocks Analyzed", stock_summary['total_stocks'])
        if 'avg_return' in stock_summary:
            st.metric("Average Return", f"{stock_summary['avg_return']:.2%}")
        if 'avg_volatility' in stock_summary:
            st.metric("Average Volatility", f"{stock_summary['avg_volatility']:.4f}")
    
    # Sentiment metrics section
    if sentiment_summary:
        st.write("**Sentiment Analysis:**")
        if 'total_articles' in sentiment_summary:
            st.metric("Articles Analyzed", sentiment_summary['total_articles'])
        if 'avg_sentiment' in sentiment_summary:
            sentiment_value = sentiment_summary['avg_sentiment']
            sentiment_label = (
                "Positive" if sentiment_value > 0.1 
                else "Negative" if sentiment_value < -0.1 
                else "Neutral"
            )
            st.metric("Overall Sentiment", sentiment_label, f"{sentiment_value:.3f}")
        if 'sentiment_volatility' in sentiment_summary:
            st.metric("Sentiment Volatility", f"{sentiment_summary['sentiment_volatility']:.3f}")

def display_sector_metrics(sector_data: Dict[str, pd.DataFrame]) -> None:
    """Display sector performance metrics in tabular format."""
    if not sector_data:
        st.info("No sector data available for display.")
        return
    
    # Calculate metrics for each sector
    sector_metrics = []
    
    for sector, data in sector_data.items():
        if not data.empty:
            # Calculate sector performance metrics
            latest_prices = data.groupby('Symbol')['Close'].last()
            earliest_prices = data.groupby('Symbol')['Close'].first()
            
            # Performance calculations
            sector_return = ((latest_prices.mean() - earliest_prices.mean()) / earliest_prices.mean()) * 100
            
            if 'Daily_Return' in data.columns:
                sector_volatility = data['Daily_Return'].std() * 100
                max_daily_return = data['Daily_Return'].max() * 100
                min_daily_return = data['Daily_Return'].min() * 100
            else:
                sector_volatility = 0
                max_daily_return = 0
                min_daily_return = 0
            
            sector_metrics.append({
                'Sector': sector,
                'Total Return (%)': f"{sector_return:.2f}",
                'Volatility (%)': f"{sector_volatility:.2f}",
                'Best Day (%)': f"{max_daily_return:.2f}",
                'Worst Day (%)': f"{min_daily_return:.2f}",
                'Stocks Count': data['Symbol'].nunique()
            })
    
    # Display metrics table
    if sector_metrics:
        metrics_df = pd.DataFrame(sector_metrics)
        st.dataframe(metrics_df, use_container_width=True)
    else:
        st.info("No sector metrics could be calculated.")