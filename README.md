# 📈 Stock Price & Tariff Impact Analysis

A comprehensive Streamlit application for analyzing the impact of tariffs and trade policies on stock prices across different market sectors. This application combines real-time financial data with news sentiment analysis to provide insights into how trade policies affect market performance.

## Features

### **Real-Time Data Analysis**
- **Live Stock Data**: Real-time price data from Yahoo Finance (free)
- **News Collection**: Current tariff-related news from NewsAPI
- **Alternative Data Sources**: Alpha Vantage integration for enhanced reliability
- **Multi-Sector Analysis**: Technology, Manufacturing, Automotive, Energy, Materials, Consumer Goods, and Financial sectors

### **Advanced Sentiment Analysis**
- **Multi-Engine Analysis**: TextBlob, VADER, and custom financial sentiment analysis
- **Keyword Filtering**: Intelligent filtering for tariff and trade-related content
- **Daily Aggregation**: Consolidated daily sentiment scores and trends
- **Event Detection**: Automatic identification of significant sentiment shifts

### **Technical Analysis**
- **Technical Indicators**: RSI, MACD, Bollinger Bands, Moving Averages
- **Volatility Analysis**: Rolling volatility calculations and trend detection
- **Anomaly Detection**: Statistical outlier identification using Z-score and IQR methods
- **Performance Metrics**: Return calculations, Sharpe ratios, Value-at-Risk

### **Statistical Correlation Analysis**
- **Stock-Sentiment Correlations**: Statistical relationships between news sentiment and stock performance
- **Granger Causality Testing**: Determine if sentiment changes predict stock movements
- **Sector Impact Scoring**: Quantify how different sectors are affected by trade news
- **Cross-Sector Analysis**: Compare impact across different market sectors

### **Interactive Visualizations**
- **Interactive Charts**: Plotly-based dynamic visualizations
- **Multi-Tab Interface**: Organized analysis across Overview, Stock Analysis, News & Sentiment, Correlations, and Reports
- **Real-Time Updates**: Live data refresh with caching for performance
- **Export Capabilities**: Download data in CSV format for further analysis

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- API keys (optional but recommended for full functionality)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/stock_tariff_analysis.git
   cd stock_tariff_analysis
   ```

2. **Create a virtual environment and activate it**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Create a `.env` file in the project root with your API keys**
   ```env
   NEWS_API_KEY=your_newsapi_key_here
   ALPHA_VANTAGE_KEY=your_alpha_vantage_key_here
   ```

   Get API keys from:
   - NewsAPI: [https://newsapi.org/register](https://newsapi.org/register)
   - Alpha Vantage: [https://www.alphavantage.co/support/#api-key](https://www.alphavantage.co/support/#api-key)

5. **Run the application**
   ```bash
   streamlit run app.py
   ```

6. **Access the application**
   - Open your browser and navigate to `http://localhost:8501`
   - The application will start with sample data if API keys aren't configured

## 📖 Usage Guide

### **Configuration Panel**
Located in the sidebar, the configuration panel allows you to:
- **Select Sectors**: Choose which market sectors to analyze
- **Set Time Periods**: Configure analysis timeframe (1 month to 2 years)
- **Adjust News Period**: Set how far back to search for news (7-90 days)
- **Enable Advanced Features**: Toggle technical indicators, correlation analysis, statistical tests, and anomaly detection

### **Overview Tab**
- **Market Summary**: High-level performance metrics across selected sectors
- **Sector Performance Chart**: Visual comparison of sector returns
- **Recent News Highlights**: Latest tariff-related news with sentiment indicators
- **Key Metrics Display**: Summary statistics for stocks and sentiment

### **Stock Analysis Tab**
- **Interactive Price Charts**: Multi-stock price visualization with zoom and pan
- **Technical Analysis**: RSI, MACD, Bollinger Bands, and moving averages
- **Volatility Charts**: Price volatility trends and analysis
- **Anomaly Detection**: Identification of unusual price movements
- **Sector Metrics**: Detailed performance statistics by sector

### **News & Sentiment Tab**
- **Sentiment Timeline**: Historical sentiment trends over time
- **Sentiment Distribution**: Breakdown of positive, negative, and neutral sentiment
- **Keyword Analysis**: Sentiment by different tariff-related keywords
- **Significant Events**: Detection of major sentiment shifts
- **News Articles Table**: Complete list of analyzed articles with sentiment scores

### **Correlations Tab**
- **Correlation Heatmap**: Visual representation of stock-sentiment relationships
- **Statistical Analysis**: Strongest positive and negative correlations
- **Granger Causality Tests**: Determine predictive relationships
- **Sector Impact Scores**: Quantify how trade news affects different sectors

### **Reports Tab**
- **Comprehensive Reports**: Detailed analysis summaries
- **Performance Metrics**: Application performance statistics
- **Data Export**: Download stock data, news data, and sentiment analysis in CSV format

## Project Structure

```
stock_tariff_analysis/
├── app.py                 # Main Streamlit application
├── config.py             # Configuration settings and API keys
├── data_collector.py     # Stock and news data collection
├── sentiment_analyzer.py # News sentiment analysis engine
├── visualizations.py     # Interactive charts and plots
├── utils.py              # Utility functions and data processing
├── requirements.txt      # Python dependencies
└── README.md            # Project documentation
```

## 📦 Dependencies

### Core Libraries
- **streamlit**: Web application framework
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **yfinance**: Yahoo Finance data access
- **requests**: HTTP requests for APIs

### Visualization
- **plotly**: Interactive plotting library
- **seaborn**: Statistical data visualization
- **matplotlib**: Static plotting

### NLP & Sentiment Analysis
- **textblob**: Natural language processing
- **vaderSentiment**: Sentiment analysis specifically tuned for social media text
- **scikit-learn**: Machine learning utilities

### Statistical Analysis
- **scipy**: Scientific computing and statistics

## 🔑 API Configuration

### NewsAPI (Recommended)
- **Free Tier**: 1,000 requests/day, 30-day article history
- **Registration**: [https://newsapi.org/register](https://newsapi.org/register)
- **Usage**: Real-time news data for sentiment analysis

### Alpha Vantage (Optional)
- **Free Tier**: 5 API calls/minute, 500 calls/day
- **Registration**: [https://www.alphavantage.co/support/#api-key](https://www.alphavantage.co/support/#api-key)
- **Usage**: Alternative stock data source for validation

### Yahoo Finance (Default)
- **Access**: Free via yfinance library
- **No API Key Required**: Direct access to Yahoo Finance data
- **Usage**: Primary source for stock price data

## 🎯 Key Components

### Data Collection (`data_collector.py`)
- **StockDataCollector**: Primary class for fetching stock data
- **NewsDataCollector**: Retrieves tariff-related news articles
- **AlphaVantageCollector**: Alternative data source
- **Data Cleaning**: Validation and preprocessing utilities

### Sentiment Analysis (`sentiment_analyzer.py`)
- **Multi-Engine Analysis**: TextBlob, VADER, and custom algorithms
- **Financial Context**: Specialized sentiment analysis for financial news
- **Keyword Filtering**: Tariff and trade-specific terminology
- **Event Detection**: Identification of significant sentiment changes

### Visualizations (`visualizations.py`)
- **Interactive Charts**: Plotly-based dynamic visualizations
- **Multi-Panel Dashboards**: Comprehensive data presentation
- **Real-Time Updates**: Live chart updates with new data
- **Export Features**: Chart download capabilities

### Utilities (`utils.py`)
- **Data Processing**: Statistical calculations and transformations
- **Technical Indicators**: RSI, MACD, Bollinger Bands calculation
- **Correlation Analysis**: Statistical relationship detection
- **Performance Tracking**: Application performance monitoring

### Configuration (`config.py`)
- **API Management**: Centralized API key configuration
- **Stock Symbols**: Sector-organized stock listings
- **Keywords**: Tariff and trade-related search terms
- **Settings**: Application parameters and defaults

### Main Application (`app.py`)
- **Streamlit Interface**: Multi-tab dashboard layout
- **Data Integration**: Combines stock and news data streams
- **Interactive Controls**: User input and parameter settings
- **Real-Time Updates**: Live data refresh and visualization

## 📊 Data Sources

### Stock Data
- **Primary**: Yahoo Finance (via yfinance)
  - Free access to historical and real-time data
  - Comprehensive coverage of US and international markets
  - No API key required

- **Secondary**: Alpha Vantage
  - High-quality financial data
  - API key required
  - Used for validation and backup

### News Data
- **NewsAPI**: Current and historical news articles
  - Keyword-based search capabilities
  - Multiple source coverage
  - API key required for full access

- **Sample Data**: Built-in fallback for development
  - Representative examples
  - No external dependencies
  - Immediate functionality

## 🔧 Technical Features

### Performance Optimization
- **Data Caching**: Streamlit caching for improved performance
- **Efficient APIs**: Bulk data retrieval where possible
- **Progress Tracking**: User feedback for long operations
- **Error Handling**: Graceful failure recovery

### Statistical Analysis
- **Correlation Calculation**: Pearson correlation coefficients
- **Granger Causality**: Time series causality testing
- **Anomaly Detection**: Z-score and IQR-based outlier detection
- **Technical Indicators**: Comprehensive financial analysis tools

### Data Validation
- **Input Validation**: Robust data quality checks
- **Error Recovery**: Fallback mechanisms for API failures
- **Data Cleaning**: Automated preprocessing and validation
- **Missing Data Handling**: Intelligent gap-filling strategies

## 🚀 Advanced Usage

### Custom Analysis
```python
# Example: Custom sector analysis
from config import config
from data_collector import StockDataCollector

collector = StockDataCollector()
tech_stocks = config.STOCK_SYMBOLS['Technology']
data = collector.get_stock_data(tech_stocks, period='6mo')
```

### Extending Functionality
- **Add New Sectors**: Update `config.py` with additional stock symbols
- **Custom Keywords**: Modify tariff keyword list for specific analysis
- **Additional APIs**: Integrate new data sources in data collection modules
- **Custom Visualizations**: Extend visualization module with new chart types

