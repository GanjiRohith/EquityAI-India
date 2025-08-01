# 📊 Indian Stock Market Research Assistant

A comprehensive stock market research tool for Indian markets (NSE/BSE) providing technical, fundamental, sentiment, and predictive analysis with interactive charts and downloadable reports.

## 🚀 Features

### 📈 Technical Analysis
- **Price Action**: Current price, trends, and volume analysis
- **Technical Indicators**: RSI, MACD (manual calculation), 50/200 Day Moving Averages, Bollinger Bands
- **Support & Resistance**: Automated level identification
- **Candlestick Patterns**: Recent pattern recognition (last 5 days)
- **Interactive Charts**: Plotly-based interactive visualizations

### 📊 Fundamental Analysis
- **Business Overview**: Sector, industry, and market cap
- **Financial Metrics**: P/E Ratio, P/B Ratio, ROE, Revenue, Net Income
- **Valuation**: Comparison with industry peers
- **Financial Statements**: Income statement, balance sheet, cash flow analysis

### 🧠 Sentiment & Derivatives Analysis
- **News Sentiment**: Market sentiment analysis (last 7 days)
- **Derivatives Data**: Options analysis including OI build-up, PCR, Max Pain, IV
- **Market Positioning**: FII/DII activity insights
- **Social Media Buzz**: Market sentiment indicators

### 🔮 Predictive Analysis
- **Future Price Prediction**: Predict closing price for a future date using Prophet (if installed) or ARIMA (statsmodels)
- **Forecast Visualization**: See actual vs. predicted price trends on interactive charts
- **🤖 AI-Powered Predictions**: LLM-based price predictions using OpenRouter API with detailed reasoning and risk analysis

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python package installer)

### Setup Instructions

1. **Clone or download the project files**
   ```bash
   git clone <repository-url>
   cd stock-research-assistant
   ```

2. **Install required dependencies**
   ```bash
   pip install -r requirements.txt
   ```
   This will install all required packages, including Prophet and statsmodels for forecasting.

3. **Run the application**

   **Option A: Web Interface (Recommended)**
   ```bash
   streamlit run streamlit_app.py
   ```
   This will open a web interface in your browser at `http://localhost:8501`

   **Option B: Command Line Interface**
   ```bash
   python stock_research_assistant.py
   ```

## 🔐 Setting up OpenRouter API Key

This project uses [OpenRouter](https://openrouter.ai/) to access LLM APIs.

### Steps to get your API key:

1. Visit [https://openrouter.ai/](https://openrouter.ai/)
2. Log in with your email or GitHub account.
3. Go to **API Keys** from your profile menu.
4. Click **Create Key**, copy it, and store it safely.

## 📖 Usage Guide

### Web Interface (Streamlit)

1. **Launch the application**
   - Run `streamlit run streamlit_app.py`
   - Open your browser to the provided URL

2. **Enter stock symbol**
   - Use the sidebar to enter a stock symbol (e.g., RELIANCE, TCS, INFY)
   - Select the analysis period (1mo, 3mo, 6mo, 1y, 2y, 5y)
   - Click "Generate Analysis"

3. **Predict future price**
   - Enter a future date in the sidebar and click "Predict Closing Price"
   - The app will use Prophet (if available) or ARIMA to forecast the closing price and show a chart
   - For AI predictions: Enter your OpenRouter API key and click "AI Predict Price" for LLM-based analysis

4. **Explore the analysis**
   - **Technical Analysis Tab**: View indicators, support/resistance, patterns
   - **Fundamentals Tab**: Financial metrics and business overview
   - **Sentiment Tab**: News sentiment and derivatives data
   - **Full Report Tab**: Complete research report with download option
   - **Charts Tab**: Interactive price charts and technical analysis

### Command Line Interface

1. **Run the application**
   ```bash
   python stock_research_assistant.py
   ```

2. **Enter stock symbol when prompted**
   - Type the stock symbol (e.g., RELIANCE, TCS, INFY)
   - Press Enter to generate the report
   - Type 'quit' to exit

3. **Review the generated report**
   - The report will be displayed in the terminal
   - A text file will be saved with the complete analysis

## 📦 Project Structure

```
stock-research-assistant/
├── stock_research_assistant.py    # Main CLI application and analysis logic
├── streamlit_app.py               # Web interface
├── requirements.txt               # Python dependencies
├── README.md                      # This file
├── demo.py                        # Demo script for sample analysis
└── sample_reports/                # Generated reports (created automatically)
```

## 🔧 Technical Details

### Data Sources
- **Stock Data**: Yahoo Finance API (yfinance)
- **Technical Indicators**: ta library (not TA-Lib)
- **Charts**: Plotly for interactive visualizations
- **Web Interface**: Streamlit framework

### Key Libraries Used
- `yfinance`: Stock data fetching
- `pandas`: Data manipulation and analysis
- `numpy`: Numerical computations
- `plotly`: Interactive charts
- `streamlit`: Web interface
- `ta`: Technical analysis indicators (not TA-Lib)
- `textblob`: Sentiment analysis
- `prophet`: Time series forecasting for price prediction
- `statsmodels`: ARIMA time series modeling for price prediction
- `requests`: HTTP requests for OpenRouter API integration

## 📝 Sample Output

### Web App (Streamlit)
- **Analysis Tabs**: See technical, fundamental, sentiment, and full report tabs for any Indian stock
- **Prediction Output**:
  - "Predicted closing price: ₹2,850.23"
  - Chart showing actual vs. forecasted price, with the predicted value highlighted
- **Downloadable Report**: Markdown file with all analysis details

### CLI Output
```
📊 Generating report for RELIANCE...
# 📊 Comprehensive Stock Research Report
## RELIANCE - Indian Markets Analysis
...
### Key Indicators
- **RSI**: 54.23 (Neutral)
- **MACD**: 12.34
- **50-Day SMA**: ₹2,800.12
- **200-Day SMA**: ₹2,750.45
...
### News Sentiment (Last 7 Days)
- **Overall Sentiment**: Positive
- **Sentiment Score**: 0.32
- **News Articles Analyzed**: 12
...
### Derivatives Data
- **Open Interest Build-up**: Moderate
- **Put/Call Ratio (PCR)**: 1.05
- **Max Pain**: ₹2,900
- **Implied Volatility**: 22.5%
...
*Report generated on 2024-07-14 20:00:00*
```

## ⚠️ Important Notes

### AI Prediction Setup
To use the AI-powered price prediction feature:
1. Get a free API key from [OpenRouter](https://openrouter.ai/)
2. Enter your API key in the Streamlit sidebar
3. Provide a stock symbol and future date
4. Click "AI Predict Price" for LLM-based analysis

### Disclaimer
- This tool is for **educational and research purposes only**
- **Not financial advice**: Always consult with a qualified financial advisor
- **Data accuracy**: While we strive for accuracy, data may have delays or errors
- **Market risks**: Stock markets are inherently risky; past performance doesn't guarantee future results
- **AI predictions**: LLM-based predictions are experimental and should not be used as sole basis for investment decisions

### Limitations
- **Data availability**: Some stocks may have limited data
- **Real-time updates**: Data may have slight delays
- **API limitations**: Yahoo Finance API has rate limits
- **Indian markets focus**: Optimized for NSE/BSE stocks

## 🤝 Contributing

Contributions are welcome! Please feel free to:
- Report bugs or issues
- Suggest new features
- Submit pull requests
- Improve documentation

## 📞 Support

If you encounter any issues:
1. Check the console for error messages
2. Verify your internet connection
3. Ensure all dependencies are installed
4. Try with a different stock symbol

## 📄 License

This project is open source and available under the MIT License.

---

**Happy Trading! 📈** 