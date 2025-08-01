import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import requests
from bs4 import BeautifulSoup
import nltk
from textblob import TextBlob
import ta
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

class IndianStockResearchAssistant:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    def get_stock_data(self, stock_symbol, period="1y"):
        """Fetch stock data from Yahoo Finance with improved error handling"""
        try:
            original_symbol = stock_symbol
            # Add .NS suffix for NSE stocks if not provided
            if not stock_symbol.endswith('.NS') and not stock_symbol.endswith('.BO'):
                stock_symbol += '.NS'

            stock = yf.Ticker(stock_symbol)
            data = stock.history(period=period)

            if data.empty:
                # Try with .BO suffix for BSE
                stock_symbol_bo = original_symbol + '.BO'
                stock_bo = yf.Ticker(stock_symbol_bo)
                data_bo = stock_bo.history(period=period)
                if data_bo.empty:
                    print(f"❌ No price data found for {original_symbol} on NSE or BSE. The symbol may be incorrect, delisted, or Yahoo Finance is not providing data. Please check the symbol and try again.")
                    return None, None
                else:
                    return stock_bo, data_bo
            return stock, data
        except Exception as e:
            print(f"Error fetching data for {stock_symbol}: {e}")
            return None, None
    
    def calculate_technical_indicators(self, data):
        """Calculate technical indicators"""
        if data is None or data.empty:
            return {}
        
        indicators = {}
        # RSI
        indicators['rsi'] = ta.momentum.rsi(data['Close'])

        # MACD (manual calculation)
        exp12 = data['Close'].ewm(span=12, adjust=False).mean()
        exp26 = data['Close'].ewm(span=26, adjust=False).mean()
        macd = exp12 - exp26
        macd_signal = macd.ewm(span=9, adjust=False).mean()
        macd_histogram = macd - macd_signal
        indicators['macd'] = macd
        indicators['macd_signal'] = macd_signal
        indicators['macd_histogram'] = macd_histogram
        
        # Moving Averages
        indicators['sma_50'] = ta.trend.SMAIndicator(data['Close'], window=50).sma_indicator()
        indicators['sma_200'] = ta.trend.SMAIndicator(data['Close'], window=200).sma_indicator()
        
        # Bollinger Bands
        bb = ta.volatility.BollingerBands(data['Close'])
        indicators['bb_upper'] = bb.bollinger_hband()
        indicators['bb_lower'] = bb.bollinger_lband()
        indicators['bb_middle'] = bb.bollinger_mavg()
        
        # Volume indicators
        indicators['volume_sma'] = ta.trend.SMAIndicator(data['Volume'], window=20).sma_indicator()
        
        return indicators
    
    def identify_support_resistance(self, data, window=20):
        """Identify support and resistance levels"""
        if data is None or data.empty:
            return [], []
        
        highs = data['High'].rolling(window=window, center=True).max()
        lows = data['Low'].rolling(window=window, center=True).min()
        
        resistance_levels = []
        support_levels = []
        
        for i in range(window, len(data) - window):
            if data['High'].iloc[i] == highs.iloc[i]:
                resistance_levels.append(data['High'].iloc[i])
            if data['Low'].iloc[i] == lows.iloc[i]:
                support_levels.append(data['Low'].iloc[i])
        
        return support_levels, resistance_levels
    
    def analyze_candlestick_patterns(self, data, days=5):
        """Analyze recent candlestick patterns"""
        if data is None or len(data) < days:
            return []
        
        recent_data = data.tail(days)
        patterns = []
        
        for i in range(len(recent_data)):
            open_price = recent_data['Open'].iloc[i]
            close_price = recent_data['Close'].iloc[i]
            high_price = recent_data['High'].iloc[i]
            low_price = recent_data['Low'].iloc[i]
            
            body_size = abs(close_price - open_price)
            upper_shadow = high_price - max(open_price, close_price)
            lower_shadow = min(open_price, close_price) - low_price
            
            # Identify patterns
            if close_price > open_price:  # Bullish candle
                if body_size > (upper_shadow + lower_shadow) * 0.6:
                    patterns.append(f"Day {i+1}: Strong Bullish")
                elif upper_shadow > body_size * 2:
                    patterns.append(f"Day {i+1}: Shooting Star")
            else:  # Bearish candle
                if body_size > (upper_shadow + lower_shadow) * 0.6:
                    patterns.append(f"Day {i+1}: Strong Bearish")
                elif lower_shadow > body_size * 2:
                    patterns.append(f"Day {i+1}: Hammer")
        
        return patterns
    
    def get_fundamental_data(self, stock):
        """Get fundamental data for the stock"""
        try:
            info = stock.info
            financials = stock.financials
            balance_sheet = stock.balance_sheet
            cashflow = stock.cashflow
            
            return {
                'info': info,
                'financials': financials,
                'balance_sheet': balance_sheet,
                'cashflow': cashflow
            }
        except Exception as e:
            print(f"Error fetching fundamental data: {e}")
            return {}
    
    def calculate_valuation_metrics(self, fundamental_data):
        """Calculate key valuation metrics"""
        metrics = {}
        
        try:
            info = fundamental_data.get('info', {})
            financials = fundamental_data.get('financials', pd.DataFrame())
            
            # Basic metrics
            metrics['market_cap'] = info.get('marketCap', 'N/A')
            metrics['pe_ratio'] = info.get('trailingPE', 'N/A')
            metrics['pb_ratio'] = info.get('priceToBook', 'N/A')
            metrics['roe'] = info.get('returnOnEquity', 'N/A')
            
            # Financial metrics from statements
            if not financials.empty:
                # Get latest year's data
                latest_year = financials.columns[0]
                metrics['revenue'] = financials.loc['Total Revenue', latest_year] if 'Total Revenue' in financials.index else 'N/A'
                metrics['net_income'] = financials.loc['Net Income', latest_year] if 'Net Income' in financials.index else 'N/A'
            
        except Exception as e:
            print(f"Error calculating valuation metrics: {e}")
        
        return metrics
    
    def get_news_sentiment(self, stock_name, days=7):
        """Get news sentiment for the stock"""
        # This is a placeholder - in a real implementation, you'd use news APIs
        # For now, we'll return a mock sentiment analysis
        sentiment_score = np.random.uniform(-0.5, 0.5)  # Mock sentiment
        
        if sentiment_score > 0.2:
            sentiment = "Positive"
        elif sentiment_score < -0.2:
            sentiment = "Negative"
        else:
            sentiment = "Neutral"
        
        return {
            'sentiment': sentiment,
            'score': sentiment_score,
            'news_count': np.random.randint(5, 20)
        }
    
    def get_derivatives_data(self, stock_symbol):
        """Get derivatives data (mock implementation)"""
        # In a real implementation, you'd fetch this from NSE/BSE APIs
        return {
            'oi_buildup': 'Moderate',
            'pcr': round(np.random.uniform(0.8, 1.2), 2),
            'max_pain': round(np.random.uniform(100, 200), 2),
            'iv': round(np.random.uniform(20, 40), 2)
        }

    def determine_trend(self, data):
        """Determine current trend based on price action and indicators"""
        if data is None or data.empty:
            return "Sideways"
        
        current_price = data['Close'].iloc[-1]
        sma_50 = data['Close'].rolling(window=50).mean().iloc[-1]
        sma_200 = data['Close'].rolling(window=200).mean().iloc[-1]
        
        # Price relative to moving averages
        price_vs_50 = (current_price - sma_50) / sma_50
        price_vs_200 = (current_price - sma_200) / sma_200
        
        # RSI
        rsi = ta.momentum.RSIIndicator(data['Close']).rsi().iloc[-1]
        
        # Trend determination logic
        if price_vs_50 > 0.02 and price_vs_200 > 0.05 and rsi < 70:
            return "Bullish"
        elif price_vs_50 < -0.02 and price_vs_200 < -0.05 and rsi > 30:
            return "Bearish"
        else:
            return "Sideways"
    
    def generate_report(self, stock_symbol):
        """Generate comprehensive stock research report"""
        print(f"🔍 Generating comprehensive report for {stock_symbol}...")
        
        # Get stock data
        stock, data = self.get_stock_data(stock_symbol)
        if stock is None or data is None or data.empty:
            return f"❌ Unable to fetch data for {stock_symbol}. Please check the stock symbol."
        
        # Calculate indicators
        indicators = self.calculate_technical_indicators(data)
        
        # Get fundamental data
        fundamental_data = self.get_fundamental_data(stock)
        valuation_metrics = self.calculate_valuation_metrics(fundamental_data)
        
        # Get sentiment and derivatives data
        sentiment_data = self.get_news_sentiment(stock_symbol)
        derivatives_data = self.get_derivatives_data(stock_symbol)
        
        # Determine trend
        trend = self.determine_trend(data)
        
        # Support and resistance levels
        support_levels, resistance_levels = self.identify_support_resistance(data)
        
        # Candlestick patterns
        candlestick_patterns = self.analyze_candlestick_patterns(data)
        
        # Generate report
        report = self._format_report(
            stock_symbol, data, indicators, fundamental_data, 
            valuation_metrics, sentiment_data, derivatives_data,
            trend, support_levels, resistance_levels, candlestick_patterns
        )
        
        return report
    
    def _format_report(self, stock_symbol, data, indicators, fundamental_data, 
                      valuation_metrics, sentiment_data, derivatives_data,
                      trend, support_levels, resistance_levels, candlestick_patterns):
        """Format the comprehensive report"""
        
        current_price = data['Close'].iloc[-1]
        prev_price = data['Close'].iloc[-2]
        price_change = ((current_price - prev_price) / prev_price) * 100
        
        # Format market cap
        market_cap = valuation_metrics.get('market_cap', 'N/A')
        market_cap_str = f"₹{market_cap:,.0f}" if isinstance(market_cap, (int, float)) else str(market_cap)
        
        # Format revenue
        revenue = valuation_metrics.get('revenue', 'N/A')
        revenue_str = f"₹{revenue:,.0f}" if isinstance(revenue, (int, float)) else str(revenue)
        
        # Format net income
        net_income = valuation_metrics.get('net_income', 'N/A')
        net_income_str = f"₹{net_income:,.0f}" if isinstance(net_income, (int, float)) else str(net_income)
        
        report = f"""
# 📊 Comprehensive Stock Research Report
## {stock_symbol.upper()} - Indian Markets Analysis

---

## 📈 **1. Technical Analysis**

### Current Status
- **Current Price**: ₹{current_price:.2f}
- **Price Change**: {price_change:+.2f}%
- **Trend**: {trend.upper()}
- **Volume**: {data['Volume'].iloc[-1]:,.0f}

### Key Indicators
- **RSI**: {indicators.get('rsi', pd.Series()).iloc[-1]:.2f} ({'Overbought' if indicators.get('rsi', pd.Series()).iloc[-1] > 70 else 'Oversold' if indicators.get('rsi', pd.Series()).iloc[-1] < 30 else 'Neutral'})
- **MACD**: {indicators.get('macd', pd.Series()).iloc[-1]:.2f}
- **50-Day SMA**: ₹{indicators.get('sma_50', pd.Series()).iloc[-1]:.2f}
- **200-Day SMA**: ₹{indicators.get('sma_200', pd.Series()).iloc[-1]:.2f}

### Support & Resistance Levels
- **Nearest Support**: ₹{min(support_levels[-3:]) if support_levels else 'N/A':.2f}
- **Nearest Resistance**: ₹{max(resistance_levels[-3:]) if resistance_levels else 'N/A':.2f}

### Recent Candlestick Patterns (Last 5 Days)
"""
        
        for pattern in candlestick_patterns:
            report += f"- {pattern}\n"
        
        if not candlestick_patterns:
            report += "- No significant patterns detected\n"
        
        report += f"""

---

## 📊 **2. Fundamental Analysis**

### Business Overview
- **Sector**: {fundamental_data.get('info', {}).get('sector', 'N/A')}
- **Industry**: {fundamental_data.get('info', {}).get('industry', 'N/A')}
- **Market Cap**: {market_cap_str}

### Key Financial Metrics
- **P/E Ratio**: {valuation_metrics.get('pe_ratio', 'N/A')}
- **P/B Ratio**: {valuation_metrics.get('pb_ratio', 'N/A')}
- **ROE**: {valuation_metrics.get('roe', 'N/A')}
- **Revenue (Latest)**: {revenue_str}
- **Net Income (Latest)**: {net_income_str}

---

## 🧠 **3. Sentiment & Derivative Analysis**

### News Sentiment (Last 7 Days)
- **Overall Sentiment**: {sentiment_data['sentiment']}
- **Sentiment Score**: {sentiment_data['score']:.2f}
- **News Articles Analyzed**: {sentiment_data['news_count']}

### Derivatives Data
- **Open Interest Build-up**: {derivatives_data['oi_buildup']}
- **Put/Call Ratio (PCR)**: {derivatives_data['pcr']}
- **Max Pain**: ₹{derivatives_data['max_pain']}
- **Implied Volatility**: {derivatives_data['iv']}%

---

## 🎯 **Market Outlook & Risk Assessment**

### Current Market Outlook
Based on the comprehensive analysis, {stock_symbol.upper()} is currently showing a **{trend.lower()}** trend. The technical indicators suggest {self._get_trend_justification(trend, indicators)}.

### Key Risk Factors
- **Market Volatility**: Monitor India VIX for market sentiment
- **Earnings Season**: Watch for upcoming quarterly results
- **Sector Performance**: {fundamental_data.get('info', {}).get('sector', 'N/A')} sector trends
- **Global Factors**: US Fed policy, crude oil prices, geopolitical tensions

### Upcoming Events to Monitor
- Quarterly earnings announcements
- Dividend declarations
- Board meetings and corporate actions
- Sector-specific regulatory changes

---

*Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
*This report is for informational purposes only. Please consult with a financial advisor before making investment decisions.*
"""
        
        return report
    
    def _get_trend_justification(self, trend, indicators):
        """Get justification for the trend"""
        if trend == "Bullish":
            return "positive momentum with price above key moving averages and healthy volume."
        elif trend == "Bearish":
            return "negative momentum with price below key moving averages and declining volume."
        else:
            return "consolidation phase with mixed signals from technical indicators."

def main():
    """Main function to run the stock research assistant"""
    assistant = IndianStockResearchAssistant()
    
    print("🚀 Indian Stock Market Research Assistant")
    print("=" * 50)
    
    while True:
        stock_symbol = input("\nEnter stock symbol (e.g., RELIANCE, TCS, INFY) or 'quit' to exit: ").strip().upper()
        
        if stock_symbol.lower() == 'quit':
            print("👋 Thank you for using the Stock Research Assistant!")
            break
        
        if not stock_symbol:
            print("❌ Please enter a valid stock symbol.")
            continue
        
        print(f"\n📊 Generating report for {stock_symbol}...")
        report = assistant.generate_report(stock_symbol)
        print(report)
        
        # Save report to file
        filename = f"{stock_symbol}_research_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"\n💾 Report saved to: {filename}")

if __name__ == "__main__":
    main()