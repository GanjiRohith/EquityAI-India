import requests
import json
import pandas as pd
from datetime import datetime, timedelta
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

class LLMPricePredictor:
    def __init__(self, api_key=None):
        """
        Initialize the LLM Price Predictor with OpenRouter API
        
        Args:
            api_key (str): OpenRouter API key. If None, will try to get from environment
        """
        self.api_key = api_key
        self.base_url = "https://openrouter.ai/api/v1"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://stock-research-assistant.streamlit.app",
            "X-Title": "Stock Research Assistant"
        }
    
    def get_stock_data_for_llm(self, symbol, period="1y"):
        """
        Get stock data formatted for LLM analysis
        
        Args:
            symbol (str): Stock symbol
            period (str): Data period
            
        Returns:
            dict: Formatted stock data
        """
        try:
            # Add .NS suffix for Indian stocks if not present
            if not symbol.endswith('.NS') and not symbol.endswith('.BO'):
                symbol = f"{symbol}.NS"
            
            stock = yf.Ticker(symbol)
            data = stock.history(period=period)
            
            if data.empty:
                return None
            
            # Calculate basic technical indicators
            data['SMA_20'] = data['Close'].rolling(window=20).mean()
            data['SMA_50'] = data['Close'].rolling(window=50).mean()
            data['RSI'] = self._calculate_rsi(data['Close'])
            data['Volume_MA'] = data['Volume'].rolling(window=20).mean()
            
            # Get recent data for LLM
            recent_data = data.tail(30)  # Last 30 days
            
            return {
                'symbol': symbol.replace('.NS', '').replace('.BO', ''),
                'current_price': float(data['Close'].iloc[-1]),
                'price_change_1d': float(((data['Close'].iloc[-1] - data['Close'].iloc[-2]) / data['Close'].iloc[-2]) * 100),
                'price_change_1w': float(((data['Close'].iloc[-1] - data['Close'].iloc[-8]) / data['Close'].iloc[-8]) * 100),
                'price_change_1m': float(((data['Close'].iloc[-1] - data['Close'].iloc[-30]) / data['Close'].iloc[-30]) * 100),
                'volume': int(data['Volume'].iloc[-1]),
                'avg_volume': int(data['Volume'].rolling(window=20).mean().iloc[-1]),
                'sma_20': float(data['SMA_20'].iloc[-1]),
                'sma_50': float(data['SMA_50'].iloc[-1]),
                'rsi': float(data['RSI'].iloc[-1]),
                'high_52w': float(data['High'].max()),
                'low_52w': float(data['Low'].min()),
                'recent_prices': recent_data['Close'].tolist(),
                'recent_volumes': recent_data['Volume'].tolist(),
                'recent_dates': recent_data.index.strftime('%Y-%m-%d').tolist()
            }
        except Exception as e:
            print(f"Error getting stock data: {e}")
            return None
    
    def _calculate_rsi(self, prices, period=14):
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def predict_price_with_llm(self, symbol, target_date, api_key=None):
        """
        Predict stock price using LLM via OpenRouter API
        
        Args:
            symbol (str): Stock symbol
            target_date (str): Target date in YYYY-MM-DD format
            api_key (str): OpenRouter API key (optional, uses instance key if not provided)
            
        Returns:
            dict: Prediction results
        """
        if api_key:
            self.api_key = api_key
            self.headers["Authorization"] = f"Bearer {api_key}"
        
        if not self.api_key:
            return {
                'success': False,
                'error': 'OpenRouter API key is required. Please provide your API key.'
            }
        
        # Get stock data
        stock_data = self.get_stock_data_for_llm(symbol)
        if not stock_data:
            return {
                'success': False,
                'error': f'Unable to fetch data for {symbol}'
            }
        
        # Calculate days to target date
        try:
            target_dt = datetime.strptime(target_date, "%Y-%m-%d")
            current_dt = datetime.now()
            days_ahead = (target_dt - current_dt).days
            
            if days_ahead <= 0:
                return {
                    'success': False,
                    'error': 'Target date must be in the future'
                }
        except ValueError:
            return {
                'success': False,
                'error': 'Invalid date format. Use YYYY-MM-DD'
            }
        
        # Prepare prompt for LLM
        prompt = self._create_prediction_prompt(stock_data, target_date, days_ahead)
        
        # Call OpenRouter API
        try:
            response = self._call_openrouter_api(prompt)
            return self._parse_llm_response(response, stock_data, target_date)
        except Exception as e:
            return {
                'success': False,
                'error': f'Error calling LLM API: {str(e)}'
            }
    
    def _create_prediction_prompt(self, stock_data, target_date, days_ahead):
        """Create a detailed prompt for the LLM"""
        
        prompt = f"""You are an expert financial analyst specializing in Indian stock market analysis. 

STOCK DATA FOR {stock_data['symbol'].upper()}:
- Current Price: ₹{stock_data['current_price']:.2f}
- 1 Day Change: {stock_data['price_change_1d']:+.2f}%
- 1 Week Change: {stock_data['price_change_1w']:+.2f}%
- 1 Month Change: {stock_data['price_change_1m']:+.2f}%
- Current Volume: {stock_data['volume']:,}
- Average Volume: {stock_data['avg_volume']:,}
- 20-Day SMA: ₹{stock_data['sma_20']:.2f}
- 50-Day SMA: ₹{stock_data['sma_50']:.2f}
- Current RSI: {stock_data['rsi']:.2f}
- 52-Week High: ₹{stock_data['high_52w']:.2f}
- 52-Week Low: ₹{stock_data['low_52w']:.2f}

Recent Price Trend (Last 30 days):
{', '.join([f"₹{price:.2f}" for price in stock_data['recent_prices'][-10:]])}

TASK: Predict the closing price for {stock_data['symbol'].upper()} on {target_date} ({days_ahead} days from now).

ANALYSIS REQUIREMENTS:
1. Consider technical indicators (RSI, moving averages, volume trends)
2. Analyze recent price momentum and volatility
3. Consider market sentiment and seasonal factors
4. Account for the time horizon ({days_ahead} days ahead)
5. Provide a realistic prediction with reasoning

RESPONSE FORMAT (JSON only):
{{
    "predicted_price": <predicted_price_in_rupees>,
    "confidence_level": <confidence_percentage_0_100>,
    "reasoning": "<detailed_reasoning_for_prediction>",
    "risk_factors": ["<risk_factor_1>", "<risk_factor_2>", "<risk_factor_3>"],
    "support_levels": [<support_price_1>, <support_price_2>],
    "resistance_levels": [<resistance_price_1>, <resistance_price_2>],
    "trend_direction": "<bullish/bearish/sideways>"
}}

Provide only the JSON response, no additional text."""

        return prompt
    
    def _call_openrouter_api(self, prompt):
        """Call OpenRouter API with the prompt"""
        
        payload = {
            "model": "anthropic/claude-3.5-sonnet",  # Using Claude for financial analysis
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.3,  # Lower temperature for more consistent financial analysis
            "max_tokens": 1000
        }
        
        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers=self.headers,
            json=payload,
            timeout=30
        )
        
        if response.status_code != 200:
            raise Exception(f"API call failed with status {response.status_code}: {response.text}")
        
        return response.json()
    
    def _parse_llm_response(self, api_response, stock_data, target_date):
        """Parse the LLM response and format the prediction"""
        
        try:
            # Extract the content from the API response
            content = api_response['choices'][0]['message']['content']
            
            # Try to extract JSON from the response
            import re
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                prediction_data = json.loads(json_str)
            else:
                # If no JSON found, try to parse the entire content
                prediction_data = json.loads(content)
            
            # Validate and format the response
            result = {
                'success': True,
                'symbol': stock_data['symbol'],
                'target_date': target_date,
                'current_price': stock_data['current_price'],
                'predicted_price': float(prediction_data.get('predicted_price', 0)),
                'confidence_level': float(prediction_data.get('confidence_level', 0)),
                'reasoning': prediction_data.get('reasoning', 'No reasoning provided'),
                'risk_factors': prediction_data.get('risk_factors', []),
                'support_levels': prediction_data.get('support_levels', []),
                'resistance_levels': prediction_data.get('resistance_levels', []),
                'trend_direction': prediction_data.get('trend_direction', 'unknown'),
                'price_change_predicted': 0,
                'llm_response': content
            }
            
            # Calculate predicted price change
            if result['predicted_price'] > 0:
                result['price_change_predicted'] = ((result['predicted_price'] - result['current_price']) / result['current_price']) * 100
            
            return result
            
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            return {
                'success': False,
                'error': f'Failed to parse LLM response: {str(e)}',
                'raw_response': api_response.get('choices', [{}])[0].get('message', {}).get('content', '')
            }
    
    def get_market_context(self, symbol):
        """Get additional market context for better predictions"""
        try:
            # Get broader market data (NIFTY 50)
            nifty = yf.Ticker("^NSEI")
            nifty_data = nifty.history(period="1mo")
            
            if not nifty_data.empty:
                nifty_current = nifty_data['Close'].iloc[-1]
                nifty_change = ((nifty_current - nifty_data['Close'].iloc[-2]) / nifty_data['Close'].iloc[-2]) * 100
                
                return {
                    'nifty_current': float(nifty_current),
                    'nifty_change': float(nifty_change),
                    'market_trend': 'bullish' if nifty_change > 0 else 'bearish'
                }
        except:
            pass
        
        return None

# Example usage
if __name__ == "__main__":
    # Initialize predictor (you'll need to provide your OpenRouter API key)
    predictor = LLMPricePredictor(api_key="your_openrouter_api_key_here")
    
    # Example prediction
    result = predictor.predict_price_with_llm("RELIANCE", "2024-02-15")
    print(json.dumps(result, indent=2)) 