#!/usr/bin/env python3
"""
Test script for LLM Price Predictor
"""

import os
from llm_price_predictor import LLMPricePredictor
from datetime import datetime, timedelta

def test_llm_predictor():
    """Test the LLM predictor functionality"""
    
    # Check if API key is available
    api_key = os.getenv('OPENROUTER_API_KEY')
    if not api_key:
        print("❌ OPENROUTER_API_KEY environment variable not set")
        print("Please set your OpenRouter API key:")
        print("export OPENROUTER_API_KEY='your_api_key_here'")
        return False
    
    # Initialize predictor
    predictor = LLMPricePredictor(api_key=api_key)
    
    # Test stock data retrieval
    print("🔍 Testing stock data retrieval...")
    stock_data = predictor.get_stock_data_for_llm("RELIANCE")
    
    if stock_data:
        print(f"✅ Successfully retrieved data for {stock_data['symbol']}")
        print(f"   Current Price: ₹{stock_data['current_price']:.2f}")
        print(f"   RSI: {stock_data['rsi']:.2f}")
        print(f"   20-Day SMA: ₹{stock_data['sma_20']:.2f}")
    else:
        print("❌ Failed to retrieve stock data")
        return False
    
    # Test prediction
    print("\n🤖 Testing AI prediction...")
    future_date = (datetime.now() + timedelta(days=30)).strftime("%Y-%m-%d")
    
    result = predictor.predict_price_with_llm("RELIANCE", future_date)
    
    if result['success']:
        print("✅ AI prediction successful!")
        print(f"   Symbol: {result['symbol']}")
        print(f"   Target Date: {result['target_date']}")
        print(f"   Current Price: ₹{result['current_price']:.2f}")
        print(f"   Predicted Price: ₹{result['predicted_price']:.2f}")
        print(f"   Predicted Change: {result['price_change_predicted']:+.2f}%")
        print(f"   Confidence: {result['confidence_level']:.1f}%")
        print(f"   Trend: {result['trend_direction']}")
        print(f"   Reasoning: {result['reasoning'][:100]}...")
        return True
    else:
        print(f"❌ AI prediction failed: {result['error']}")
        if 'raw_response' in result:
            print(f"Raw response: {result['raw_response'][:200]}...")
        return False

if __name__ == "__main__":
    print("🧪 Testing LLM Price Predictor...")
    success = test_llm_predictor()
    
    if success:
        print("\n✅ All tests passed! The LLM predictor is working correctly.")
    else:
        print("\n❌ Some tests failed. Please check the configuration.") 