import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import sys
import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=UserWarning, module='statsmodels')

# Try to import Prophet, fallback to ARIMA if not available
PROPHET_AVAILABLE = False
ARIMA_AVAILABLE = False

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    pass

try:
    from statsmodels.tsa.arima.model import ARIMA
    ARIMA_AVAILABLE = True
except ImportError:
    pass

# Import the stock research assistant
from stock_research_assistant import IndianStockResearchAssistant

def main():
    print("\n=== Stock Price Predictor ===\n")
    stock_symbol = input("Enter stock symbol (e.g., RELIANCE, TCS, INFY): ").strip().upper()
    future_date_str = input("Enter future date to predict closing price (YYYY-MM-DD): ").strip()

    # Validate future date
    try:
        future_date = datetime.strptime(future_date_str, "%Y-%m-%d")
        if future_date <= datetime.now():
            print("❌ Please enter a future date.")
            return
    except Exception as e:
        print(f"❌ Invalid date format: {e}")
        return

    # Load historical data
    assistant = IndianStockResearchAssistant()
    stock, data = assistant.get_stock_data(stock_symbol, period="5y")
    if data is None or data.empty:
        print(f"❌ Unable to fetch data for {stock_symbol}.")
        return

    # Prepare data
    df = data.reset_index()
    if 'Date' not in df.columns:
        # yfinance may use 'index' as date
        df.rename(columns={df.columns[0]: 'Date'}, inplace=True)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')
    df = df[['Date', 'Close']].dropna()

    # Prophet expects columns: ds, y
    prophet_df = df.rename(columns={'Date': 'ds', 'Close': 'y'})
    # Ensure 'ds' is timezone-naive
    prophet_df['ds'] = prophet_df['ds'].dt.tz_localize(None)

    # Train and predict
    if PROPHET_AVAILABLE:
        print("Using Prophet for forecasting...")
        model = Prophet()
        model.fit(prophet_df)
        # Create future dataframe
        last_date = prophet_df['ds'].max()
        days_ahead = (future_date - last_date).days
        if days_ahead <= 0:
            print("❌ Future date must be after the last available date in the data.")
            return
        future = model.make_future_dataframe(periods=days_ahead)
        forecast = model.predict(future)
        # Get prediction for the requested date
        pred_row = forecast[forecast['ds'] == pd.to_datetime(future_date_str)]
        if pred_row.empty:
            print("❌ Unable to predict for the given date. Try a different date.")
            return
        predicted_price = pred_row['yhat'].values[0]
        # Plot
        plt.figure(figsize=(12,6))
        plt.plot(df['Date'], df['Close'], label='Actual Close')
        plt.plot(forecast['ds'], forecast['yhat'], label='Forecasted Trend', linestyle='--')
        plt.scatter([future_date], [predicted_price], color='red', label=f'Predicted {future_date_str}: ₹{predicted_price:.2f}', zorder=5)
        plt.title(f"{stock_symbol} Closing Price Prediction")
        plt.xlabel("Date")
        plt.ylabel("Close Price (INR)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        print(f"\nPredicted closing price for {stock_symbol} on {future_date_str}: ₹{predicted_price:.2f}")
    elif ARIMA_AVAILABLE:
        print("Prophet not available. Using ARIMA for forecasting...")
        # Use ARIMA on the close price series
        ts = df.set_index('Date')['Close']
        # Ensure the time series has proper frequency information
        ts = ts.asfreq('B')  # Business day frequency
        ts = ts.fillna(method='ffill')  # Forward fill any missing values
        # Simple ARIMA order selection (can be improved)
        order = (5,1,0)
        model = ARIMA(ts, order=order)
        model_fit = model.fit()
        # Forecast steps ahead
        last_date = ts.index.max()
        days_ahead = (future_date - last_date).days
        if days_ahead <= 0:
            print("❌ Future date must be after the last available date in the data.")
            return
        forecast = model_fit.get_forecast(steps=days_ahead)
        forecast_index = pd.date_range(start=last_date + timedelta(days=1), periods=days_ahead, freq='B')
        forecast_series = pd.Series(forecast.predicted_mean.values, index=forecast_index)
        # Get prediction for the requested date
        if pd.to_datetime(future_date_str) not in forecast_series.index:
            print("❌ Unable to predict for the given date (may not be a business day). Try a different date.")
            return
        predicted_price = forecast_series[pd.to_datetime(future_date_str)]
        # Plot
        plt.figure(figsize=(12,6))
        plt.plot(df['Date'], df['Close'], label='Actual Close')
        plt.plot(forecast_series.index, forecast_series.values, label='Forecasted Trend', linestyle='--')
        plt.scatter([future_date], [predicted_price], color='red', label=f'Predicted {future_date_str}: ₹{predicted_price:.2f}', zorder=5)
        plt.title(f"{stock_symbol} Closing Price Prediction")
        plt.xlabel("Date")
        plt.ylabel("Close Price (INR)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        print(f"\nPredicted closing price for {stock_symbol} on {future_date_str}: ₹{predicted_price:.2f}")
    else:
        print("❌ Neither Prophet nor ARIMA is available. Please install one of them:")
        print("   pip install prophet")
        print("   pip install statsmodels")

if __name__ == "__main__":
    main() 