import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=UserWarning, module='statsmodels')

# Import our stock research assistant
from stock_research_assistant import IndianStockResearchAssistant
from llm_price_predictor import LLMPricePredictor

# Page configuration
st.set_page_config(
    page_title="Indian Stock Market Research Assistant",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .trend-bullish {
        color: #28a745;
        font-weight: bold;
    }
    .trend-bearish {
        color: #dc3545;
        font-weight: bold;
    }
    .trend-sideways {
        color: #ffc107;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

def create_price_chart(data, indicators):
    """Create interactive price chart with indicators"""
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=('Price & Moving Averages', 'Volume', 'RSI'),
        row_heights=[0.6, 0.2, 0.2]
    )
    
    # Candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=data.index,
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            name='Price'
        ),
        row=1, col=1
    )
    
    # Moving averages
    if 'sma_50' in indicators and not indicators['sma_50'].isna().all():
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=indicators['sma_50'],
                name='50 SMA',
                line=dict(color='orange', width=1)
            ),
            row=1, col=1
        )
    
    if 'sma_200' in indicators and not indicators['sma_200'].isna().all():
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=indicators['sma_200'],
                name='200 SMA',
                line=dict(color='red', width=1)
            ),
            row=1, col=1
        )
    
    # Bollinger Bands
    if 'bb_upper' in indicators and not indicators['bb_upper'].isna().all():
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=indicators['bb_upper'],
                name='BB Upper',
                line=dict(color='gray', width=1, dash='dash'),
                showlegend=False
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=indicators['bb_lower'],
                name='BB Lower',
                line=dict(color='gray', width=1, dash='dash'),
                fill='tonexty',
                fillcolor='rgba(128,128,128,0.1)',
                showlegend=False
            ),
            row=1, col=1
        )
    
    # Volume
    colors = ['red' if close < open else 'green' for close, open in zip(data['Close'], data['Open'])]
    fig.add_trace(
        go.Bar(
            x=data.index,
            y=data['Volume'],
            name='Volume',
            marker_color=colors,
            opacity=0.7
        ),
        row=2, col=1
    )
    
    # RSI
    if 'rsi' in indicators and not indicators['rsi'].isna().all():
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=indicators['rsi'],
                name='RSI',
                line=dict(color='purple', width=2)
            ),
            row=3, col=1
        )
        
        # Add RSI overbought/oversold lines
        # Plotly's add_hline does not support row/col; to restrict to the RSI subplot, set yaxis="y3" in add_shape
        fig.add_shape(type="line", x0=data.index.min(), x1=data.index.max(), y0=70, y1=70, line=dict(dash="dash", color="red"), xref="x", yref="y3")
        fig.add_shape(type="line", x0=data.index.min(), x1=data.index.max(), y0=30, y1=30, line=dict(dash="dash", color="green"), xref="x", yref="y3")
    
    fig.update_layout(
        title='Technical Analysis Chart',
        xaxis_rangeslider_visible=False,
        height=800
    )
    
    return fig

def create_macd_chart(data, indicators):
    """Create MACD chart"""
    if 'macd' not in indicators or indicators['macd'].isna().all():
        return None
    
    fig = go.Figure()
    
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=indicators['macd'],
            name='MACD',
            line=dict(color='blue', width=2)
        )
    )
    
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=indicators['macd_signal'],
            name='Signal',
            line=dict(color='red', width=2)
        )
    )
    
    # MACD histogram
    colors = ['green' if val >= 0 else 'red' for val in indicators['macd_histogram']]
    fig.add_trace(
        go.Bar(
            x=data.index,
            y=indicators['macd_histogram'],
            name='MACD Histogram',
            marker_color=colors,
            opacity=0.7
        )
    )
    
    fig.update_layout(
        title='MACD Analysis',
        xaxis_title='Date',
        yaxis_title='MACD',
        height=400
    )
    
    return fig

def main():
    st.markdown('<h1 class="main-header">📊 Indian Stock Market Research Assistant</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.header("🔍 Stock Analysis")
    
    # Stock symbol input
    stock_symbol = st.sidebar.text_input(
        "Enter Stock Symbol",
        placeholder="e.g., RELIANCE, TCS, INFY",
        help="Enter the stock symbol without .NS or .BO suffix"
    )

    # Analysis period
    period = st.sidebar.selectbox(
        "Analysis Period",
        ["1mo", "3mo", "6mo", "1y", "2y", "5y"],
        index=3
    )

    # Future date input for prediction
    future_date_str = st.sidebar.text_input(
        "Enter Future Date to Predict Closing Price (YYYY-MM-DD)",
        value=(datetime.now() + timedelta(days=30)).strftime("%Y-%m-%d"),
        help="Enter a future date in YYYY-MM-DD format."
    )
    
    # LLM Prediction section
    st.sidebar.markdown("---")
    st.sidebar.subheader("🤖 AI-Powered Prediction")
    
    # OpenRouter API key input
    openrouter_api_key = st.sidebar.text_input(
        "OpenRouter API Key",
        type="password",
        help="Enter your OpenRouter API key for AI-powered predictions"
    )
    
    predict_button = st.sidebar.button("🔮 Predict Closing Price")
    llm_predict_button = st.sidebar.button("🤖 AI Predict Price", type="secondary")

    # Analysis button
    analyze_button = st.sidebar.button("🚀 Generate Analysis", type="primary")
    
    if analyze_button and stock_symbol:
        with st.spinner("🔍 Analyzing stock data..."):
            assistant = IndianStockResearchAssistant()
            
            # Get stock data
            stock, data = assistant.get_stock_data(stock_symbol, period)
            
            # Check if data is None or empty
            if stock is None or data is None or data.empty:
                st.error(f"❌ Unable to fetch data for {stock_symbol}. Please check the stock symbol.")
                return
            
            # Calculate indicators
            indicators = assistant.calculate_technical_indicators(data)
            
            # Get fundamental data
            fundamental_data = assistant.get_fundamental_data(stock)
            valuation_metrics = assistant.calculate_valuation_metrics(fundamental_data)
            
            # Get sentiment and derivatives data
            sentiment_data = assistant.get_news_sentiment(stock_symbol)
            derivatives_data = assistant.get_derivatives_data(stock_symbol)
            
            # Determine trend
            trend = assistant.determine_trend(data)
            
            # Support and resistance levels
            support_levels, resistance_levels = assistant.identify_support_resistance(data)
            
            # Candlestick patterns
            candlestick_patterns = assistant.analyze_candlestick_patterns(data)
            
            # Display results
            st.success(f"✅ Analysis completed for {stock_symbol.upper()}")
            
            # Current price and trend
            if data is not None:
                current_price = data['Close'].iloc[-1]
                prev_price = data['Close'].iloc[-2]
                price_change = ((current_price - prev_price) / prev_price) * 100
            else:
                current_price = prev_price = price_change = None
            
            # Header metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                if current_price is not None and price_change is not None:
                    st.metric(
                        "Current Price",
                        f"₹{current_price:.2f}",
                        f"{price_change:+.2f}%"
                    )
                else:
                    st.metric("Current Price", "N/A", "N/A")
            
            with col2:
                trend_color = "trend-bullish" if trend == "Bullish" else "trend-bearish" if trend == "Bearish" else "trend-sideways"
                st.markdown(f'<div class="metric-card"><strong>Trend:</strong> <span class="{trend_color}">{trend.upper()}</span></div>', unsafe_allow_html=True)
            
            with col3:
                rsi_value = indicators.get('rsi', pd.Series()).iloc[-1]
                if not pd.isna(rsi_value):
                    st.metric("RSI", f"{rsi_value:.1f}")
                else:
                    st.metric("RSI", "N/A")
            
            with col4:
                if data is not None:
                    volume = data['Volume'].iloc[-1]
                    st.metric("Volume", f"{volume:,.0f}")
                else:
                    st.metric("Volume", "N/A")
            
            # Tabs for different sections
            tab1, tab2, tab3, tab4, tab5 = st.tabs(["📈 Technical Analysis", "📊 Fundamentals", "🧠 Sentiment", "📋 Full Report", "📈 Charts"])
            
            with tab1:
                st.header("Technical Analysis")
                
                # Key indicators
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Key Indicators")
                    
                    # RSI
                    rsi_value = indicators.get('rsi', pd.Series()).iloc[-1]
                    if not pd.isna(rsi_value):
                        rsi_status = "Overbought" if rsi_value > 70 else "Oversold" if rsi_value < 30 else "Neutral"
                        st.write(f"**RSI**: {rsi_value:.2f} ({rsi_status})")
                    
                    # MACD
                    macd_value = indicators.get('macd', pd.Series()).iloc[-1]
                    if not pd.isna(macd_value):
                        st.write(f"**MACD**: {macd_value:.2f}")
                    
                    # Moving Averages
                    sma_50 = indicators.get('sma_50', pd.Series()).iloc[-1]
                    sma_200 = indicators.get('sma_200', pd.Series()).iloc[-1]
                    
                    if not pd.isna(sma_50):
                        st.write(f"**50-Day SMA**: ₹{sma_50:.2f}")
                    if not pd.isna(sma_200):
                        st.write(f"**200-Day SMA**: ₹{sma_200:.2f}")
                
                with col2:
                    st.subheader("Support & Resistance")
                    if data is not None and support_levels:
                        nearest_support = min(support_levels[-3:])
                        st.write(f"**Nearest Support**: ₹{nearest_support:.2f}")
                    if data is not None and resistance_levels:
                        nearest_resistance = max(resistance_levels[-3:])
                        st.write(f"**Nearest Resistance**: ₹{nearest_resistance:.2f}")
                    st.subheader("Recent Patterns")
                    if data is not None and candlestick_patterns:
                        for pattern in candlestick_patterns:
                            st.write(f"• {pattern}")
                    else:
                        st.write("• No significant patterns detected")
            
            with tab2:
                st.header("Fundamental Analysis")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Business Overview")
                    
                    sector = fundamental_data.get('info', {}).get('sector', 'N/A')
                    industry = fundamental_data.get('info', {}).get('industry', 'N/A')
                    market_cap = valuation_metrics.get('market_cap', 'N/A')
                    
                    st.write(f"**Sector**: {sector}")
                    st.write(f"**Industry**: {industry}")
                    
                    if isinstance(market_cap, (int, float)):
                        st.write(f"**Market Cap**: ₹{market_cap:,.0f}")
                    else:
                        st.write(f"**Market Cap**: {market_cap}")
                
                with col2:
                    st.subheader("Valuation Metrics")
                    
                    pe_ratio = valuation_metrics.get('pe_ratio', 'N/A')
                    pb_ratio = valuation_metrics.get('pb_ratio', 'N/A')
                    roe = valuation_metrics.get('roe', 'N/A')
                    
                    st.write(f"**P/E Ratio**: {pe_ratio}")
                    st.write(f"**P/B Ratio**: {pb_ratio}")
                    st.write(f"**ROE**: {roe}")
                    
                    revenue = valuation_metrics.get('revenue', 'N/A')
                    net_income = valuation_metrics.get('net_income', 'N/A')
                    
                    if isinstance(revenue, (int, float)):
                        st.write(f"**Revenue**: ₹{revenue:,.0f}")
                    else:
                        st.write(f"**Revenue**: {revenue}")
                    
                    if isinstance(net_income, (int, float)):
                        st.write(f"**Net Income**: ₹{net_income:,.0f}")
                    else:
                        st.write(f"**Net Income**: {net_income}")
            
            with tab3:
                st.header("Sentiment & Derivatives Analysis")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("News Sentiment")
                    
                    sentiment = sentiment_data['sentiment']
                    sentiment_score = sentiment_data['score']
                    news_count = sentiment_data['news_count']
                    
                    st.write(f"**Overall Sentiment**: {sentiment}")
                    st.write(f"**Sentiment Score**: {sentiment_score:.2f}")
                    st.write(f"**News Articles**: {news_count}")
                    
                    # Sentiment gauge
                    fig = go.Figure(go.Indicator(
                        mode = "gauge+number+delta",
                        value = sentiment_score,
                        domain = {'x': [0, 1], 'y': [0, 1]},
                        title = {'text': "Sentiment Score"},
                        delta = {'reference': 0},
                        gauge = {
                            'axis': {'range': [-1, 1]},
                            'bar': {'color': "darkblue"},
                            'steps': [
                                {'range': [-1, -0.3], 'color': "lightgray"},
                                {'range': [-0.3, 0.3], 'color': "yellow"},
                                {'range': [0.3, 1], 'color': "lightgreen"}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': 0
                            }
                        }
                    ))
                    
                    fig.update_layout(height=300)
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    st.subheader("Derivatives Data")
                    
                    oi_buildup = derivatives_data['oi_buildup']
                    pcr = derivatives_data['pcr']
                    max_pain = derivatives_data['max_pain']
                    iv = derivatives_data['iv']
                    
                    st.write(f"**OI Build-up**: {oi_buildup}")
                    st.write(f"**Put/Call Ratio**: {pcr}")
                    st.write(f"**Max Pain**: ₹{max_pain}")
                    st.write(f"**Implied Volatility**: {iv}%")
            
            with tab4:
                st.header("Complete Research Report")
                
                # Generate full report
                report = assistant._format_report(
                    stock_symbol, data, indicators, fundamental_data, 
                    valuation_metrics, sentiment_data, derivatives_data,
                    trend, support_levels, resistance_levels, candlestick_patterns
                )
                
                st.markdown(report)
                
                # Download button
                st.download_button(
                    label="📥 Download Report",
                    data=report,
                    file_name=f"{stock_symbol}_research_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                    mime="text/markdown"
                )
            
            with tab5:
                st.header("Interactive Charts")
                
                # Price chart with indicators
                if data is not None:
                    price_chart = create_price_chart(data, indicators)
                    st.plotly_chart(price_chart, use_container_width=True)
                else:
                    st.warning("No data available for price chart.")
                
                # MACD chart
                if data is not None:
                    macd_chart = create_macd_chart(data, indicators)
                    if macd_chart:
                        st.plotly_chart(macd_chart, use_container_width=True)
                else:
                    st.warning("No data available for MACD chart.")
                
                # Volume analysis
                st.subheader("Volume Analysis")
                if data is not None:
                    volume_data = pd.DataFrame({
                        'Date': data.index,
                        'Volume': data['Volume'],
                        'Price': data['Close']
                    })
                    fig = px.scatter(
                        volume_data,
                        x='Price',
                        y='Volume',
                        title='Volume vs Price Analysis',
                        color='Volume',
                        size='Volume'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.error("No data available for volume analysis.")
    
    if predict_button and stock_symbol and future_date_str:
        st.subheader(f"🔮 Predicted Closing Price for {stock_symbol.upper()} on {future_date_str}")
        try:
            assistant = IndianStockResearchAssistant()
            stock, data = assistant.get_stock_data(stock_symbol, period="5y")
            if data is None or data.empty:
                st.error(f"❌ Unable to fetch data for {stock_symbol}.")
            else:
                df = data.reset_index()
                if 'Date' not in df.columns:
                    df.rename(columns={df.columns[0]: 'Date'}, inplace=True)
                df['Date'] = pd.to_datetime(df['Date'])
                df = df.sort_values('Date')
                df = df[['Date', 'Close']].dropna()
                prophet_df = df.rename(columns={'Date': 'ds', 'Close': 'y'})
                prophet_df['ds'] = prophet_df['ds'].dt.tz_localize(None)
                future_date = datetime.strptime(future_date_str, "%Y-%m-%d")
                from importlib.util import find_spec
                if find_spec("prophet"):
                    from prophet import Prophet
                    model = Prophet()
                    model.fit(prophet_df)
                    last_date = prophet_df['ds'].max()
                    days_ahead = (future_date - last_date).days
                    if days_ahead <= 0:
                        st.error("❌ Future date must be after the last available date in the data.")
                    else:
                        future = model.make_future_dataframe(periods=days_ahead)
                        forecast = model.predict(future)
                        pred_row = forecast[forecast['ds'] == pd.to_datetime(future_date_str)]
                        if pred_row.empty:
                            st.error("❌ Unable to predict for the given date. Try a different date.")
                        else:
                            predicted_price = pred_row['yhat'].values[0]
                            st.success(f"Predicted closing price: ₹{predicted_price:.2f}")
                            import matplotlib.pyplot as plt
                            fig, ax = plt.subplots(figsize=(10,5))
                            ax.plot(df['Date'], df['Close'], label='Actual Close')
                            ax.plot(forecast['ds'], forecast['yhat'], label='Forecasted Trend', linestyle='--')
                            ax.scatter([future_date], [predicted_price], color='red', label=f'Predicted {future_date_str}: ₹{predicted_price:.2f}', zorder=5)
                            ax.set_title(f"{stock_symbol} Closing Price Prediction")
                            ax.set_xlabel("Date")
                            ax.set_ylabel("Close Price (INR)")
                            ax.legend()
                            ax.grid(True)
                            st.pyplot(fig)
                else:
                    try:
                        from statsmodels.tsa.arima.model import ARIMA
                        ts = df.set_index('Date')['Close']
                        # Ensure the time series has proper frequency information
                        ts = ts.asfreq('B')  # Business day frequency
                        ts = ts.fillna(method='ffill')  # Forward fill any missing values
                        order = (5,1,0)
                        model = ARIMA(ts, order=order)
                        model_fit = model.fit()
                        last_date = ts.index.max()
                        days_ahead = (future_date - last_date).days
                        if days_ahead <= 0:
                            st.error("❌ Future date must be after the last available date in the data.")
                        else:
                            forecast = model_fit.get_forecast(steps=days_ahead)
                            forecast_index = pd.date_range(start=last_date + timedelta(days=1), periods=days_ahead, freq='B')
                            forecast_series = pd.Series(forecast.predicted_mean.values, index=forecast_index)
                            if pd.to_datetime(future_date_str) not in forecast_series.index:
                                st.error("❌ Unable to predict for the given date (may not be a business day). Try a different date.")
                            else:
                                predicted_price = forecast_series[pd.to_datetime(future_date_str)]
                                st.success(f"Predicted closing price: ₹{predicted_price:.2f}")
                                import matplotlib.pyplot as plt
                                fig, ax = plt.subplots(figsize=(10,5))
                                ax.plot(df['Date'], df['Close'], label='Actual Close')
                                ax.plot(forecast_series.index, forecast_series.values, label='Forecasted Trend', linestyle='--')
                                ax.scatter([future_date], [predicted_price], color='red', label=f'Predicted {future_date_str}: ₹{predicted_price:.2f}', zorder=5)
                                ax.set_title(f"{stock_symbol} Closing Price Prediction")
                                ax.set_xlabel("Date")
                                ax.set_ylabel("Close Price (INR)")
                                ax.legend()
                                ax.grid(True)
                                st.pyplot(fig)
                    except ImportError:
                        st.error("❌ ARIMA forecasting not available. Please install statsmodels: pip install statsmodels")
                    except Exception as e:
                        st.error(f"❌ Error in ARIMA forecasting: {str(e)}")
        except Exception as e:
            st.error(f"❌ Error: {e}")
    
    # LLM Prediction
    if llm_predict_button and stock_symbol and future_date_str and openrouter_api_key:
        st.subheader(f"🤖 AI-Powered Price Prediction for {stock_symbol.upper()} on {future_date_str}")
        
        with st.spinner("🤖 AI is analyzing market data and generating prediction..."):
            try:
                predictor = LLMPricePredictor(api_key=openrouter_api_key)
                result = predictor.predict_price_with_llm(stock_symbol, future_date_str)
                
                if result['success']:
                    # Display prediction results
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric(
                            "Current Price",
                            f"₹{result['current_price']:.2f}"
                        )
                    
                    with col2:
                        st.metric(
                            "Predicted Price",
                            f"₹{result['predicted_price']:.2f}",
                            f"{result['price_change_predicted']:+.2f}%"
                        )
                    
                    with col3:
                        confidence_color = "green" if result['confidence_level'] >= 70 else "orange" if result['confidence_level'] >= 50 else "red"
                        st.markdown(f"""
                        <div style="text-align: center;">
                            <div style="color: {confidence_color}; font-size: 1.5rem; font-weight: bold;">
                                {result['confidence_level']:.1f}%
                            </div>
                            <div style="font-size: 0.9rem;">Confidence</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Trend direction
                    trend_color = "green" if result['trend_direction'] == 'bullish' else "red" if result['trend_direction'] == 'bearish' else "orange"
                    st.markdown(f"""
                    <div style="text-align: center; margin: 1rem 0;">
                        <span style="color: {trend_color}; font-size: 1.2rem; font-weight: bold;">
                            Trend: {result['trend_direction'].upper()}
                        </span>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # AI Reasoning
                    st.subheader("🤖 AI Analysis & Reasoning")
                    st.info(result['reasoning'])
                    
                    # Risk factors and levels
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("⚠️ Risk Factors")
                        if result['risk_factors']:
                            for risk in result['risk_factors']:
                                st.write(f"• {risk}")
                        else:
                            st.write("• No specific risk factors identified")
                    
                    with col2:
                        st.subheader("📊 Key Levels")
                        if result['support_levels']:
                            st.write("**Support Levels:**")
                            for level in result['support_levels']:
                                st.write(f"• ₹{level:.2f}")
                        
                        if result['resistance_levels']:
                            st.write("**Resistance Levels:**")
                            for level in result['resistance_levels']:
                                st.write(f"• ₹{level:.2f}")
                    
                    # Disclaimer
                    st.warning("""
                    ⚠️ **Disclaimer**: This AI prediction is for educational and research purposes only. 
                    It should not be considered as financial advice. Always do your own research and 
                    consult with financial professionals before making investment decisions.
                    """)
                    
                else:
                    st.error(f"❌ AI Prediction Error: {result['error']}")
                    
            except Exception as e:
                st.error(f"❌ Error in AI prediction: {str(e)}")
    
    elif llm_predict_button and (not stock_symbol or not future_date_str or not openrouter_api_key):
        st.warning("⚠️ Please provide stock symbol, future date, and OpenRouter API key for AI prediction.")
    
    else:
        # Welcome message
        st.markdown("""
        ## Welcome to the Indian Stock Market Research Assistant! 🚀
        
        This comprehensive tool provides detailed analysis for Indian stocks listed on NSE or BSE.
        
        ### Features:
        - 📈 **Technical Analysis**: RSI, MACD, Moving Averages, Bollinger Bands
        - 📊 **Fundamental Analysis**: Financial metrics, valuation ratios
        - 🧠 **Sentiment Analysis**: News sentiment and market mood
        - 📋 **Derivatives Data**: Options analysis and market positioning
        - 📈 **Interactive Charts**: Visual analysis with Plotly charts
        - 🤖 **AI-Powered Predictions**: LLM-based price predictions using OpenRouter API
        
        ### How to use:
        1. Enter a stock symbol (e.g., RELIANCE, TCS, INFY)
        2. Select the analysis period
        3. Click "Generate Analysis"
        4. Explore the different tabs for comprehensive insights
        5. For AI predictions: Enter your OpenRouter API key and click "AI Predict Price"
        
        ### Popular Indian Stocks:
        - **RELIANCE** - Reliance Industries
        - **TCS** - Tata Consultancy Services
        - **INFY** - Infosys
        - **HDFCBANK** - HDFC Bank
        - **ICICIBANK** - ICICI Bank
        - **SBIN** - State Bank of India
        
        ### 🤖 AI Prediction Setup:
        To use the AI-powered price prediction feature:
        1. Get a free API key from [OpenRouter](https://openrouter.ai/)
        2. Enter your API key in the sidebar
        3. Provide a stock symbol and future date
        4. Click "AI Predict Price" for LLM-based analysis
        """)
        
        # Sample analysis
        st.subheader("💡 Sample Analysis")
        if st.button("Try Sample Analysis (RELIANCE)"):
            st.info("Click 'Generate Analysis' in the sidebar with 'RELIANCE' as the stock symbol to see a sample analysis.")

if __name__ == "__main__":
    main() 