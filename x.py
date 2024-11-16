# Import necessary libraries
import pandas as pd
import numpy as np
import yfinance as yf
import streamlit as st
import tensorflow as tf
import plotly.graph_objects as go
import seaborn as sns
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
import time
import matplotlib.pyplot as plt

# Helper Functions
def fetch_stock_data(ticker, interval="1m", period="1d"):
    """Fetch real-time stock data."""
    stock_data = yf.download(ticker, interval=interval, period=period)
    return stock_data

def compute_technical_indicators(df):
    """Compute technical indicators like SMA, EMA, RSI, MACD."""
    # SMA and EMA
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['SMA_200'] = df['Close'].rolling(window=200).mean()
    df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
    df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
    
    # RSI
    df['RSI'] = compute_rsi(df['Close'])
    
    # MACD
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    return df

def compute_rsi(series, period=14):
    """Calculate Relative Strength Index (RSI)."""
    delta = series.diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=period).mean()
    avg_loss = pd.Series(loss).rolling(window=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def prepare_data(df):
    """Prepare data for TensorFlow model training."""
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df[['Close']])
    X, y = [], []
    for i in range(60, len(scaled_data)):
        X.append(scaled_data[i-60:i])
        y.append(scaled_data[i])
    X, y = np.array(X), np.array(y)
    return X, y, scaler

def build_and_train_model(X, y):
    """Build and train LSTM model using TensorFlow."""
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(50, return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
        tf.keras.layers.LSTM(50, return_sequences=False),
        tf.keras.layers.Dense(25),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X, y, batch_size=32, epochs=5)
    return model

# Streamlit UI layout
st.title("Stock Market Trend Prediction and Technical Analysis")
st.sidebar.subheader("Stock Ticker and Time Range")
ticker = st.sidebar.text_input("Enter Stock Ticker (e.g., AAPL)", value="AAPL")
start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2023-01-01"))
end_date = st.sidebar.date_input("End Date", value=datetime.now().date())

# Fetch stock data
df = yf.download(ticker, start=start_date, end=end_date)

if df.empty:
    st.warning(f"No data found for {ticker}. Please check the ticker symbol or date range.")
else:
    # Technical Indicators Computation
    df = compute_technical_indicators(df)
    
    # Data Visualization using Seaborn
    st.subheader(f"Closing Price and Technical Indicators for {ticker}")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(data=df[['Close', 'SMA_50', 'SMA_200']], ax=ax)
    ax.set_title(f'{ticker} Closing Price and Moving Averages')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price (USD)')
    st.pyplot(fig)

    # Real-time stock data display
    st.subheader("Real-Time Stock Price (Updated Every 5 Seconds)")
    real_time_data = fetch_stock_data(ticker, interval="1m", period="1d")
    st.write(f"Current Price: ${real_time_data['Close'].iloc[-1]:.2f}")
    st.line_chart(real_time_data['Close'], width=700, height=300)

    # Prepare data for LSTM model
    X, y, scaler = prepare_data(df)
    
    # Train TensorFlow Model
    model = build_and_train_model(X, y)
    
    # Predict using the trained model
    pred_data = real_time_data[['Close']].values
    scaled_pred = scaler.transform(pred_data[-60:].reshape(-1, 1))
    scaled_pred = scaled_pred.reshape(1, 60, 1)
    predicted_price = model.predict(scaled_pred)
    predicted_price = scaler.inverse_transform(predicted_price)

    st.subheader(f"Predicted Next Price for {ticker}")
    st.write(f"Predicted Price: ${predicted_price[0][0]:.2f}")

    # More charts using Plotly
    st.subheader(f"MACD and Signal Line for {ticker}")
    fig_macd = go.Figure()
    fig_macd.add_trace(go.Scatter(x=df.index, y=df['MACD'], mode='lines', name='MACD'))
    fig_macd.add_trace(go.Scatter(x=df.index, y=df['Signal_Line'], mode='lines', name='Signal Line', line=dict(dash='dot')))
    fig_macd.update_layout(title="MACD and Signal Line", xaxis_title="Date", yaxis_title="MACD Value")
    st.plotly_chart(fig_macd, use_container_width=True)

    st.subheader(f"RSI for {ticker}")
    fig_rsi = go.Figure(go.Scatter(x=df.index, y=df['RSI'], mode='lines', name='RSI', line=dict(color='orange')))
    fig_rsi.update_layout(title="Relative Strength Index (RSI)", xaxis_title="Date", yaxis_title="RSI Value")
    st.plotly_chart(fig_rsi, use_container_width=True)

# Mobile Friendly Customization
st.markdown("""
    <style>
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            padding: 10px 20px;
            border-radius: 5px;
            font-size: 16px;
            transition: background-color 0.3s ease;
        }

        .stButton>button:hover {
            background-color: #45a049;
        }

        .stTextInput input {
            font-size: 16px;
            padding: 10px;
        }

        .stTextInput label {
            font-size: 14px;
        }
    </style>
""", unsafe_allow_html=True)

# Additional Suggestions
st.write("""
    ## Suggestions:
    - Explore different technical indicators and see how they affect your stock predictions.
    - Monitor real-time stock prices for better decision-making.
    - Use the TensorFlow model to predict stock trends in the short term.

    Happy trading!
""")

            font-size: 14px;
        }
    </style>
""", unsafe_allow_html=True)

# Final Suggestions
st.write("Explore different technical indicators and track your stocks on the portfolio and watchlist tabs!")
