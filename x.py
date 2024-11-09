import numpy as np
import pandas as pd
import xgboost as xgb
import yfinance as yf
import streamlit as st
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from datetime import datetime



# Define a helper function for stock data
def get_stock_data(stock_symbol, start_date, end_date):
    try:
        df = yf.download(stock_symbol, start=start_date, end=end_date)
        df.drop(columns=['Adj Close'], inplace=True)
        return df
    except Exception as e:
        st.error(f"Error fetching stock data: {e}")
        return pd.DataFrame()

# Helper function to calculate RSI
def compute_rsi(df, window=14):
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=window, min_periods=1).mean()
    avg_loss = loss.rolling(window=window, min_periods=1).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Helper function to calculate MACD
def compute_macd(df, fast=12, slow=26, signal=9):
    macd_line = df['Close'].ewm(span=fast, adjust=False).mean() - df['Close'].ewm(span=slow, adjust=False).mean()
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line

# Helper function to calculate Stochastic Oscillator
def compute_stochastic(df, window=14):
    low_min = df['Low'].rolling(window=window).min()
    high_max = df['High'].rolling(window=window).max()
    stochastic = 100 * (df['Close'] - low_min) / (high_max - low_min)
    return stochastic

# Sidebar setup
st.title("Stock Market Trend Predictor")
st.subheader("~ by Jatan Shah")

# Sidebar for stock selection
st.sidebar.header("Stock Selection")
stock_symbol = st.sidebar.selectbox("Select Stock Ticker", ["JSWSTEEL.NS", "AAPL", "TSLA", "AMZN", "GOOGL", "MSFT", "NFLX", "META"])

# Sidebar for stock history
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2024-01-01"))
end_date = st.sidebar.date_input("End Date", datetime.now().date())

# Sidebar for technical indicators
st.sidebar.header("Technical Indicators")
indicator_options = [
    "50-Day Simple Moving Average (SMA)",
    "200-Day Simple Moving Average (SMA)",
    "RSI (Relative Strength Index)",
    "MACD (Moving Average Convergence Divergence)",
    "Stochastic Oscillator"
]

selected_indicators = st.sidebar.multiselect(
    "Select Technical Indicators to Display",
    indicator_options,
    default=["50-Day Simple Moving Average (SMA)", "200-Day Simple Moving Average (SMA)"]  # default selected indicators
)
if "50-Day Simple Moving Average (SMA)" in selected_indicators:
    sma_50 = True
else:
    sma_50 = False

if "200-Day Simple Moving Average (SMA)" in selected_indicators:
    sma_200 = True
else:
    sma_200 = False

if "RSI (Relative Strength Index)" in selected_indicators:
    rsi = True
else:
    rsi = False

if "MACD (Moving Average Convergence Divergence)" in selected_indicators:
    macd = True
else:
    macd = False

if "Stochastic Oscillator" in selected_indicators:
    stochastic = True
else:
    stochastic = False
# Sidebar for portfolio and watchlist tabs
tab = st.sidebar.selectbox("Portfolio & Watchlist", ["Portfolio", "Watchlist"])

# Portfolio and Watchlist Management
# Initialize portfolio and watchlist in session_state if they do not exist
if 'portfolio' not in st.session_state:
    st.session_state['portfolio'] = []

if 'watchlist' not in st.session_state:
    st.session_state['watchlist'] = []

# Sidebar setup for Portfolio & Watchlist Management
st.sidebar.header("Manage Portfolio & Watchlist")

# Select tab for adding stocks to portfolio or watchlist
portfolio_add = st.sidebar.button("Add to Portfolio")
watchlist_add = st.sidebar.button("Add to Watchlist")

# Add stock to portfolio
if portfolio_add:
    if stock_symbol not in st.session_state['portfolio']:
        st.session_state['portfolio'].append(stock_symbol)
        st.sidebar.success(f"{stock_symbol} added to Portfolio.")
    else:
        st.sidebar.warning(f"{stock_symbol} is already in your Portfolio.")

# Add stock to watchlist
if watchlist_add:
    if stock_symbol not in st.session_state['watchlist']:
        st.session_state['watchlist'].append(stock_symbol)
        st.sidebar.success(f"{stock_symbol} added to Watchlist.")
    else:
        st.sidebar.warning(f"{stock_symbol} is already in your Watchlist.")

# Fetch stock data
df = get_stock_data(stock_symbol, start_date, end_date)
if df.empty:
    st.warning("No data found for the selected stock or date range.")
    st.stop()

# Display stock data
st.subheader(f"Stock Data for {stock_symbol}")
st.write(f"Here is the historical stock data for {stock_symbol} from {start_date} to {end_date}.")
st.dataframe(df.tail())

# Data Visualization: Closing Price
with st.expander("Data Visualization"):
    st.subheader("Closing Price Over Time")
    fig, ax = plt.subplots(figsize=(15, 5))
    ax.plot(df['Close'], label='Close Price', color='blue')
    ax.set_title(f"{stock_symbol} - Closing Price History", fontsize=15)
    ax.set_ylabel('Price', fontsize=12)
    ax.set_xlabel('Date', fontsize=12)
    ax.grid(True)
    plt.legend()
    st.pyplot(fig)

# Feature Engineering and Model Preparation
df['Previous Close'] = df['Close'].shift(1)
df['Daily Return'] = df['Close'].pct_change()
df['Target'] = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)  # Binary target for up/down trend
df.dropna(inplace=True)

# Prepare features and target
features = df[['Previous Close', 'Daily Return']].values
target = df['Target'].values

# Scaling features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Split data
X_train, X_valid, Y_train, Y_valid = train_test_split(features_scaled, target, test_size=0.1, random_state=2500)

# Model Setup
models = {
    "Logistic Regression": LogisticRegression(),
    "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42),
    "Support Vector Classifier (SVC)": SVC(kernel="rbf", gamma='scale', C=1),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42),
    "K-Nearest Neighbors (KNN)": KNeighborsClassifier(n_neighbors=5),
    "XGBoost": xgb.XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Naive Bayes": GaussianNB(),
    "Neural Network (MLP)": MLPClassifier(max_iter=1000, random_state=42)
}

# Train and store model accuracies
model_accuracies = {}
for model_name, model in models.items():
    with st.spinner(f"Training {model_name}..."):
        model.fit(X_train, Y_train)
        y_pred = model.predict(X_valid)
        accuracy = accuracy_score(Y_valid, y_pred) * 100
        model_accuracies[model_name] = accuracy

# Model selection
selected_model = st.selectbox("Select Model for Accuracy", list(models.keys()))
for model_name, accuracy in model_accuracies.items():
    if model_name == selected_model:
        st.write(f"{model_name}: {accuracy:.2f}%")

# Confusion Matrix
st.subheader("Confusion Matrix")
cm = confusion_matrix(Y_valid, models[selected_model].predict(X_valid))
fig, ax = plt.subplots(figsize=(6, 6))
ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
ax.figure.colorbar(ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues))
classes = ['Down', 'Up']
ax.set(xticks=np.arange(len(classes)),
       yticks=np.arange(len(classes)),
       xticklabels=classes, yticklabels=classes,
       title="Confusion Matrix",
       ylabel="True Label", xlabel="Predicted Label")
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j, i, format(cm[i, j], 'd'), ha="center", va="center", color="black")
st.pyplot(fig)

# Real-time Prediction Section
st.subheader("Real-time Prediction for the Latest Data")
latest_data = df.iloc[-1:][['Previous Close', 'Daily Return']].values.reshape(1, -1)
latest_data_scaled = scaler.transform(latest_data)
predicted_trend = models[selected_model].predict(latest_data_scaled)

# Display recommendation with button and icon
st.subheader("Recommendation for Tomorrow's Trading")
if predicted_trend == 1:
    st.write(":green[**Recommendation:** Hold the stock for tomorrow.]")
else:
    st.write(":red[**Recommendation:** Sell the stock for tomorrow.]")
tab1, tab2 = st.tabs(["Portfolio", "Watchlist"])

# portf
with tab1:
    st.header("Your Portfolio")
    if st.session_state['portfolio']:
        for stock in st.session_state['portfolio']:
            st.write(stock)
    else:
        st.write("Your Portfolio is empty.")

# Display Watchlist Tab
with tab2:
    st.header("Your Watchlist")
    if st.session_state['watchlist']:
        for stock in st.session_state['watchlist']:
            st.write(stock)
    else:
        st.write("Your Watchlist is empty.")
# Technical Indicators
with st.expander("Technical Indicators (SMA(50 and 200 day),RSI, MACD, Stochastic)"):
    if sma_50:
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        fig, ax = plt.subplots(figsize=(15, 5))
        ax.plot(df['SMA_50'], label="50-Day SMA", color='orange')
        ax.set_title(f"{stock_symbol} - 50-Day Simple Moving Average", fontsize=15)
        ax.set_ylabel('Price', fontsize=12)
        ax.set_xlabel('Date', fontsize=12)
        ax.legend(loc='best')
        st.pyplot(fig)

    if sma_200:
        df['SMA_200'] = df['Close'].rolling(window=200).mean()
        fig, ax = plt.subplots(figsize=(15, 5))
        ax.plot(df['SMA_200'], label="200-Day SMA", color='green')
        ax.set_title(f"{stock_symbol} - 200-Day Simple Moving Average", fontsize=15)
        ax.set_ylabel('Price', fontsize=12)
        ax.set_xlabel('Date', fontsize=12)
        ax.legend(loc='best')
        st.pyplot(fig)

    if rsi:
        df['RSI'] = compute_rsi(df)
        fig, ax = plt.subplots(figsize=(15, 5))
        ax.plot(df['RSI'], label="RSI", color='purple')
        ax.axhline(70, linestyle='--', color='red')
        ax.axhline(30, linestyle='--', color='green')
        ax.set_title(f"{stock_symbol} - Relative Strength Index (RSI)", fontsize=15)
        ax.set_ylabel('RSI', fontsize=12)
        ax.set_xlabel('Date', fontsize=12)
        ax.legend(loc='best')
        st.pyplot(fig)

    if macd:
        macd_line, signal_line = compute_macd(df)
        fig, ax = plt.subplots(figsize=(15, 5))
        ax.plot(macd_line, label="MACD", color='blue')
        ax.plot(signal_line, label="Signal Line", color='orange')
        ax.set_title(f"{stock_symbol} - MACD", fontsize=15)
        ax.set_ylabel('Value', fontsize=12)
        ax.set_xlabel('Date', fontsize=12)
        ax.legend(loc='best')
        st.pyplot(fig)

    if stochastic:
        stochastic_oscillator = compute_stochastic(df)
        fig, ax = plt.subplots(figsize=(15, 5))
        ax.plot(stochastic_oscillator, label="Stochastic Oscillator", color='green')
        ax.axhline(80, linestyle='--', color='red')
        ax.axhline(20, linestyle='--', color='blue')
        ax.set_title(f"{stock_symbol} - Stochastic Oscillator", fontsize=15)
        ax.set_ylabel('Stochastic Value', fontsize=12)
        ax.set_xlabel('Date', fontsize=12)
        ax.legend(loc='best')
        st.pyplot(fig)
        ax.axhline(70, linestyle='--', color='red')
        ax.axhline(30, linestyle='--', color='green')
        ax.set_title(f"{stock_symbol} - Relative Strength Index (RSI)", fontsize=15)
        ax.set_ylabel('RSI', fontsize=12)
        ax.set_xlabel('Date', fontsize=12)
        ax.legend(loc='best')
        st.pyplot(fig)

    if macd:
        macd_line, signal_line = compute_macd(df)
        fig, ax = plt.subplots(figsize=(15, 5))
        ax.plot(macd_line, label="MACD", color='blue')
        ax.plot(signal_line, label="Signal Line", color='orange')
        ax.set_title(f"{stock_symbol} - MACD", fontsize=15)
        ax.set_ylabel('Value', fontsize=12)
        ax.set_xlabel('Date', fontsize=12)
        ax.legend(loc='best')
        st.pyplot(fig)

    if stochastic:
        stochastic_oscillator = compute_stochastic(df)
        fig, ax = plt.subplots(figsize=(15, 5))
        ax.plot(stochastic_oscillator, label="Stochastic Oscillator", color='green')
        ax.axhline(80, linestyle='--', color='red')
        ax.axhline(20, linestyle='--', color='blue')
        ax.set_title(f"{stock_symbol} - Stochastic Oscillator", fontsize=15)
        ax.set_ylabel('Stochastic Value', fontsize=12)
        ax.set_xlabel('Date', fontsize=12)
        ax.legend(loc='best')
        st.pyplot(fig)
        latest_market_price = current_stock.history(period='1d')['Close'].iloc[-1]
        st.subheader("Real-time Market Price")
        st.markdown(f"The real-time market price of **{stock_symbol}** is :green[**{latest_market_price:.2f}**].")
    except Exception as e:
        st.error(f"Error fetching real-time stock price: {e}")
