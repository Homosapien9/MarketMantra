import numpy as np
import pandas as pd
import yfinance as yf
import xgboost as xgb
import plotly.graph_objects as go
import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from datetime import datetime

def get_stock_data(ticker, start_date, end_date):
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    return stock_data
# Helper functions for technical indicators
def compute_bollinger_bands(df, window=20, num_std=2):
    rolling_mean = df['Close'].rolling(window=window).mean()
    rolling_std = df['Close'].rolling(window=window).std()
    upper_band = rolling_mean + (rolling_std * num_std)
    lower_band = rolling_mean - (rolling_std * num_std)
    return upper_band, lower_band

def compute_rsi(df, window=14):
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def compute_macd(df, fast=12, slow=26, signal=9):
    macd_line = df['Close'].ewm(span=fast, adjust=False).mean() - df['Close'].ewm(span=slow, adjust=False).mean()
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line

# Initialize session state variables
if 'portfolio' not in st.session_state:
    st.session_state['portfolio'] = []
if 'watchlist' not in st.session_state:
    st.session_state['watchlist'] = []

# Load QR code image
qr_image = Image.open("MarketMantra_website.png")

# Layout with columns
col1, col2 = st.columns([4, 1])
with col1:
    st.markdown('<h1 style="color: Cyan; font-size: 30px;">MarketMantra - A Stock Trend Predictor</h1>', unsafe_allow_html=True)
with col2:
    st.image(qr_image, caption="Scan for website", width=100)
    st.subheader("~ Made By Jatan Shah")

# Expander for input
with st.expander("Select Stock, Date Range, and Technical Indicators", expanded=True):
    stock_symbol = st.text_input("Enter Stock Ticker (e.g., AAPL)", value="JSWSTEEL.NS")
    start_date = st.date_input("Start Date", value=pd.to_datetime("2024-01-01"))
    end_date = st.date_input("End Date", value=datetime.now().date())
    
    selected_indicators = st.multiselect(
        "Select Technical Indicators to Display",
        [
            "50-Day Simple Moving Average (SMA)",
            "200-Day Simple Moving Average (SMA)",
            "MACD (Moving Average Convergence Divergence)",
            "Stochastic Oscillator",
            "Bollinger Bands",
            "Relative Strength Index (RSI)"
        ],
        default=["50-Day Simple Moving Average (SMA)", "200-Day Simple Moving Average (SMA)"]
    )

# Fetch stock data
df = get_stock_data(stock_symbol, start_date, end_date)
if df.empty:
    st.warning("No data found for the selected stock or date range.")
    st.stop()

# Candlestick chart generation using Plotly
def plot_candlestick(df):
    fig = go.Figure(data=[go.Candlestick(x=df.index,
                                        open=df['Open'],
                                        high=df['High'],
                                        low=df['Low'],
                                        close=df['Close'],
                                        increasing_line_color='green', decreasing_line_color='red')])
    fig.update_layout(title=f'{stock_symbol} Candlestick Chart', xaxis_title='Date', yaxis_title='Price')
    st.plotly_chart(fig, use_container_width=True)

# Plot the Candlestick Chart
with st.expander("Candlestick Chart", expanded=True):
    plot_candlestick(df)

# Data visualization and processing
with st.expander("Data Visualization"):
    st.subheader(f"Closing Price for {stock_symbol}")
    fig, ax = plt.subplots(figsize=(15, 5))
    ax.plot(df['Close'], label="Close Price", color="blue")
    ax.set_title(f"{stock_symbol} Closing Price", fontsize=15)
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend()
    st.pyplot(fig)

# Feature Engineering and Model Preparation
df['Previous Close'] = df['Close'].shift(1)
df['Daily Return'] = df['Close'].pct_change()
df['Target'] = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)  # Binary target for up/down trend
df.dropna(inplace=True)

# Add additional features
df['RSI'] = compute_rsi(df)
upper_band, lower_band = compute_bollinger_bands(df)
df['Bollinger Upper'] = upper_band
df['Bollinger Lower'] = lower_band

# Prepare features and target
features = df[['Previous Close', 'Daily Return', 'RSI', 'Bollinger Upper', 'Bollinger Lower']].values
target = df['Target'].values

# Scaling features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Split data
X_train, X_valid, Y_train, Y_valid = train_test_split(features_scaled, target, test_size=0.1, random_state=2500)

# Model Setup
models = {
    "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=15, random_state=45),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=15, random_state=45),
    "XGBoost": xgb.XGBClassifier(n_estimators=100, max_depth=15, learning_rate=0.1, random_state=45),
    "Decision Tree": DecisionTreeClassifier(random_state=45)
}

# Train models and store predictions
model_predictions = []
model_accuracies = {}
for model_name, model in models.items():
    with st.spinner(f"Training {model_name}..."):
        model.fit(X_train, Y_train)
        model_pred = model.predict(X_valid)
        model_predictions.append(model_pred)
        
        # Model accuracy
        accuracy = accuracy_score(Y_valid, model_pred) * 100
        model_accuracies[model_name] = accuracy

# Display accuracies of all models
st.subheader("Model Performance Metrics")
for model_name, accuracy in model_accuracies.items():
    st.write(f"{model_name}: {accuracy:.2f}%")

# Compute the average prediction (0 = Down, 1 = Up)
model_predictions = np.array(model_predictions)
average_predictions = np.mean(model_predictions, axis=0)
final_predictions = np.round(average_predictions)

# Confusion Matrix
cm = confusion_matrix(Y_valid, final_predictions)
fig, ax = plt.subplots(figsize=(6, 6))
cax = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
fig.colorbar(cax)
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

# Continue with other tabs (Portfolio and Watchlist)
tab1, tab2, tab3, tab4 = st.tabs(["Portfolio", "Watchlist", "Technical Indicators", "Predictions"])
# Continued from previous code...

# Portfolio Tab
with tab1:
    st.subheader("Portfolio")
    st.write("Manage your stock portfolio by adding or removing stocks.")
    
    stock_name = st.text_input("Enter Stock Symbol to Add to Portfolio", value="")
    if st.button("Add Stock to Portfolio"):
        if stock_name and stock_name not in st.session_state['portfolio']:
            st.session_state['portfolio'].append(stock_name)
            st.success(f"Added {stock_name} to your portfolio!")
        elif stock_name in st.session_state['portfolio']:
            st.warning(f"{stock_name} is already in your portfolio.")
        else:
            st.warning("Please enter a stock symbol to add.")

    # Display portfolio
    st.write("Your Current Portfolio:")
    if st.session_state['portfolio']:
        for stock in st.session_state['portfolio']:
            st.write(f"- {stock}")
    else:
        st.write("No stocks in your portfolio yet.")

    # Option to remove stock
    remove_stock = st.selectbox("Select Stock to Remove", st.session_state['portfolio'], index=0)
    if st.button("Remove Selected Stock"):
        if remove_stock:
            st.session_state['portfolio'].remove(remove_stock)
            st.success(f"Removed {remove_stock} from your portfolio!")
        else:
            st.warning("Please select a stock to remove.")

# Watchlist Tab
with tab2:
    st.subheader("Watchlist")
    st.write("Manage your stock watchlist to track stocks you're interested in.")
    
    watchlist_stock = st.text_input("Enter Stock Symbol to Add to Watchlist", value="")
    if st.button("Add Stock to Watchlist"):
        if watchlist_stock and watchlist_stock not in st.session_state['watchlist']:
            st.session_state['watchlist'].append(watchlist_stock)
            st.success(f"Added {watchlist_stock} to your watchlist!")
        elif watchlist_stock in st.session_state['watchlist']:
            st.warning(f"{watchlist_stock} is already in your watchlist.")
        else:
            st.warning("Please enter a stock symbol to add.")

    # Display watchlist
    st.write("Your Current Watchlist:")
    if st.session_state['watchlist']:
        for stock in st.session_state['watchlist']:
            st.write(f"- {stock}")
    else:
        st.write("No stocks in your watchlist yet.")

    # Option to remove stock
    remove_watchlist_stock = st.selectbox("Select Stock to Remove from Watchlist", st.session_state['watchlist'], index=0)
    if st.button("Remove Selected Stock"):
        if remove_watchlist_stock:
            st.session_state['watchlist'].remove(remove_watchlist_stock)
            st.success(f"Removed {remove_watchlist_stock} from your watchlist!")
        else:
            st.warning("Please select a stock to remove.")

# Technical Indicators Tab
with tab3:
    st.header("Technical Indicators")
    
    if "50-Day Simple Moving Average (SMA)" in selected_indicators:
        st.header("50-Day Simple Moving Average (SMA)")
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        st.line_chart(df['SMA_50'])

    if "200-Day Simple Moving Average (SMA)" in selected_indicators:
        st.header("200-Day Simple Moving Average (SMA)")
        df['SMA_200'] = df['Close'].rolling(window=200).mean()
        st.line_chart(df['SMA_200'])

    if "MACD (Moving Average Convergence Divergence)" in selected_indicators:
        st.header("MACD")
        macd_line, signal_line = compute_macd(df)
        df['MACD'] = macd_line
        df['MACD_Signal'] = signal_line
        st.line_chart(df[['MACD', 'MACD_Signal']])

    if "Bollinger Bands" in selected_indicators:
        st.header("Bollinger Bands")
        df['Bollinger Upper'] = upper_band
        df['Bollinger Lower'] = lower_band
        st.line_chart(df[['Bollinger Upper', 'Bollinger Lower']])

    if "Relative Strength Index (RSI)" in selected_indicators:
        st.header("Relative Strength Index (RSI)")
        df['RSI'] = compute_rsi(df)
        st.line_chart(df['RSI'])

# Predictions Tab
with tab4:
    st.header("Stock Trend Predictions")
    st.write(f"The predicted trend for {stock_symbol} over the next period based on the model ensemble is:")

    prediction = "Up" if final_predictions[-1] == 1 else "Down"
    st.write(f"**Predicted Trend**: {prediction}")

    st.subheader("Prediction Metrics")
    st.write(f"**Accuracy of Predictions**:")
    for model_name, accuracy in model_accuracies.items():
        st.write(f"{model_name}: {accuracy:.2f}%")

    st.subheader("Model Performance: Confusion Matrix")
    st.pyplot(fig)

    st.subheader("Final Prediction Based on Model Voting")
    if final_predictions[-1] == 1:
        st.success("The model predicts an **UP** trend.")
    else:
        st.error("The model predicts a **DOWN** trend.")

# Additional Customizations: Hover effects, animations, etc.
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

# Final Suggestions
st.write("Explore different technical indicators and track your stocks on the portfolio and watchlist tabs!")
