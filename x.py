import numpy as np
import pandas as pd
import xgboost as xgb
import yfinance as yf
from PIL import Image
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
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

qr_image = Image.open("MarketMantra_website.png")  # Replace with your QR code file

col1, col2 = st.columns([4, 1])  # Adjust column proportions as needed

# Title in Column 1
with col1:
    st.markdown('<h1 style="color: Cyan; font-size: 30px;">MarketMantra - A Stock Trend Predictor</h1>', unsafe_allow_html=True)

# QR Code in Column 2
with col2:
    st.image(qr_image, caption="scan for webite", width=100)  # Adjust size as needed
col1, col2 = st.columns(2)
with col2:
    st.subheader("~ Made By Jatan Shah")

with st.expander("Select Stock, Date Range and Technical Indicators ", expanded=True):
    stock_symbol = st.text_input("Enter Stock Ticker (e.g., AAPL)", value="JSWSTEEL.NS")
    start_date = st.date_input("Start Date", value=pd.to_datetime("2024-01-01"))
    end_date = st.date_input("End Date", value=datetime.now().date())
    
    selected_indicators = st.multiselect(
    "Select Technical Indicators to Display",
    indicator_options,
    default=["50-Day Simple Moving Average (SMA)", "200-Day Simple Moving Average (SMA)"]  # default selected indicators
)

# Sidebar for technical indicators
st.expander("Technical Indicators")
st.header("Technical Indicators")
indicator_options = [
    "50-Day Simple Moving Average (SMA)",
    "200-Day Simple Moving Average (SMA)",
    "MACD (Moving Average Convergence Divergence)",
    "Stochastic Oscillator"
]

if "50-Day Simple Moving Average (SMA)" in selected_indicators:
    sma_50 = True
else:
    sma_50 = False

if "200-Day Simple Moving Average (SMA)" in selected_indicators:
    sma_200 = True
else:
    sma_200 = False

if "MACD (Moving Average Convergence Divergence)" in selected_indicators:
    macd = True
else:
    macd = False

if "Stochastic Oscillator" in selected_indicators:
    stochastic = True
else:
    stochastic = False
    


# Sidebar setup for Portfolio & Watchlist Management
st.expander.header("Add to Portfolio & Watchlist")

# Select tab for adding stocks to portfolio or watchlist
portfolio_add = st.button("Add to Portfolio")
watchlist_add = st.button("Add to Watchlist")

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
# Split data
X_train, X_valid, Y_train, Y_valid = train_test_split(features_scaled, target, test_size=0.1, random_state=2500)

# Model Setup
models = {
    "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=15, random_state=45),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, learning_rate=10, max_depth=15, random_state=45),
    "XGBoost": xgb.XGBClassifier(n_estimators=100, max_depth=15, learning_rate=10, random_state=45),
    "Decision Tree": DecisionTreeClassifier(random_state=45)
}

# Initialize a list to store predictions
model_predictions = []

# Train models and store predictions
for model_name, model in models.items():
    with st.spinner(f"Training {model_name}..."):
        model.fit(X_train, Y_train)  # Train the model
        model_pred = model.predict(X_valid)  # Get predictions
        model_predictions.append(model_pred)  # Store predictions

# Convert list of predictions into a numpy array (shape: [n_models, n_samples])
model_predictions = np.array(model_predictions)

# Compute the average prediction (0 = Down, 1 = Up)
average_predictions = np.mean(model_predictions, axis=0)

# Round to get final prediction (0 or 1)
final_predictions = np.round(average_predictions)

# Calculate confusion matrix based on the averaged predictions
cm = confusion_matrix(Y_valid, final_predictions)

# Display confusion matrix
st.subheader("Confusion Matrix (Averaged Model Predictions)")
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

# display individual model accuracy
model_accuracies = {}
for model_name, model in models.items():
    model.fit(X_train, Y_train)
    y_pred = model.predict(X_valid)
    accuracy = accuracy_score(Y_valid, y_pred) * 100
    model_accuracies[model_name] = accuracy

# Select model to display accuracy
selected_model = st.selectbox("Select Model for Accuracy", list(models.keys()))
for model_name, accuracy in model_accuracies.items():
    if model_name == selected_model:
        st.write(f"{model_name}: {accuracy:.2f}%")
# Real-time Prediction Section
st.subheader("Real-time Prediction for the Latest Data")
latest_data = df.iloc[-1:][['Previous Close', 'Daily Return']].values.reshape(1, -1)
latest_data_scaled = scaler.transform(latest_data)
predicted_trend = models[selected_model].predict(latest_data_scaled)

#defining tabs
tab1, tab2, tab3, tab4 = st.tabs(["Portfolio", "Watchlist", "Technical indicators", "predictions"])

# Initialize portfolio and watchlist in session_state if they do not exist
if 'portfolio' not in st.session_state:
    st.session_state['portfolio'] = []

if 'watchlist' not in st.session_state:
    st.session_state['watchlist'] = []

# portfolio
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
with tab3:
    if sma_50:
        st.header("Simple Moving Average (SMA) of 50 Days")
        st.write("The **50-day** SMA looks at the average price over the last 50 days")
        df['SMA_50'] = df['Close'].rolling(window=200).mean()
        fig, ax = plt.subplots(figsize=(15, 5))
        ax.plot(df['SMA_50'], label="50-Day SMA", color='orange')
        ax.set_title(f"{stock_symbol} - 50-Day Simple Moving Average", fontsize=15)
        ax.set_ylabel('Price', fontsize=12)
        ax.set_xlabel('Date', fontsize=12)
        ax.legend(loc='best')
        st.pyplot(fig)

    if sma_200:
        st.header("Simple Moving Average (SMA) of 200 Days")
        st.write("The **200-day** SMA looks at the average price over the last 200 days")
        df['SMA_200'] = df['Close'].rolling(window=50).mean()
        fig, ax = plt.subplots(figsize=(15, 5))
        ax.plot(df['SMA_200'], label="200-Day SMA", color='green')
        ax.set_title(f"{stock_symbol} - 200-Day Simple Moving Average", fontsize=15)
        ax.set_ylabel('Price', fontsize=12)
        ax.set_xlabel('Date', fontsize=12)
        ax.legend(loc='best')
        st.pyplot(fig)

    if macd:
        st.header("MACD (Moving Average Convergence Divergence)")
        st.write("It helps us understand if the stock price is likely to go up or down.")
        st.write("**Its margings are:**")
        st.write("If the **MACD line** is **higher** than the **signal line**, it means the stock price could go **up**.")
        st.write("If the **MACD line** is **lower** than the **signal line**, it means the stock price could go **down**.")
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
        st.header("stochastic oscillator")
        st.write("It helps to see if a stock is high or low compared to its recent prices.")
        st.write("**Its margings are:**")
        st.write("If the value is above _**80**_, it might mean the stock is _**high (and could come down)**_.")
        st.write("If the value is below _**20**_, it might mean the stock is _**low (and could go up)**_.")
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

# Display recommendation with button and icon
with tab4:
   st.subheader("Predictions for Tomorrow's Trading")
   if predicted_trend == 1:
       st.write(":green[**Recommendation:** Hold/Buy the stock for tomorrow.]")
       st.write("**Stock price may go up**")
   else:
       st.write(":red[**Recommendation:** Sell the stock for tomorrow.]")
       st.write("**Stock price may go down**")
