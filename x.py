import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from datetime import datetime

# Title and Subheader
st.title("Stock Market Trend Predictor")
col1, col2 = st.columns(2)
with col2:
    st.subheader("~ Made By Jatan Shah")

# Sidebar for stock selection and date range
st.sidebar.header("Stock Selection")
stock_symbol = st.sidebar.selectbox("Select Stock ticker", ["JSWSTEEL.NS", "AAPL", "TSLA", "AMZN", "GOOGL", "MSFT"])

# Sidebar for date selection
st.sidebar.header("Date Range")
start_date = st.sidebar.date_input("Start date", pd.to_datetime("2024-01-01"))
end_date = st.sidebar.date_input("End date", datetime.now().date())

# Fetch stock data with progress indicator
try:
    df = yf.download(stock_symbol, start=start_date, end=end_date)
    df.drop(columns=['Adj Close'])
    if df.empty:
        st.warning("No data found for the selected stock or date range.")
        st.stop()
except Exception as e:
    st.error(f"Error fetching stock data: {e}")
    st.stop()


# Display stock data
st.subheader(f"Stock Data for {stock_symbol}")
st.write(df.tail())

# Section 1: Data Visualization
with st.expander("📊 Data Visualization"):
    st.subheader("Closing Price Over Time")
    fig, ax = plt.subplots(figsize=(15, 5))
    ax.plot(df['Close'], label='Close Price', color='blue')
    ax.set_title(f"{stock_symbol} - Closing Price History", fontsize=15)
    ax.set_ylabel('Price', fontsize=12)
    ax.set_xlabel('Date', fontsize=12)
    ax.grid(True)
    plt.legend()
    st.pyplot(fig)

# Feature Engineering for prediction
df['Previous Close'] = df['Close'].shift(1)
df['Daily Return'] = df['Close'].pct_change()

# Target variable
df['Target'] = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)

# Drop rows with NaN values
df.dropna(inplace=True)

if df.empty:
    st.warning("Not enough data after shifting for model to learn")
    st.stop()

# Features and scaling
features = df[['Previous Close', 'Daily Return']]
target = df['Target']
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Train-test split
X_train, X_valid, Y_train, Y_valid = train_test_split(features_scaled, target, test_size=0.1, random_state=2500)

# Define models
models = {
    "Logistic Regression": LogisticRegression(C=100.0),
    "Support Vector Classifier": SVC(C=100.0, kernel='linear', probability=True),
    "Random Forest Classifier": RandomForestClassifier(n_estimators=150, max_depth=50, random_state=150)
}

# Train models and make predictions
predictions = {}
for model_name, model in models.items():
    model.fit(X_train, Y_train)
    predictions[model_name] = model.predict(X_valid)

# Average predictions
avg_predictions = np.mean(list(predictions.values()), axis=0)
final_predictions = np.where(avg_predictions >= 0.5, 1, 0)

# Section 2: Model Predictions
with st.expander("Model Predictions"):
    st.subheader("Model Accuracy")
    for model_name in models.keys():
        valid_accuracy = metrics.accuracy_score(Y_valid, predictions[model_name]) * 100
        st.write(f"{model_name} Validation Accuracy: {valid_accuracy:.2f}%")

    # Confusion Matrix
    st.subheader("Average Confusion Matrix")
    avg_cm = confusion_matrix(Y_valid, final_predictions)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(avg_cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(ax.imshow(avg_cm, interpolation='nearest', cmap=plt.cm.Blues))
    classes = ['Down', 'Up']
    ax.set(xticks=np.arange(len(classes)),
           yticks=np.arange(len(classes)),
           xticklabels=classes, yticklabels=classes,
           title="Average Matrix",
           ylabel="True Label", xlabel="Predicted Label")
    for i in range(avg_cm.shape[0]):
        for j in range(avg_cm.shape[1]):
            ax.text(j, i, format(avg_cm[i, j], 'd'), ha="center", va="center", color="black")
    st.pyplot(fig)

# Section 3: Real-time Stock Data and Prediction
with st.expander("Real-time Data & Prediction"):
    # Store prediction history
    if 'prediction_history' not in st.session_state:
        st.session_state.prediction_history = []

    # Latest Prediction
    latest_data = df.iloc[-1][['Previous Close', 'Daily Return']].values.reshape(1, -1)
    if latest_data.size == 0:
        st.error("No data available for making predictions.")
    else:
        latest_data_scaled = scaler.transform(latest_data)
        latest_predictions = [model.predict(latest_data_scaled) for model in models.values()]
        latest_avg_prediction = np.mean(latest_predictions, axis=0)
        latest_final_prediction = 1 if latest_avg_prediction >= 0.5 else 0
        
        # Store the latest prediction
        st.session_state.prediction_history.append(latest_final_prediction)
        
        # Show recommendation
        st.subheader("Recommendation for Tomorrow")
        st.write(f"Prediction Trend for {stock_symbol}", fontsize=15)
        if latest_final_prediction == 1:
            st.write("The model recommends to **Hold** the stock.")
        else:
            st.write("The model recommends to **Sell** the stock.")

    # Real-time stock price
    try:
        current_stock = yf.Ticker(stock_symbol)
        latest_market_price = current_stock.history(period='1d')['Close'].iloc[-1]
        st.subheader("Real-time Market Price")
        st.markdown(f"The real-time market price of **{stock_symbol}** is :green[**{latest_market_price:.2f}**].")
    except Exception as e:
        st.error(f"Error fetching real-time stock price: {e}")
