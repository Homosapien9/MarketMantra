import numpy as np
import pandas as pd
import xgboost as xgb
import yfinance as yf
from PIL import Image
import streamlit as st
from datetime import datetime
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

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

 # Bollinger Bands
def compute_rsi(df, window=14):
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Function to plot the RSI graph
def plot_rsi(df, window=14):
    df['RSI'] = compute_rsi(df, window)
    
    # Plot the RSI graph
    fig, ax = plt.subplots(figsize=(15, 5))
    ax.plot(df.index, df['RSI'], label="RSI", color="blue")
    ax.axhline(70, color='red', linestyle='--', label="Overbought (70)")
    ax.axhline(30, color='green', linestyle='--', label="Oversold (30)")
    ax.set_title('Relative Strength Index (RSI)', fontsize=15)
    ax.set_xlabel('Date')
    ax.set_ylabel('RSI Value')
    ax.legend(loc="upper left")
    st.pyplot(fig)

def compute_bollinger_bands(df, window=20):
    df['Middle_BB'] = df['Close'].rolling(window=window).mean()  # Middle Band (SMA)
    df['Std_Dev'] = df['Close'].rolling(window=window).std()  # Standard deviation
    df['Upper_BB'] = df['Middle_BB'] + (df['Std_Dev'] * 2)  # Upper Band
    df['Lower_BB'] = df['Middle_BB'] - (df['Std_Dev'] * 2)  # Lower Band
    return df

# Function to plot the Bollinger Bands graph
def plot_bollinger_bands(df, window=20):
    df = compute_bollinger_bands(df, window)
    
    # Plot the Bollinger Bands graph
    fig, ax = plt.subplots(figsize=(15, 5))
    ax.plot(df.index, df['Close'], label='Close Price', color='blue')
    ax.plot(df.index, df['Upper_BB'], label='Upper Bollinger Band', color='red', linestyle='--')
    ax.plot(df.index, df['Middle_BB'], label='Middle Bollinger Band (SMA)', color='orange', linestyle='--')
    ax.plot(df.index, df['Lower_BB'], label='Lower Bollinger Band', color='green', linestyle='--')
    ax.fill_between(df.index, df['Upper_BB'], df['Lower_BB'], color='gray', alpha=0.2)  # Fill between bands
    ax.set_title('Bollinger Bands', fontsize=15)
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    ax.legend(loc='upper left')
    st.pyplot(fig)

def compute_volumetric_data(df):
    df['Buy_Volume'] = df['Volume'].where(df['Close'] > df['Open'], 0)  # Volume on upward price movement
    df['Sell_Volume'] = df['Volume'].where(df['Close'] <= df['Open'], 0)  # Volume on downward price movement
    return df
        
qr_image = Image.open("Website qr.png")
col1, col2 = st.columns([3, 1])

# Title in Column 1
with col1:
    st.markdown('<h1 style="color: white; font-size: 29.7px;">MarketMantra - An Asset Trend Predictor</h1>', unsafe_allow_html=True)
    st.subheader("~ Developed By Jatan Shah")
# QR Code in Column 2
with col2:
    st.image(qr_image, caption="scan for webite", width=100)

with st.expander("Select Asset And Data Range(Minimum 5 Days Gap)"):
    st.header("Asset Selection")
    stock_symbol = st.text_input("Select asset Ticker", value="JSWSTEEL.NS")
    stock_symbol = stock_symbol.upper()
    start_date = st.date_input("Start Date", pd.to_datetime("2024-01-01"))
    end_date = st.date_input("End Date", datetime.now().date())
with st.expander("Select Technical Indicators"):
    st.header("Technical Indicators")
    indicator_options = ["50-Day Simple Moving Average (SMA)","200-Day Simple Moving Average (SMA)","MACD (Moving Average Convergence Divergence)","Stochastic Oscillator","Bollinger Bands","(RSI)Relative Strength Index","Volume Chart"]
    selected_indicators = st.multiselect("Select Technical Indicators to Display",indicator_options,default=["50-Day Simple Moving Average (SMA)", "200-Day Simple Moving Average (SMA)",]  )

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

if "Bollinger Bands" in selected_indicators:
    Bollingers = True
else:
    Bollingers = False

if "(RSI)Relative Strength Index" in selected_indicators:
    RSI = True
else:
    RSI = False

if "Volume Chart" in selected_indicators:
    volume = True
else:
    volume = False

# Data Visualization: Closing Price
with st.expander("Data Visualization"):
    # Fetch stock data
    df = get_stock_data(stock_symbol, start_date, end_date)
    st.subheader(f"asset Data for {stock_symbol}")
    st.write(f"Historical data for {stock_symbol} from {start_date} to {end_date}, in its listed currency")
    st.dataframe(df.tail())
    if df.empty:
        st.warning("No data found for the selected asset or date range. Model needs atleast 5 days to predict results")
        st.stop()
        
    st.subheader("Closing Price Over Time")
    fig, ax = plt.subplots(figsize=(15, 5))
    ax.plot(df['Close'], label='Close Price', color='blue')
    ax.set_title(f"{stock_symbol} - Closing Price History", fontsize=15)
    ax.set_ylabel('Price', fontsize=12)
    ax.set_xlabel('Date', fontsize=12)
    ax.grid(True)
    plt.legend()
    st.pyplot(fig)

st.header("Portfolio & Watchlist")

col1, col2, col3, col4,= st.columns(4)
with col1:
    portfolio_add = st.button("Add to Portfolio")
with col2:
    watchlist_add = st.button("Add to Watchlist")

# Add stock to portfolio
if portfolio_add:
    if stock_symbol not in st.session_state['portfolio']:
        st.session_state['portfolio'].append(stock_symbol)
        st.success(f"{stock_symbol} added to Portfolio.")
    else:
        st.warning(f"{stock_symbol} is already in your Portfolio.")

# Add stock to watchlist
if watchlist_add:
    if stock_symbol not in st.session_state['watchlist']:
        st.session_state['watchlist'].append(stock_symbol)
        st.success(f"{stock_symbol} added to Watchlist.")
    else:
        st.warning(f"{stock_symbol} is already in your Watchlist.")
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
    "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=20, random_state=50),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, learning_rate=20, max_depth=20, random_state=50),
    "XGBoost": xgb.XGBClassifier(n_estimators=100, max_depth=20, learning_rate=20, random_state=50),
    "Decision Tree": DecisionTreeClassifier(random_state=50)}

# Initialize a list to store predictions
model_predictions = []

# Train models and store predictions
for model_name, model in models.items():
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

latest_data = df.iloc[-1:][['Previous Close', 'Daily Return']].values.reshape(1, -1)
latest_data_scaled = scaler.transform(latest_data)
predicted_trend = models[selected_model].predict(latest_data_scaled)

tab1, tab2, tab3, tab4 ,tab5= st.tabs(["Portfolio", "Watchlist", "Technical indicators", "Predictions", "calculate ROI"])

# Initialize portfolio and watchlist in session_state if they do not exist
if 'portfolio' not in st.session_state:
    st.session_state['portfolio'] = []

if 'watchlist' not in st.session_state:
    st.session_state['watchlist'] = []

# Portfolio Tab
with tab1:
    st.subheader("Manage Your Portfolio")

    # Display all stocks in the portfolio
    if st.session_state['portfolio']:
        for stock in st.session_state['portfolio']:
            st.write(f"- {stock}")

            # Button to remove stock from portfolio
            if st.button(f"Remove {stock} from Portfolio"):
                st.session_state['portfolio'].remove(stock)
                st.success(f"Removed {stock} from Portfolio!")
    else:
        st.write("No assets in your portfolio yet.")
        
# Watchlist Tab
with tab2:
    st.subheader("Manage Your Watchlist")

    # Display all stocks in the watchlist
    if st.session_state['watchlist']:
        for stock in st.session_state['watchlist']:
            st.write(f"- {stock}")

            # Button to remove stock from watchlist
            if st.button(f"Remove {stock} from Watchlist"):
                st.session_state['watchlist'].remove(stock)
                st.success(f"Removed {stock} from Watchlist!")
    else:
        st.write("No assets in your watchlist yet.")

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
        st.write("It helps us understand if the asset price is likely to go up or down.")
        st.write("If the **MACD line** is **higher** than the **signal line**, it means the asset price could go **up**.")
        st.write("If the **MACD line** is **lower** than the **signal line**, it means the asset price could go **down**.")
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
        st.write("It helps to see if a asset is high or low compared to its recent prices.")
        st.write("If the value is above _**80**_, it might mean the asset is _**high (and could come down)**_.")
        st.write("If the value is below _**20**_, it might mean the asset is _**low (and could go up)**_.")
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

    if Bollingers:
        st.subheader("Bollinger Bands")
        st.write("Bollinger Bands show the volatility and price range of a asset.")
        st.write("If the price hits or exceeds the **upper band**, signaling a potential **sell**.")
        st.write("If the price hits or drops below the **lower band**, signaling a potential **buy**.")
        plot_bollinger_bands(df)

    if RSI:
        st.subheader("Relative Strength Index (RSI)")
        st.write("RSI shows if a asset is overbought or oversold.")
        st.write("RSI above _**70**_ means that it's a good time to _**sell**_ the stock.")
        st.write("RSI below _**30**_ means that it's a good time to _**buy**_ the stock.")
        plot_rsi(df)

    if volume:
        st.subheader("Volume Chart")
        df['Date'] = df.index
        df.set_index('Date', inplace=True)
        def compute_volumetric_data(df):
            df['Buy_Volume'] = df['Volume'].where(df['Close'] > df['Open'], 0)  # Buy volume
            df['Sell_Volume'] = df['Volume'].where(df['Close'] <= df['Open'], 0)  # Sell volume
            return df
        
        def plot_volumetric_chart(df):
            st.write("Volume chart tracks the number of shares/contracts traded.")
            st.write("High volume: Confirms price trends (up or down).")
            st.write("Low volume: Signals lack of interest or indecision.")
            # Compute volumetric data
            df = compute_volumetric_data(df)

            # Create the plot
            fig, ax = plt.subplots(figsize=(15, 7))

            # Plot buying pressure
            ax.bar(df.index, df['Buy_Volume'], color='green', alpha=0.6, label='Buying Pressure')
            # Plot selling pressure
            ax.bar(df.index, df['Sell_Volume'], color='red', alpha=0.6, label='Selling Pressure')

            # Add chart title and labels
            ax.set_title('Volumetric Chart: Buying vs Selling Pressure', fontsize=15)
            ax.set_xlabel('Date')
            ax.set_ylabel('Volume')
            ax.legend(loc='upper left')

            # Display the plot in Streamlit
            st.pyplot(fig)
        plot_volumetric_chart(df)
        
with tab4:
    st.subheader("Predictions for Tomorrow's Trading")

    try:
        # Calculate probabilities for tomorrow's prediction
        all_model_predictions = [model.predict_proba(latest_data_scaled) for model in models.values()]

        # Check dimensions of each prediction
        for idx, prediction in enumerate(all_model_predictions):
            if len(prediction.shape) == 1 or prediction.shape[1] != 2:
                st.error(f"Model {list(models.keys())[idx]} returned unexpected shape: {prediction.shape}")
                st.stop()

        # Average probabilities across models
        avg_probabilities = np.mean(all_model_predictions, axis=0)

        # Ensure avg_probabilities shape is valid
        if avg_probabilities.shape[0] != 1:
            st.error(f"Unexpected shape of avg_probabilities: {avg_probabilities.shape}")
            st.stop()

        avg_prob_up = avg_probabilities[0, 1]
        avg_prob_down = avg_probabilities[0, 0]

        # Display predictions
        if avg_prob_up > avg_prob_down:
            st.write(":green[The asset is likely to rise tomorrow.]")
            st.metric(label="Probability (Up)", value=f"{avg_prob_up * 100:.2f}%")
        else:
            st.write(":red[The asset is likely to fall tomorrow.]")
            st.metric(label="Probability (Down)", value=f"{avg_prob_down * 100:.2f}%")

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

with tab5:
        def fetch_stock_data(stock_ticker, start_date):
            stock_data = yf.Ticker(stock_ticker)
            try:
                # Fetch data from start_date till today
                hist = stock_data.history(start=start_date, end=datetime.today().strftime('%Y-%m-%d'))
                if hist.empty:
                    return None, f"No data available for the asset '{stock_ticker}' from {start_date}."
                else:
                    return hist, None
            except Exception as e:
                return None, f"Failed to fetch data for '{stock_ticker}': {str(e)}"
        
        # Function to calculate investment return
        def calculate_investment_return(start_date, stock_ticker, investment_amount):
            # Fetch stock data
            hist, error = fetch_stock_data(stock_ticker, start_date)
            if error:
                st.error(error)  # Display error if there's a problem fetching the data
                return
        
            # Get the price at the start date
            try:
                start_price = hist.loc[start_date]["Close"]
            except KeyError:
                st.error(f"Start date {start_date} is not available in the historical data.")
                return
            
            # Get the current price (most recent close price)
            current_price = hist["Close"].iloc[-1]
            
            # Calculate total dividends (if available)
            total_dividends = hist["Dividends"].sum()
        
            # Final value considering price change and dividends
            final_value = (investment_amount / start_price) * current_price + total_dividends
        
            # Calculate total return and return percentage
            total_return = final_value - investment_amount
            return_percentage = (total_return / investment_amount) * 100
        
            # Display the results directly using Streamlit components
            st.subheader(f"Investment in {stock_ticker} from {start_date}")
            st.write(f"Initial Investment: {investment_amount:,.2f}")
            st.write(f"Start Price: {start_price:,.2f}")
            st.write(f"Current Price: {current_price:,.2f}")
            
            # Dividend Details
            if total_dividends > 0:
                st.write(f"Total Dividends Earned: {total_dividends:,.2f}")
            else:
                st.write("This stock does not offer dividends or no dividends were paid during the selected period.")
            
            # Final Value and Return Calculation
            st.write(f"Final Value (including price change and dividends): {final_value:,.2f}")
            st.write(f"Total Return: {total_return:,.2f}")
            st.write(f"Return Percentage: {return_percentage:,.2f}%")
        
        # Streamlit UI components
        st.title("Asset Investment Return Calculator")
        
        # User inputs
        stock_ticker = stock_symbol
        start_date = st.date_input("Enter Start Date", pd.to_datetime("2016-01-01"))
        investment_amount = st.number_input("Enter Investment Amount", min_value=1, value=1000000)
        
        # Ensure the start date doesn't exceed today's date
        if start_date > datetime.today().date():
            st.warning("Start date cannot be in the future. Using today's date instead.")
            start_date = datetime.today().date()
        
        # Automatically calculate investment return when inputs are provided
        if stock_ticker and start_date and investment_amount:
            calculate_investment_return(start_date.strftime('%Y-%m-%d'), stock_ticker, investment_amount)
