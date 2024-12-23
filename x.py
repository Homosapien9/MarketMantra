import requests
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

def compute_macd(df, fast=12, slow=26, signal=9):# Helper function to calculate MACD
    macd_line = df['Close'].ewm(span=fast, adjust=False).mean() - df['Close'].ewm(span=slow, adjust=False).mean()
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line

def compute_stochastic(df, window=14):# Helper function to calculate Stochastic Oscillator
    low_min = df['Low'].rolling(window=window).min()
    high_max = df['High'].rolling(window=window).max()
    stochastic = 100 * (df['Close'] - low_min) / (high_max - low_min)
    return stochastic

def compute_rsi(df, window=14): # Bollinger Bands
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def plot_rsi(df, window=14):# Function to plot the RSI graph
    df['RSI'] = compute_rsi(df, window)
    fig, ax = plt.subplots(figsize=(15, 5))  # Plot the RSI graph
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
    
def plot_bollinger_bands(df, window=20):# Function to plot the Bollinger Bands graph
    df = compute_bollinger_bands(df, window)   # Plot the Bollinger Bands graph
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
with col1:
    st.markdown('<h1 style="color: white; font-size: 29.7px;">MarketMantra - Stock Trend Predictor</h1>', unsafe_allow_html=True)
    st.subheader("~ Developed By Jatan Shah")
with col2:
    st.image(qr_image, caption="scan for webite", width=100)

with st.expander("Select Stock And Data Range(Minimum 5 Days Gap)"):
    st.header("Stock Selection")
    def get_stock_data(stock_symbol, start_date, end_date):
        try:
            # Fetch historical stock data
            df = yf.download(stock_symbol, start=start_date, end=end_date)
            if df.empty:
                raise ValueError(f"No data found for {stock_symbol} between {start_date} and {end_date}.")
            return df
        except Exception as e:
            st.error(f"Error fetching stock data for {stock_symbol}: {e}")
            return pd.DataFrame()
    
    # Function to fetch metadata for a single stock
    def fetch_stock_metadata(stock_symbol):
        try:
            ticker = yf.Ticker(stock_symbol)
            info = ticker.info
            return {
                "symbol": stock_symbol,
                "name": info.get("shortName", "Unknown"),
                "sector": info.get("sector", "Unknown"),
                "industry": info.get("industry", "Unknown"),
            }
        except Exception:
            return {"symbol": stock_symbol, "name": "Unknown", "sector": "Unknown", "industry": "Unknown"}
    
    # Function to recommend similar stocks based on sector and industry
    def recommend_stocks(input_stock):
        input_metadata = fetch_stock_metadata(input_stock)
    
        if input_metadata["sector"] == "Unknown":
            return f"Could not fetch metadata for {input_stock}. Please check the symbol."
    
        try:
            tickers = yf.Tickers()  # Fetch all tickers dynamically
            recommendations = []
    
            for symbol in tickers.tickers:
                metadata = fetch_stock_metadata(symbol)
                if (
                    metadata["sector"] == input_metadata["sector"]
                    and metadata["industry"] == input_metadata["industry"]
                    and metadata["symbol"] != input_stock
                ):
                    recommendations.append(metadata)
    
            return pd.DataFrame(recommendations) if recommendations else None
        except Exception as e:
            return f"Error fetching recommendations: {str(e)}"
    
    # Function to search stocks by partial keyword
    def search_stocks(keyword):
        keyword = keyword.upper()
        try:
            tickers = yf.Tickers()  # Fetch all tickers
            matched_stocks = []
    
            for symbol in tickers.tickers:
                metadata = fetch_stock_metadata(symbol)
                if keyword in metadata["symbol"] or keyword in metadata["name"].upper():
                    matched_stocks.append(metadata)
    
            return pd.DataFrame(matched_stocks) if matched_stocks else None
        except Exception as e:
            return f"Error fetching stock data: {str(e)}"
    
    # Streamlit UI
    st.title("Stock Data and Recommendation System")
    
    # User input for stock data
    stock_symbol = st.text_input("Enter a stock symbol (e.g., AAPL, JSWSTEEL.NS):")
    start_date = st.date_input("Start Date", datetime(2023, 1, 1), key="start_date_input")
    end_date = st.date_input("End Date", datetime.now(), key="end_date_input")
    
    if stock_symbol:
        # Fetch and display stock data
        df = get_stock_data(stock_symbol, start_date, end_date)
        if not df.empty:
            st.subheader(f"Stock Data for {stock_symbol}")
            st.write(f"Historical data for {stock_symbol} from {start_date} to {end_date}")
            st.dataframe(df.tail())
        else:
            st.warning(f"No data found for {stock_symbol} in the selected date range.")
    
        # Fetch and display recommendations
        st.subheader("Recommended Stocks")
        recommendations = recommend_stocks(stock_symbol)
        if isinstance(recommendations, str):
            st.write(recommendations)
        elif recommendations is None:
            st.write(f"No similar stocks found for {stock_symbol}.")
        else:
            st.write("Here are some similar stocks:")
            st.dataframe(recommendations)
    
        # Search stocks dynamically
        st.subheader("Search Stocks by Keyword")
        keyword = st.text_input("Enter a keyword to search for stocks (e.g., JSW, RELIANCE):")
        if keyword:
            search_results = search_stocks(keyword)
            if isinstance(search_results, str):
                st.write(search_results)
            elif search_results is None:
                st.write(f"No stocks found matching '{keyword}'.")
            else:
                st.write("Matched Stocks:")
                st.dataframe(search_results)

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

with st.expander("Data Visualization"):# Data Visualization: Closing Price
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

if portfolio_add:# Add stock to portfolio
    if stock_symbol not in st.session_state['portfolio']:
        st.session_state['portfolio'].append(stock_symbol)
        st.success(f"{stock_symbol} added to Portfolio.")
    else:
        st.warning(f"{stock_symbol} is already in your Portfolio.")

if watchlist_add:# Add stock to watchlist
    if stock_symbol not in st.session_state['watchlist']:
        st.session_state['watchlist'].append(stock_symbol)
        st.success(f"{stock_symbol} added to Watchlist.")
    else:
        st.warning(f"{stock_symbol} is already in your Watchlist.")

df['Previous Close'] = df['Close'].shift(1)# Feature Engineering and Model Preparation
df['Daily Return'] = df['Close'].pct_change()
df['Target'] = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)  # Binary target for up/down trend
df.dropna(inplace=True)
features = df[['Previous Close', 'Daily Return']].values# Prepare features and target
target = df['Target'].values

scaler = StandardScaler()# Scaling features
features_scaled = scaler.fit_transform(features)
X_train, X_valid, Y_train, Y_valid = train_test_split(features_scaled, target, test_size=0.1, random_state=2500)# Split data

models = {"Random Forest": RandomForestClassifier(n_estimators=100, max_depth=20, random_state=50),
          "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, learning_rate=20, max_depth=20, random_state=50),
          "XGBoost": xgb.XGBClassifier(n_estimators=100, max_depth=20, learning_rate=20, random_state=50),
          "Decision Tree": DecisionTreeClassifier(random_state=50)}# Model Setup
model_predictions = []# Initialize a list to store predictions

for model_name, model in models.items():# Train models and store predictions
    model.fit(X_train, Y_train)  # Train the model
    model_pred = model.predict(X_valid)  # Get predictions
    model_predictions.append(model_pred)  # Store predictions

model_predictions = np.array(model_predictions)# Convert list of predictions into a numpy array (shape: [n_models, n_samples])
average_predictions = np.mean(model_predictions, axis=0)# Compute the average prediction (0 = Down, 1 = Up)
final_predictions = np.round(average_predictions)# Round to get final prediction (0 or 1)

cm = confusion_matrix(Y_valid, final_predictions)# Calculate confusion matrix based on the averaged predictions
st.subheader("Confusion Matrix")# Display confusion matrix
fig, ax = plt.subplots(figsize=(6, 6))
cax = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
fig.colorbar(cax)
classes = ['Down', 'Up']
ax.set(xticks=np.arange(len(classes)),
       yticks=np.arange(len(classes)),
       xticklabels=classes, yticklabels=classes,
       title="Confusion Matrix",
       ylabel="True Value", xlabel="Predicted Value")
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j, i, format(cm[i, j], 'd'), ha="center", va="center", color="black")
st.pyplot(fig)

model_accuracies = {}# display individual model accuracy
for model_name, model in models.items():
    model.fit(X_train, Y_train)
    y_pred = model.predict(X_valid)
    accuracy = accuracy_score(Y_valid, y_pred) * 100
    model_accuracies[model_name] = accuracy

selected_model = st.selectbox("Select Model for Accuracy", list(models.keys()))# Select model to display accuracy
for model_name, accuracy in model_accuracies.items():
    if model_name == selected_model:
        st.write(f"{model_name}: {accuracy:.2f}%")

latest_data = df.iloc[-1:][['Previous Close', 'Daily Return']].values.reshape(1, -1)
latest_data_scaled = scaler.transform(latest_data)
predicted_trend = models[selected_model].predict(latest_data_scaled)

tab1, tab2, tab3, tab4 , tab5, tab6= st.tabs(["Portfolio", "Watchlist", "Technical indicators", "Predictions", "calculate ROI","Investment Chatbot"])

if 'portfolio' not in st.session_state:# Initialize portfolio and watchlist in session_state if they do not exist
    st.session_state['portfolio'] = []
if 'watchlist' not in st.session_state:
    st.session_state['watchlist'] = []

with tab1:# Portfolio Tab
    st.subheader("Manage Your Portfolio")
    if st.session_state['portfolio']:# Display all stocks in the portfolio
        for stock in st.session_state['portfolio']:
            st.write(f"- {stock}")
            if st.button(f"Remove {stock} from Portfolio"):# Button to remove stock from portfolio
                st.session_state['portfolio'].remove(stock)
                st.success(f"Removed {stock} from Portfolio!")
    else:
        st.write("No Stocks in your portfolio yet.")
        
with tab2:# Watchlist Tab
    st.subheader("Manage Your Watchlist")
    if st.session_state['watchlist']:# Display all stocks in the watchlist
        for stock in st.session_state['watchlist']:
            st.write(f"- {stock}")
            if st.button(f"Remove {stock} from Watchlist"):# Button to remove stock from watchlist
               st.session_state['watchlist'].remove(stock)
               st.success(f"Removed {stock} from Watchlist!")
    else:
        st.write("No Stocks in your watchlist yet.")

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
        st.write("It helps to see if a stock is high or low compared to its recent prices.")
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

    if Bollingers:
        st.subheader("Bollinger Bands")
        st.write("Bollinger Bands show the volatility and price range of a asset.")
        st.write("If the price hits or exceeds the **upper band**, signaling a potential **sell**.")
        st.write("If the price hits or drops below the **lower band**, signaling a potential **buy**.")
        plot_bollinger_bands(df)

    if RSI:
        st.subheader("Relative Strength Index (RSI)")
        st.write("RSI shows if a stock is overbought or oversold.")
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
            df = compute_volumetric_data(df)# Compute volumetric data
            fig, ax = plt.subplots(figsize=(15, 7))# Create the plot
            ax.bar(df.index, df['Buy_Volume'], color='green', alpha=0.6, label='Buying Pressure')# Plot buying pressure
            ax.bar(df.index, df['Sell_Volume'], color='red', alpha=0.6, label='Selling Pressure')# Plot selling pressure
            ax.set_title('Volumetric Chart: Buying vs Selling Pressure', fontsize=15)# Add chart title and labels
            ax.set_xlabel('Date')
            ax.set_ylabel('Volume')
            ax.legend(loc='upper left')
            st.pyplot(fig)
        plot_volumetric_chart(df)
        
with tab4:
    st.subheader("Predictions For Next Day's Trading")
    try:
        all_model_predictions = [model.predict_proba(latest_data_scaled) for model in models.values()]

        for idx, prediction in enumerate(all_model_predictions):# Check dimensions of each prediction
            if len(prediction.shape) == 1 or prediction.shape[1] != 2:
                st.error(f"Model {list(models.keys())[idx]} returned unexpected shape: {prediction.shape}")
                st.stop()

        avg_probabilities = np.mean(all_model_predictions, axis=0)   # Average probabilities across models
        if avg_probabilities.shape[0] != 1:# Ensure avg_probabilities shape is valid
            st.error(f"Unexpected shape of avg_probabilities: {avg_probabilities.shape}")
            st.stop()

        avg_prob_up = avg_probabilities[0, 1]
        avg_prob_down = avg_probabilities[0, 0]

        if avg_prob_up > avg_prob_down:# Display predictions
            st.write(":green[The stock is likely to rise next working day.]")
            st.metric(label="Probability (Up)", value=f"{avg_prob_up * 100:.2f}%")
        else:
            st.write(":red[The stock is likely to fall next working day.]")
            st.metric(label="Probability (Down)", value=f"{avg_prob_down * 100:.2f}%")

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

with tab5:
        def fetch_stock_data(stock_ticker, start_date):
            stock_data = yf.Ticker(stock_ticker)
            try:
                hist = stock_data.history(start=start_date, end=datetime.today().strftime('%Y-%m-%d'))
                if hist.empty:
                    return None, f"No data available for the stock '{stock_ticker}' from {start_date}."
                else:
                    return hist, None
            except Exception as e:
                return None, f"Failed to fetch data for '{stock_ticker}': {str(e)}"
        
        def calculate_investment_return(start_date, stock_ticker, investment_amount):
            hist, error = fetch_stock_data(stock_ticker, start_date)
            if error:
                st.error(error)  # Display error if there's a problem fetching the data
                return
                
            try:
                start_price = hist.loc[start_date]["Close"]
            except KeyError:
                st.error(f"Start date {start_date} is not available in the historical data.")
                return
            
            current_price = hist["Close"].iloc[-1] # Get the current price (most recent close price)
            total_dividends = hist["Dividends"].sum()
            final_value = (investment_amount / start_price) * current_price + total_dividends
            total_return = final_value - investment_amount
            return_percentage = (total_return / investment_amount) * 100
        
            st.subheader(f"Investment in {stock_ticker} from {start_date}") # Display the results directly using Streamlit components
            st.write(f"Initial Investment: {investment_amount:,.2f}")
            st.write(f"Start Price: {start_price:,.2f}")
            st.write(f"Current Price: {current_price:,.2f}")
         
            if total_dividends > 0:   # Dividend Details
                st.write(f"Total Dividends Earned: {total_dividends:,.2f}")
            else:
                st.write("This stock does not offer dividends or no dividends were paid during the selected period.")
            
            st.write(f"Final Value (including price change and dividends): {final_value:,.2f}")# Final Value and Return Calculation
            st.write(f"Total Return: {total_return:,.2f}")
            st.write(f"Return Percentage: {return_percentage:,.2f}%")
        
        st.title("Stock Investment Return Calculator")
        stock_ticker = stock_symbol# User inputs
        start_date = st.date_input("Enter Start Date", pd.to_datetime("2016-01-01"))
        investment_amount = st.number_input("Enter Investment Amount", min_value=1, value=1000000)
    
        if start_date > datetime.today().date():# Ensure the start date doesn't exceed today's date
            st.warning("Start date cannot be in the future. Using today's date instead.")
        
        if stock_ticker and start_date and investment_amount:# Automatically calculate investment return when inputs are provided
            calculate_investment_return(start_date.strftime('%Y-%m-%d'), stock_ticker, investment_amount)
with tab6:
    def get_stock_price(stock_symbol):# Function to fetch stock price from Yahoo Finance
        try:
            url = f"https://query1.finance.yahoo.com/v7/finance/quote?symbols={stock_symbol}"
            response = requests.get(url)
            response.raise_for_status()  # Raise HTTP error for bad responses (4xx, 5xx)
            
            data = response.json()
            
            # Check if the stock data is available
            if 'quoteResponse' in data and 'result' in data['quoteResponse'] and len(data['quoteResponse']['result']) > 0:
                stock_data = data['quoteResponse']['result'][0]
                price = stock_data['regularMarketPrice']
                return price
            else:
                return None
        except requests.exceptions.RequestException as e:
            return f"Error fetching stock data: {str(e)}"
        except ValueError:
            return "Error parsing the response from the server."
    
    def get_investment_info(query):
        query = query.lower()
        if 'price' in query or 'what is the price' in query:# Extract stock symbol (e.g., JSWSTEEL.NS)
            words = query.split()
            for word in words:
                if '.' in word:  # Looks like a valid stock symbol
                    price = get_stock_price(word.upper())
                    if price is not None:
                        return f"The current price of {word.upper()} is {price} USD."
                    else:
                        return f"Sorry, I couldn't find the stock price for {word.upper()}. Please check the symbol."
    
        elif 'compare' in query or 'which is better' in query or 'compare price' in query:# Extract stock symbols from the query after "compare"
            if 'compare' in query:
                query_part = query.split("compare")[1]  # Get part after "compare"
                stock_symbols = [symbol.strip() for symbol in query_part.split("and")]  # Split by 'and'
    
                stock_prices = {}
                for symbol in stock_symbols:
                    price = get_stock_price(symbol.upper())
                    if price is not None:
                        stock_prices[symbol.upper()] = price
                    else:
                        stock_prices[symbol.upper()] = "Not found or invalid symbol"
                
                return stock_prices# Return the stock comparison result
        investment_terms = {
            "what is bond": "A bond is when you lend money to someone, like the government or a company, and they pay you back with interest after a while.",
            "what is stock market": "The stock market is where people buy and sell pieces of companies, called stocks.",
            "what is mutual fund": "A mutual fund is a pool of money collected from many investors, managed by professionals to invest in different assets like stocks and bonds.",
            "what is roi": "ROI means Return on Investment. It's a way to measure how much profit you made relative to the cost of your investment.",
            "what is diversification": "Diversification means spreading your investments across different areas to reduce risk. Don't put all your eggs in one basket.",
            "what is portfolio management": "Portfolio management is the art of choosing and managing the best mix of investments to achieve your financial goals.",
            "what is etf": "An ETF, or exchange-traded fund, is like a mutual fund, but it trades on the stock exchange like a regular stock.",
            "what is cryptocurrency": "Cryptocurrency is a type of digital or virtual currency that uses encryption techniques to regulate the generation of units and verify the transfer of funds.",
            "what is bitcoin": "Bitcoin is the first and most popular cryptocurrency. It's decentralized and uses blockchain technology for secure transactions.",
            "what is inflation": "Inflation is the rate at which the general level of prices for goods and services rises, and subsequently, the purchasing power of currency falls.",
            "what is interest rate": "An interest rate is the cost of borrowing money, typically expressed as a percentage of the principal loan amount, paid periodically.",
            "what is asset": "An asset is something of value or a resource that can provide future economic benefits, like property, stocks, or bonds.",
            "what is hedge fund": "A hedge fund is a pooled investment fund that uses a range of strategies to earn high returns for its investors, often with high risk.",
            "what is ipo": "An IPO, or Initial Public Offering, is when a company offers its shares to the public for the first time, usually to raise capital.",
            "what is commodity": "A commodity is a basic good used in commerce that is interchangeable with other goods of the same type, like gold, oil, or wheat.",
            "what is real estate investment": "Real estate investment involves buying, owning, managing, and/or renting property for profit. It can generate regular income or long-term gains.",
            "what is savings account": "A savings account is a bank account that earns interest on your deposits, typically used for short-term or emergency savings.",
            "what is 401k": "A 401(k) is a retirement savings plan offered by employers that allows workers to save and invest a portion of their paycheck before taxes.",
            "what is dividend": "A dividend is a payment made by a corporation to its shareholders, usually out of profits, in the form of cash or additional shares.",
            "what is stock split": "A stock split occurs when a company issues additional shares to shareholders, increasing the total supply while keeping the overall value the same.",
            "what is bear market": "A bear market is a period when the prices of securities are falling or are expected to fall, typically by 20% or more from recent highs.",
            "what is bull market": "A bull market is when the prices of securities are rising or are expected to rise, often driven by investor confidence and economic growth.",
            "what is private equity": "Private equity is capital invested in companies that are not listed on a public exchange. It's often used for startup financing or buyouts.",
            "what is credit rating": "A credit rating is an evaluation of the creditworthiness of a borrower, based on their financial history and ability to repay debt.",
            "what is stock exchange": "A stock exchange is a marketplace where stocks, bonds, and other securities are bought and sold. The New York Stock Exchange (NYSE) is one example.",
            "what is capital gains": "Capital gains are the profits made from the sale of an asset or investment, such as stocks or property, for more than its purchase price.",
            "what is market capitalization": "Market capitalization (market cap) is the total market value of a company's outstanding shares, calculated by multiplying the stock price by the number of shares.",
            "what is venture capital": "Venture capital is financing provided to early-stage, high-growth companies that have the potential to grow rapidly and generate high returns.",
            "what is leveraged buyout": "A leveraged buyout (LBO) is a financial transaction where a company is purchased using a combination of equity and borrowed money.",
            "what is an index fund": "An index fund is a type of mutual fund or ETF designed to replicate the performance of a specific market index, like the S&P 500.",
            "what is sensex":"Sensex, or the S&P BSE Sensex, is the stock market index of the Bombay Stock Exchange (BSE) in India, tracking 30 large, financially stable companies across various sectors to represent the overall market performance."}
    
        for term in investment_terms:# Check if the query contains investment-related terms
            if term in query:
                return investment_terms[term]
        return "Please ask a question about investment, savings, or finance stuff."
    
    st.title("Investment Chatbot")# Streamlit UI
    user_query = st.text_input("Ask a question about investments, stocks, or finance:")
    if user_query:
        response = get_investment_info(user_query)
        if isinstance(response, dict):
            st.write("Stock Price Comparison:")# Display stock comparison
            for symbol, price in response.items():
                st.write(f"{symbol}: {price}")
        else:
            st.write(response)
