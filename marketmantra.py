# =========================
# IMPORTS (merged)
# =========================
import requests
import numpy as np
import pandas as pd
import xgboost as xgb
import yfinance as yf
import streamlit as st
from datetime import datetime
import matplotlib.pyplot as plt
from PIL import Image
from bs4 import BeautifulSoup
from googlesearch import search
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier

# =========================
# SESSION STATE
# =========================
if 'portfolio' not in st.session_state:
    st.session_state['portfolio'] = []

if 'watchlist' not in st.session_state:
    st.session_state['watchlist'] = []

# =========================
# DATA FETCH (robust)
# =========================
def get_stock_data(stock_symbol, start_date, end_date):
    """Download stock data with error handling."""
    try:
        df = yf.download(stock_symbol, start=start_date, end=end_date)
        if 'Adj Close' in df.columns:
            df.drop(columns=['Adj Close'], inplace=True)
        return df
    except Exception as e:
        st.error(f"Error fetching stock data: {e}")
        return pd.DataFrame()

# =========================
# INDICATOR FUNCTIONS
# =========================
def compute_rsi(df, window=14):
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0.0).rolling(window).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def plot_rsi(df, window=14):
    df_plot = df.copy()
    df_plot['RSI'] = compute_rsi(df_plot, window)
    fig, ax = plt.subplots(figsize=(15, 5))
    ax.plot(df_plot.index, df_plot['RSI'], label="RSI", color="blue")
    ax.axhline(70, color='red', linestyle='--', label="Overbought (70)")
    ax.axhline(30, color='green', linestyle='--', label="Oversold (30)")
    ax.set_title('Relative Strength Index (RSI)', fontsize=15)
    ax.set_xlabel('Date')
    ax.set_ylabel('RSI Value')
    ax.legend(loc="upper left")
    st.pyplot(fig)

def compute_macd(df, fast=12, slow=26, signal=9):
    macd_line = df['Close'].ewm(span=fast, adjust=False).mean() - \
                df['Close'].ewm(span=slow, adjust=False).mean()
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line

def compute_stochastic(df, window=14):
    low_min = df['Low'].rolling(window=window).min()
    high_max = df['High'].rolling(window=window).max()
    stochastic = 100 * (df['Close'] - low_min) / (high_max - low_min)
    return stochastic

def compute_bollinger_bands(df, window=20):
    df = df.copy()
    df['Middle_BB'] = df['Close'].rolling(window=window).mean()
    df['Std_Dev'] = df['Close'].rolling(window=window).std()
    df['Upper_BB'] = df['Middle_BB'] + (df['Std_Dev'] * 2)
    df['Lower_BB'] = df['Middle_BB'] - (df['Std_Dev'] * 2)
    return df

def plot_bollinger_bands(df, window=20):
    df_plot = compute_bollinger_bands(df, window)
    fig, ax = plt.subplots(figsize=(15, 5))
    ax.plot(df_plot.index, df_plot['Close'], label='Close Price', color='blue')
    ax.plot(df_plot.index, df_plot['Upper_BB'], label='Upper Bollinger Band', color='red', linestyle='--')
    ax.plot(df_plot.index, df_plot['Middle_BB'], label='Middle Bollinger Band (SMA)', color='orange', linestyle='--')
    ax.plot(df_plot.index, df_plot['Lower_BB'], label='Lower Bollinger Band', color='green', linestyle='--')
    ax.fill_between(df_plot.index, df_plot['Upper_BB'], df_plot['Lower_BB'], color='gray', alpha=0.2)
    ax.set_title('Bollinger Bands', fontsize=15)
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    ax.legend(loc='upper left')
    st.pyplot(fig)

def compute_volumetric_data(df):
    df = df.copy()
    df['Buy_Volume'] = df['Volume'].where(df['Close'] > df['Open'], 0)
    df['Sell_Volume'] = df['Volume'].where(df['Close'] <= df['Open'], 0)
    return df

def plot_volumetric_chart(df):
    st.write("Volume chart tracks the number of shares/contracts traded.")
    st.write("High volume: Confirms price trends (up or down).")
    st.write("Low volume: Signals lack of interest or indecision.")
    df_vol = compute_volumetric_data(df)
    fig, ax = plt.subplots(figsize=(15, 7))
    ax.bar(df_vol.index, df_vol['Buy_Volume'], color='green', alpha=0.6, label='Buying Pressure')
    ax.bar(df_vol.index, df_vol['Sell_Volume'], color='red', alpha=0.6, label='Selling Pressure')
    ax.set_title('Volumetric Chart: Buying vs Selling Pressure', fontsize=15)
    ax.set_xlabel('Date')
    ax.set_ylabel('Volume')
    ax.legend(loc='upper left')
    st.pyplot(fig)

# =========================
# FEATURE ENGINEERING (for ML)
# =========================
def add_features(df):
    """Add technical features and target for the model."""
    df = df.copy()
    df['Return'] = df['Close'].pct_change()
    df['SMA_10'] = df['Close'].rolling(10).mean()
    df['SMA_50'] = df['Close'].rolling(50).mean()
    df['EMA_10'] = df['Close'].ewm(span=10).mean()
    df['Volatility'] = df['Close'].rolling(10).std()
    df['Momentum'] = df['Close'] - df['Close'].shift(5)
    df['Lag1'] = df['Close'].shift(1)
    df['Lag2'] = df['Close'].shift(2)
    df['Lag3'] = df['Close'].shift(3)
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    df.dropna(inplace=True)
    return df

# =========================
# GOOGLE SEARCH FALLBACK
# =========================
def google_search_answer(query):
    try:
        for url in search(query, num_results=3):
            res = requests.get(url, timeout=5)
            soup = BeautifulSoup(res.text, "html.parser")
            paragraphs = soup.find_all('p')
            text = " ".join([p.get_text() for p in paragraphs[:5]])
            if len(text) > 200:
                return text[:500] + "..."
        return "No useful results found."
    except Exception as e:
        return f"Error: {e}"

# =========================
# CHATBOT HELPER (Information Hub)
# =========================
def get_stock_price(stock_symbol):
    """Fetch current price from Yahoo Finance API."""
    try:
        url = f"https://query1.finance.yahoo.com/v7/finance/quote?symbols={stock_symbol}"
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        if 'quoteResponse' in data and 'result' in data['quoteResponse'] and len(data['quoteResponse']['result']) > 0:
            stock_data = data['quoteResponse']['result'][0]
            return stock_data['regularMarketPrice']
        return None
    except Exception:
        return None

def get_investment_info(query):
    """Answer investment questions using predefined dictionary or Google fallback."""
    query = query.lower()

    # Price / compare queries
    if 'price' in query:
        words = query.split()
        for word in words:
            if '.' in word:  # likely a symbol
                price = get_stock_price(word.upper())
                if price is not None:
                    return f"The current price of {word.upper()} is {price} USD."
                else:
                    return f"Sorry, I couldn't find the stock price for {word.upper()}. Please check the symbol."

    if 'compare' in query:
        parts = query.split("compare")[1].split("and")
        symbols = [s.strip().upper() for s in parts if s.strip()]
        result = {}
        for sym in symbols:
            price = get_stock_price(sym)
            result[sym] = f"{price} USD" if price else "Not found"
        return result

    # Predefined financial terms
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
        "what is sensex": "Sensex, or the S&P BSE Sensex, is the stock market index of the Bombay Stock Exchange (BSE) in India, tracking 30 large, financially stable companies across various sectors to represent the overall market performance.",
        "what is capital": "Capital refers to money or assets used to generate income or invest in projects or businesses.",
        "what is bear market rally": "A bear market rally is a short-term recovery in stock prices during a longer-term bear market, which often ends in a downturn.",
        "what is leverage": "Leverage involves borrowing money to increase the potential return on an investment, but it also increases the risk of loss.",
        "what is margin trading": "Margin trading is borrowing money from a broker to trade financial assets, allowing you to buy more than you could with your own funds.",
        "what is short selling": "Short selling is selling a security you do not own, hoping to buy it back at a lower price to make a profit.",
        "what is bond yield": "Bond yield is the return an investor can expect to earn on a bond, often expressed as an annual percentage rate.",
        "what is debt-to-equity ratio": "The debt-to-equity ratio is a financial ratio that compares a company's total debt to its shareholder equity, indicating financial leverage.",
        "what is credit default swap": "A credit default swap (CDS) is a financial derivative contract that allows investors to hedge or speculate on the credit risk of a company or government.",
        "what is sovereign debt": "Sovereign debt is the money borrowed by a country's government, typically through the issuance of bonds.",
        "what is quantitative easing": "Quantitative easing is a form of monetary policy in which a central bank buys government securities to increase the money supply and stimulate the economy.",
        "what is financial derivative": "A financial derivative is a contract whose value is based on the price of an underlying asset, like options, futures, or swaps.",
        "what is yield curve": "The yield curve is a graph that plots the interest rates of bonds with different maturity dates, often used to gauge economic conditions.",
        "what is cost of capital": "The cost of capital is the rate of return required by investors for providing capital to a business, used in investment decision-making.",
        "what is private placement": "Private placement is the sale of securities to a small group of institutional or accredited investors, rather than the public market.",
        "what is an annuity": "An annuity is a financial product that provides a series of fixed payments over time, often used for retirement income.",
        "what is income statement": "An income statement is a financial document that shows a company’s revenues, expenses, and profits over a specific period.",
        "what is balance sheet": "A balance sheet is a financial statement that lists a company's assets, liabilities, and equity, providing a snapshot of its financial position.",
        "what is dividend yield": "Dividend yield is a financial ratio that shows how much cash a company pays out in dividends relative to its stock price.",
        "what is insider trading": "Insider trading refers to buying or selling a security based on non-public, material information about the company.",
        "what is credit risk": "Credit risk is the risk that a borrower will default on a loan or bond, causing the lender to lose part or all of the investment.",
        "what is financial leverage": "Financial leverage is the use of borrowed funds to amplify the potential return on an investment, but it increases the risk of loss.",
        "what is risk tolerance": "Risk tolerance refers to the level of risk an investor is willing to take in their investment decisions, based on their financial situation and goals.",
        "what is market efficiency": "Market efficiency is the degree to which market prices reflect all available information. A perfectly efficient market would reflect all known data immediately.",
        "what is diversification strategy": "Diversification strategy involves investing in a variety of assets or asset classes to reduce overall investment risk.",
        "what is roth ira": "A Roth IRA is a retirement account that allows your investments to grow tax-free, and withdrawals are also tax-free in retirement if certain conditions are met.",
        "what is traditional ira": "A Traditional IRA is a retirement account where contributions may be tax-deductible, but withdrawals are taxed as income during retirement.",
        "what is expense ratio": "The expense ratio is the annual fee that mutual funds or ETFs charge to manage an investment portfolio, expressed as a percentage of assets under management.",
        "what is stop-loss order": "A stop-loss order is an order placed with a broker to buy or sell once a stock reaches a certain price, used to limit losses or lock in profits.",
        "what is price-to-earnings ratio": "The price-to-earnings (P/E) ratio is a valuation ratio, calculated by dividing the stock price by the earnings per share (EPS), indicating if a stock is over or under-valued.",
        "what is bear trap": "A bear trap occurs when a security's price briefly drops, luring short-sellers, only for the price to reverse and rise, leading to losses for those betting on a decline.",
        "what is bull trap": "A bull trap happens when a security’s price rises, attracting buyers, only for the price to reverse and fall, causing losses for investors who bought in.",
        "what is exchange rate": "The exchange rate is the value of one currency in terms of another, affecting international trade and investments.",
        "what is a credit line": "A credit line is a pre-approved loan limit provided by a lender to a borrower, which can be drawn upon as needed, typically for short-term needs.",
        "what is debt servicing": "Debt servicing refers to the payments made towards the interest and principal of debt obligations, such as loans or bonds.",
        "what is economic moat": "An economic moat refers to a company’s ability to maintain a competitive advantage and protect itself from the competition, resulting in long-term profitability.",
        "what is blue chip stock": "Blue chip stocks are shares in well-established, financially stable companies known for their reliability and consistent performance.",
        "what is market volatility": "Market volatility refers to the extent of price fluctuations in a market, often indicating uncertainty or risk, and can be measured using the VIX index.",
        "what is stock buyback": "A stock buyback occurs when a company repurchases its own shares from the market, reducing the number of outstanding shares and potentially increasing the stock price.",
        "what is wealth management": "Wealth management is a comprehensive financial service that involves managing an individual's or family's investments, estate, tax, and retirement planning.",
        "what is financial advisor": "A financial advisor is a professional who provides advice on investments, insurance, retirement planning, and other financial matters to individuals or businesses.",
        "what is dollar-cost averaging": "Dollar-cost averaging is an investment strategy where you invest a fixed amount regularly, regardless of market conditions, reducing the impact of volatility.",
        "what is real estate investment trust": "A real estate investment trust (REIT) is a company that owns, operates, or finances real estate properties, allowing investors to pool capital and earn returns through dividends."
    }

    for term, answer in investment_terms.items():
        if term in query:
            return answer

    # Fallback to Google search
    return google_search_answer(query)

# =========================
# ROI CALCULATOR
# =========================
def calculate_investment_return(start_date_str, stock_ticker, investment_amount):
    hist = get_stock_data(stock_ticker, start_date_str, datetime.today().strftime('%Y-%m-%d'))
    if hist.empty:
        st.error("No data available for ROI calculation.")
        return
    start_date_dt = pd.to_datetime(start_date_str)
    if start_date_dt not in hist.index:
        st.error(f"Start date {start_date_str} not in historical data.")
        return
    start_price = hist.loc[start_date_dt]["Close"]
    current_price = hist["Close"].iloc[-1]
    total_dividends = hist["Dividends"].sum() if "Dividends" in hist.columns else 0
    final_value = (investment_amount / start_price) * current_price + total_dividends
    total_return = final_value - investment_amount
    return_percentage = (total_return / investment_amount) * 100

    st.subheader(f"Investment in {stock_ticker} from {start_date_str}")
    st.write(f"Initial Investment: {investment_amount:,.2f}")
    st.write(f"Start Price: {start_price:,.2f}")
    st.write(f"Current Price: {current_price:,.2f}")
    if total_dividends > 0:
        st.write(f"Total Dividends Earned: {total_dividends:,.2f}")
    else:
        st.write("This stock does not offer dividends or no dividends were paid during the selected period.")
    st.write(f"Final Value (including price change and dividends): {final_value:,.2f}")
    st.write(f"Total Return: {total_return:,.2f}")
    st.write(f"Return Percentage: {return_percentage:,.2f}%")

# =========================
# STREAMLIT UI
# =========================
# ---- Header with QR code ----
qr_image = Image.open("Website qr.png")
col1, col2 = st.columns([3, 1])
with col1:
    st.markdown('<h1 style="color: white; font-size: 29.7px;">MarketMantra - Stock Trend Predictor</h1>', unsafe_allow_html=True)
    st.subheader("~ Developed By JEFF")
with col2:
    st.image(qr_image, caption="scan for website", width=100)

# ---- Stock selection ----
with st.expander("Select Stock And Data Range (Minimum 5 Days Gap)"):
    st.header("Stock Selection")
    stock_symbol = st.text_input("Select Stock Symbol", value="^BSESN").upper()
    start_date = st.date_input("Start Date", pd.to_datetime("2024-01-01"))
    end_date = st.date_input("End Date", datetime.now().date())

# ---- Indicator selection ----
with st.expander("Select Technical Indicators"):
    st.header("Technical Indicators")
    indicator_options = [
        "50-Day Simple Moving Average (SMA)",
        "200-Day Simple Moving Average (SMA)",
        "MACD (Moving Average Convergence Divergence)",
        "Stochastic Oscillator",
        "Bollinger Bands",
        "(RSI) Relative Strength Index",
        "Volume Chart"
    ]
    selected_indicators = st.multiselect(
        "Select Technical Indicators to Display",
        indicator_options,
        default=["50-Day Simple Moving Average (SMA)", "200-Day Simple Moving Average (SMA)"]
    )

# Flags for indicators
sma_50 = "50-Day Simple Moving Average (SMA)" in selected_indicators
sma_200 = "200-Day Simple Moving Average (SMA)" in selected_indicators
macd_ind = "MACD (Moving Average Convergence Divergence)" in selected_indicators
stochastic_ind = "Stochastic Oscillator" in selected_indicators
bollinger_ind = "Bollinger Bands" in selected_indicators
rsi_ind = "(RSI) Relative Strength Index" in selected_indicators
volume_ind = "Volume Chart" in selected_indicators

# ---- Fetch raw data ----
df_raw = get_stock_data(stock_symbol, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
if df_raw.empty:
    st.warning("No data found for the selected stock or date range. Model needs at least 5 days to predict results.")
    st.stop()

# ---- Data Visualization ----
with st.expander("Data Visualization"):
    st.subheader(f"Stock Data for {stock_symbol}")
    st.write(f"Historical data for {stock_symbol} from {start_date} to {end_date}, in its listed currency")
    st.dataframe(df_raw.tail())

    st.subheader("Closing Price Over Time")
    fig, ax = plt.subplots(figsize=(15, 5))
    ax.plot(df_raw['Close'], label='Close Price', color='blue')
    ax.set_title(f"{stock_symbol} - Closing Price History", fontsize=15)
    ax.set_ylabel('Price', fontsize=12)
    ax.set_xlabel('Date', fontsize=12)
    ax.grid(True)
    plt.legend()
    st.pyplot(fig)

# ---- Portfolio & Watchlist buttons ----
st.header("Portfolio & Watchlist")
col_btn1, col_btn2, _, _ = st.columns(4)
with col_btn1:
    portfolio_add = st.button("Add to Portfolio")
with col_btn2:
    watchlist_add = st.button("Add to Watchlist")

if portfolio_add:
    if stock_symbol not in st.session_state['portfolio']:
        st.session_state['portfolio'].append(stock_symbol)
        st.success(f"{stock_symbol} added to Portfolio.")
    else:
        st.warning(f"{stock_symbol} is already in your Portfolio.")

if watchlist_add:
    if stock_symbol not in st.session_state['watchlist']:
        st.session_state['watchlist'].append(stock_symbol)
        st.success(f"{stock_symbol} added to Watchlist.")
    else:
        st.warning(f"{stock_symbol} is already in your Watchlist.")

# ---- Feature engineering for ML ----
df_ml = add_features(df_raw)  # adds features and drops NA

# ---- Tabs ----
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Portfolio", "Watchlist", "Technical Indicators", "Predictions", "Calculate ROI", "Information Hub"
])

# ---------- Tab 1: Portfolio ----------
with tab1:
    st.subheader("Manage Your Portfolio")
    if st.session_state['portfolio']:
        for stock in st.session_state['portfolio']:
            col_a, col_b = st.columns([3, 1])
            with col_a:
                st.write(f"**{stock}**")
            with col_b:
                if st.button(f"Remove {stock}", key=f"remove_port_{stock}"):
                    st.session_state['portfolio'].remove(stock)
                    st.rerun()
    else:
        st.write("No Stocks in your portfolio yet.")

# ---------- Tab 2: Watchlist ----------
with tab2:
    st.subheader("Manage Your Watchlist")
    if st.session_state['watchlist']:
        for stock in st.session_state['watchlist']:
            col_a, col_b = st.columns([3, 1])
            with col_a:
                st.write(f"**{stock}**")
            with col_b:
                if st.button(f"Remove {stock}", key=f"remove_watch_{stock}"):
                    st.session_state['watchlist'].remove(stock)
                    st.rerun()
    else:
        st.write("No Stocks in your watchlist yet.")

# ---------- Tab 3: Technical Indicators ----------
with tab3:
    st.subheader("Technical Indicators")
    if sma_50:
        st.header("Simple Moving Average (SMA) of 50 Days")
        st.write("The **50-day** SMA looks at the average price over the last 50 days")
        df_raw['SMA_50'] = df_raw['Close'].rolling(window=50).mean()
        fig, ax = plt.subplots(figsize=(15, 5))
        ax.plot(df_raw['SMA_50'], label="50-Day SMA", color='orange')
        ax.set_title(f"{stock_symbol} - 50-Day Simple Moving Average", fontsize=15)
        ax.set_ylabel('Price', fontsize=12)
        ax.set_xlabel('Date', fontsize=12)
        ax.legend(loc='best')
        st.pyplot(fig)

    if sma_200:
        st.header("Simple Moving Average (SMA) of 200 Days")
        st.write("The **200-day** SMA looks at the average price over the last 200 days")
        df_raw['SMA_200'] = df_raw['Close'].rolling(window=200).mean()
        fig, ax = plt.subplots(figsize=(15, 5))
        ax.plot(df_raw['SMA_200'], label="200-Day SMA", color='green')
        ax.set_title(f"{stock_symbol} - 200-Day Simple Moving Average", fontsize=15)
        ax.set_ylabel('Price', fontsize=12)
        ax.set_xlabel('Date', fontsize=12)
        ax.legend(loc='best')
        st.pyplot(fig)

    if macd_ind:
        st.header("MACD (Moving Average Convergence Divergence)")
        st.write("If the **MACD line** is higher than the **signal line**, the asset price could go **up**.")
        st.write("If the **MACD line** is lower than the **signal line**, the asset price could go **down**.")
        macd_line, signal_line = compute_macd(df_raw)
        fig, ax = plt.subplots(figsize=(15, 5))
        ax.plot(macd_line, label="MACD", color='blue')
        ax.plot(signal_line, label="Signal Line", color='orange')
        ax.set_title(f"{stock_symbol} - MACD", fontsize=15)
        ax.set_ylabel('Value', fontsize=12)
        ax.set_xlabel('Date', fontsize=12)
        ax.legend(loc='best')
        st.pyplot(fig)

    if stochastic_ind:
        st.header("Stochastic Oscillator")
        st.write("Above **80** → might be overbought (could come down). Below **20** → might be oversold (could go up).")
        stochastic_oscillator = compute_stochastic(df_raw)
        fig, ax = plt.subplots(figsize=(15, 5))
        ax.plot(stochastic_oscillator, label="Stochastic Oscillator", color='green')
        ax.axhline(80, linestyle='--', color='red')
        ax.axhline(20, linestyle='--', color='blue')
        ax.set_title(f"{stock_symbol} - Stochastic Oscillator", fontsize=15)
        ax.set_ylabel('Stochastic Value', fontsize=12)
        ax.set_xlabel('Date', fontsize=12)
        ax.legend(loc='best')
        st.pyplot(fig)

    if bollinger_ind:
        st.subheader("Bollinger Bands")
        st.write("If price hits/exceeds the **upper band**, potential **sell**. If it drops below the **lower band**, potential **buy**.")
        plot_bollinger_bands(df_raw)

    if rsi_ind:
        st.subheader("Relative Strength Index (RSI)")
        st.write("RSI above **70** → overbought (sell). RSI below **30** → oversold (buy).")
        plot_rsi(df_raw)

    if volume_ind:
        st.subheader("Volume Chart")
        plot_volumetric_chart(df_raw)

# ---------- Tab 4: Predictions ----------
with tab4:
    st.subheader("Predictions For Next Day's Trading")
    if df_ml.empty:
        st.error("Not enough data after feature engineering. Select a larger date range.")
    else:
        features = df_ml[['Return', 'SMA_10', 'SMA_50', 'EMA_10', 'Volatility', 'Momentum', 'Lag1', 'Lag2', 'Lag3']]
        target = df_ml['Target']

        split = int(len(df_ml) * 0.9)
        X_train, X_test = features[:split], features[split:]
        y_train, y_test = target[:split], target[split:]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        models = {
            "Random Forest": RandomForestClassifier(n_estimators=300, max_depth=8, random_state=50),
            "Gradient Boosting": GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, max_depth=5, random_state=50),
            "XGBoost": xgb.XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.05, random_state=50),
            "Decision Tree": DecisionTreeClassifier(max_depth=6, random_state=50)
        }

        # Train & collect probabilities for ensemble
        probs = []
        model_accuracies = {}
        for name, model in models.items():
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            acc = accuracy_score(y_test, y_pred) * 100
            model_accuracies[name] = acc
            probs.append(model.predict_proba(X_test_scaled)[:, 1])

        avg_prob = np.mean(probs, axis=0)
        final_preds = (avg_prob > 0.5).astype(int)
        ensemble_acc = accuracy_score(y_test, final_preds) * 100

        st.subheader(f"Ensemble Accuracy: {ensemble_acc:.2f}%")

        # Confusion matrix
        cm = confusion_matrix(y_test, final_preds)
        fig, ax = plt.subplots(figsize=(6, 5))
        cax = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        fig.colorbar(cax)
        classes = ['Down', 'Up']
        ax.set(xticks=np.arange(len(classes)), yticks=np.arange(len(classes)),
               xticklabels=classes, yticklabels=classes,
               title="Confusion Matrix", ylabel="True Value", xlabel="Predicted Value")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, format(cm[i, j], 'd'), ha="center", va="center", color="black")
        st.pyplot(fig)

        # Individual model accuracy dropdown
        st.subheader("Individual Model Accuracies")
        selected_model_name = st.selectbox("Select Model", list(models.keys()))
        st.write(f"{selected_model_name} Accuracy: {model_accuracies[selected_model_name]:.2f}%")

        # Next day prediction (ensemble)
        latest_features = scaler.transform(features.iloc[-1:].values)
        latest_probs = [m.predict_proba(latest_features)[0][1] for m in models.values()]
        final_up_prob = np.mean(latest_probs)

        st.subheader("Next Day Prediction")
        if final_up_prob > 0.5:
            st.success(f"UP ({final_up_prob*100:.2f}% probability)")
        else:
            st.error(f"DOWN ({(1-final_up_prob)*100:.2f}% probability)")

# ---------- Tab 5: ROI Calculator ----------
with tab5:
    st.subheader("Stock Investment Return Calculator")
    roi_start_date = st.date_input("Enter Start Date for ROI", pd.to_datetime("2016-01-01"), key="roi_start")
    investment_amount = st.number_input("Enter Investment Amount", min_value=1, value=1000000, key="roi_amount")
    if roi_start_date > datetime.today().date():
        st.warning("Start date cannot be in the future. Using today's date.")
    if stock_symbol and roi_start_date and investment_amount:
        calculate_investment_return(roi_start_date.strftime('%Y-%m-%d'), stock_symbol, investment_amount)

# ---------- Tab 6: Information Hub (Chatbot with Google fallback) ----------
with tab6:
    st.subheader("Information Hub")
    st.write("Ask anything about stocks, investments, or finance. I'll answer from my knowledge base or search the web.")
    user_query = st.text_input("Ask a question about investments, stocks, or finance:")
    if user_query:
        response = get_investment_info(user_query)
        if isinstance(response, dict):
            st.write("**Stock Price Comparison:**")
            for sym, price_info in response.items():
                st.write(f"- {sym}: {price_info}")
        else:
            st.write(response)

# ---- Footer ----
st.markdown("---")
st.caption("MarketMantra – combining technical analysis with machine learning for smarter trading decisions.")
