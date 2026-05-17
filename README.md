# 📈 MarketMantra — Stock Trend Predictor

> A full-stack stock market analytics and prediction platform built with Python and Streamlit.  
> Combines real-time market data, technical analysis, and ensemble machine learning to forecast next-day price trends.

[![Python](https://img.shields.io/badge/Python-3.9+-blue?style=flat-square&logo=python)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-red?style=flat-square&logo=streamlit)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

MarketMantra web app - https://marketmantra-hckhdps.gamma.site/
---

## 🌐 Live Demo

Scan the QR code inside the app or visit the deployed link.

---

## 🔍 What It Does

MarketMantra lets you pick any stock (NSE, BSE, NYSE, NASDAQ, crypto — anything on Yahoo Finance), set a date range, and instantly get:

- **Historical price charts** with interactive technical indicators
- **ML-powered next-day trend prediction** (Up / Down) with probability scores
- **Portfolio & watchlist** management across sessions
- **ROI calculator** — enter an investment amount and see what it's worth today (including dividends)

---

## ✨ Features

### 📊 Technical Indicators
| Indicator | What It Tells You |
|-----------|-------------------|
| 50-Day SMA | Short-term trend direction |
| 200-Day SMA | Long-term trend direction |
| MACD | Momentum & trend reversals |
| Stochastic Oscillator | Overbought / oversold signal |
| Bollinger Bands | Volatility & breakout zones |
| RSI | Relative strength (buy/sell signal) |
| Volume Chart | Buying vs. selling pressure |

### 🤖 Machine Learning Models
MarketMantra trains **four models in parallel** and averages their predictions for a more robust signal:

- Random Forest
- Gradient Boosting
- XGBoost
- Decision Tree

Features used: previous close price, daily return percentage. Output: probability of an upward or downward move the next trading day.

### 💼 Portfolio & Watchlist
Add any ticker to your portfolio or watchlist, track multiple positions, and remove them with one click — all persisted in your Streamlit session.

### 📉 ROI Calculator
Input a start date and investment amount for any ticker. MarketMantra calculates final value, total return, return percentage, and cumulative dividends earned.


---

## 🛠 Tech Stack

| Category | Tools |
|----------|-------|
| Language | Python 3.9+ |
| Web app | Streamlit |
| ML | scikit-learn, XGBoost |
| Data | Pandas, NumPy |
| Market data | yfinance, Requests |
| Visualization | Matplotlib, Plotly |
| Image | Pillow |

---

## 🚀 Getting Started

### 1. Clone the repo

```bash
git clone https://github.com/Homosapien9/MarketMantra.git
cd MarketMantra
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the app

```bash
streamlit run marketmantra.py
```

The app opens in your browser at `http://localhost:8501`.

---

## 📦 Requirements

```
numpy
pandas
xgboost
yfinance
Pillow
streamlit
matplotlib
scikit-learn
plotly
requests
```

Install all at once:
```bash
pip install -r requirements.txt
```

---

## 📁 Project Structure

```
MarketMantra/
├── marketmantra.py               # Main Streamlit application
├── requirements.txt   # Python dependencies
├── Website qr.png     # QR code for deployed web app
├── app QR.png         # QR code for mobile app
└── README.md
```

---

## 📖 How to Use

1. **Select a stock** — Enter any valid Yahoo Finance ticker (e.g. `^BSESN` for Sensex, `AAPL` for Apple, `BTC-USD` for Bitcoin)
2. **Set a date range** — Minimum 5 days of data required for the model to run
3. **Choose indicators** — Select any combination of the 7 technical indicators to display
4. **View prediction** — Head to the **Predictions** tab for next-day forecast with confidence percentages
5. **Manage your portfolio** — Use the Portfolio and Watchlist tabs to track stocks
6. **Calculate ROI** — Enter a start date and investment amount under the ROI tab

---

## 🧠 Model Details

The ML pipeline:

1. Pulls OHLCV data using `yfinance`
2. Engineers two features: `Previous Close` and `Daily Return`
3. Creates a binary label: 1 if next day's close > today's close, else 0
4. Normalises features with `StandardScaler`
5. Splits 90/10 train/test
6. Trains all four classifiers in parallel
7. Averages predicted probabilities across models for the final signal
8. Displays a confusion matrix and per-model accuracy selector

---

## ⚠️ Disclaimer

MarketMantra is an educational project. Nothing in this app constitutes financial advice. Past performance and model predictions do not guarantee future results. Always do your own research before making investment decisions.

---

## 👤 Author

**Jeff (Homosapien9)**  
GitHub: [@Homosapien9](https://github.com/Homosapien9)

---

## 🌟 Show Your Support

If you found this useful, consider starring the repo — it helps others discover the project!
