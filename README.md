# Stock Price Predictor

A machine learning application that predicts stock price movements using historical stock data and Reddit sentiment analysis.

## Features
- Scrapes Reddit r/stocks for mentions of specific stocks
- Collects historical stock data using Yahoo Finance
- Trains a Random Forest model to predict stock price movements
- Web interface for making predictions on supported stocks

## Supported Stocks
- ORCL (Oracle)
- NKE (Nike)
- CVX (Chevron)
- ADBE (Adobe)
- FB (Meta/Facebook)
- MCD (McDonald's)

## Prerequisites
- Python 3.8 or higher
- pip (Python package installer)

## Installation

1. Clone the repository:
2. git clone https://github.com/your-username/Stock-prediction.git
cd stock-price-predictor

pip install -r requirements.txt

Set up API keys

REDDIT_CLIENT_ID=your_reddit_client_id
REDDIT_CLIENT_SECRET=your_reddit_client_secret
REDDIT_USER_AGENT=your_user_agent

python app.py


