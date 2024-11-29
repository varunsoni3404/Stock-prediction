from flask import Flask, render_template, request, jsonify
from models.predictor import StockPredictor
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

app = Flask(__name__)
predictor = StockPredictor()
predictor.load_model()

def get_stock_data(symbol):
    """Fetch current stock data and calculate required metrics"""
    try:
        # Get stock data for the last 30 days to calculate moving averages
        stock = yf.Ticker(symbol)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        hist = stock.history(start=start_date)
        
        if hist.empty:
            raise ValueError(f"No data found for symbol {symbol}")
        
        # Calculate required metrics
        current_data = hist.iloc[-1]  # Most recent data
        hist['returns'] = hist['Close'].pct_change()
        hist['volatility'] = hist['returns'].rolling(window=20).std()
        hist['ma5'] = hist['Close'].rolling(window=5).mean()
        hist['ma20'] = hist['Close'].rolling(window=20).mean()
        
        # Prepare data for prediction
        prediction_data = {
            'Open': current_data['Open'],
            'High': current_data['High'],
            'Low': current_data['Low'],
            'Close': current_data['Close'],
            'Volume': current_data['Volume'],
            'returns': hist['returns'].iloc[-1],
            'volatility': hist['volatility'].iloc[-1],
            'ma5': hist['ma5'].iloc[-1],
            'ma20': hist['ma20'].iloc[-1],
            'post_count': 0,  # Default values for Reddit features
            'avg_score': 0,
            'avg_comments': 0
        }
        
        return prediction_data
    except Exception as e:
        raise ValueError(f"Error fetching data for {symbol}: {str(e)}")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        symbol = request.form['symbol'].upper()
        
        # Get current stock data
        stock_data = get_stock_data(symbol)
        
        # Create DataFrame with single row
        df = pd.DataFrame([stock_data])
        
        # Make prediction
        prediction, probabilities = predictor.predict(df)
        
        # Format response
        result = {
            'symbol': symbol,
            'prediction': 'Up' if prediction[0] == 1 else 'Down',
            'confidence': f"{max(probabilities[0]) * 100:.2f}%",
            'stock_data': stock_data
        }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True) 