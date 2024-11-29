import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

class StockDataCollector:
    def __init__(self):
        # List of stock symbols we're tracking
        self.symbols = ['ORCL', 'NKE', 'CVX', 'ADBE', 'FB', 'MCD']

    def fetch_stock_data(self, start_date):
        """Fetch historical stock data for all symbols"""
        all_stock_data = []
        
        for symbol in self.symbols:
            print(f"Fetching data for {symbol}...")
            try:
                stock = yf.Ticker(symbol)
                hist = stock.history(start=start_date)
                
                if hist.empty:
                    print(f"No data received for {symbol}")
                    continue
                
                # Reset index to make Date a column
                hist = hist.reset_index()
                
                # Add symbol column
                hist['Symbol'] = symbol
                
                # Calculate daily returns
                hist['returns'] = hist['Close'].pct_change()
                
                # Calculate volatility (20-day rolling standard deviation)
                hist['volatility'] = hist['returns'].rolling(window=20).std()
                
                # Calculate moving averages
                hist['ma5'] = hist['Close'].rolling(window=5).mean()
                hist['ma20'] = hist['Close'].rolling(window=20).mean()
                
                print(f"Got {len(hist)} days of data for {symbol}")
                all_stock_data.append(hist)
            except Exception as e:
                print(f"Error fetching data for {symbol}: {e}")
        
        if not all_stock_data:
            raise ValueError("No stock data was collected!")
            
        return pd.concat(all_stock_data, ignore_index=True)

    def prepare_training_data(self, stock_data, reddit_data):
        """Prepare training data by combining stock and Reddit data"""
        # Process each stock symbol
        all_training_data = []
        
        for symbol in self.symbols:
            # Get stock data for this symbol
            symbol_data = stock_data[stock_data['Symbol'] == symbol].copy()
            
            # Initialize sentiment columns with default values
            symbol_data['post_count'] = 0
            symbol_data['avg_score'] = 0
            symbol_data['avg_comments'] = 0
            
            # Get Reddit data for this symbol
            if not reddit_data.empty:
                daily_sentiment = {}
                for _, row in reddit_data.iterrows():
                    stocks = row['stocks_mentioned'].split(',') if pd.notna(row['stocks_mentioned']) else []
                    if symbol in stocks:
                        date = row['created_utc'].date()
                        if date not in daily_sentiment:
                            daily_sentiment[date] = {
                                'post_count': 0,
                                'total_score': 0,
                                'total_comments': 0
                            }
                        daily_sentiment[date]['post_count'] += 1
                        daily_sentiment[date]['total_score'] += row['score']
                        daily_sentiment[date]['total_comments'] += row['num_comments']
                
                if daily_sentiment:  # Only process if we have sentiment data
                    # Convert daily_sentiment to DataFrame
                    sentiment_df = pd.DataFrame.from_dict(daily_sentiment, orient='index')
                    sentiment_df.index = pd.to_datetime(sentiment_df.index)
                    
                    # Match dates from stock data
                    for date in symbol_data['Date'].dt.date:
                        if date in daily_sentiment:
                            mask = symbol_data['Date'].dt.date == date
                            symbol_data.loc[mask, 'post_count'] = daily_sentiment[date]['post_count']
                            symbol_data.loc[mask, 'avg_score'] = daily_sentiment[date]['total_score'] / daily_sentiment[date]['post_count']
                            symbol_data.loc[mask, 'avg_comments'] = daily_sentiment[date]['total_comments'] / daily_sentiment[date]['post_count']
            
            # Create target variable (1 if price goes up tomorrow, 0 if down)
            symbol_data['target'] = (symbol_data['returns'].shift(-1) > 0).astype(int)
            
            all_training_data.append(symbol_data)
        
        # Combine all processed data
        combined_data = pd.concat(all_training_data, ignore_index=True)
        
        # Remove rows with NaN values
        combined_data = combined_data.dropna()
        
        return combined_data 