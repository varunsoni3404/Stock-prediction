from data.reddit_scraper import RedditScraper
from data.stock_data import StockDataCollector
from models.predictor import StockPredictor
from datetime import datetime, timedelta

def analyze_dataframe(df, name="DataFrame"):
    """Helper function to analyze a DataFrame"""
    print(f"\n=== {name} Analysis ===")
    print(f"Shape: {df.shape}")
    print("\nColumns:", df.columns.tolist())
    print("\nSample data:")
    print(df.head())
    print("\nMissing values:")
    print(df.isnull().sum())
    print("\nData types:")
    print(df.dtypes)

def main():
    # Initialize components
    reddit_scraper = RedditScraper()
    stock_collector = StockDataCollector()
    predictor = StockPredictor()
    
    try:
        # Collect and analyze Reddit data
        print("Scraping Reddit data...")
        reddit_data = reddit_scraper.scrape_subreddits()
        if reddit_data.empty:
            raise ValueError("No Reddit data collected!")
        analyze_dataframe(reddit_data, "Reddit Data")
        reddit_scraper.save_data(reddit_data)
        
        # Collect and analyze stock data
        print("\nFetching stock data...")
        start_date = datetime.now() - timedelta(days=90)
        stock_data = stock_collector.fetch_stock_data(start_date=start_date)
        if stock_data.empty:
            raise ValueError("No stock data collected!")
        analyze_dataframe(stock_data, "Stock Data")
        
        print("\nPreparing training data...")
        training_data = stock_collector.prepare_training_data(stock_data, reddit_data)
        if training_data.empty:
            raise ValueError("No training data generated!")
        analyze_dataframe(training_data, "Training Data")
        
        print("\nData Distribution:")
        print(training_data['target'].value_counts())
        
        if len(training_data) < 2:  # Need at least 2 samples for train/test split
            raise ValueError("Not enough data for training!")
            
        # Train and evaluate model
        print("\nTraining model...")
        evaluation = predictor.train(training_data)
        
        # Save the model
        predictor.save_model()
        print("\nModel saved successfully!")
        
    except Exception as e:
        print(f"\nError: {e}")
        print("Please check your Reddit API credentials and internet connection.")

if __name__ == "__main__":
    main() 