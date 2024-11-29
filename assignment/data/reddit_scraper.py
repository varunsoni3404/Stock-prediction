import praw
import pandas as pd
from datetime import datetime
import os
from dotenv import load_dotenv
import re

class RedditScraper:
    def __init__(self):
        load_dotenv()
        
        # Initialize PRAW with your credentials
        self.reddit = praw.Reddit(
            client_id=os.getenv('REDDIT_CLIENT_ID'),
            client_secret=os.getenv('REDDIT_CLIENT_SECRET'),
            user_agent=os.getenv('REDDIT_USER_AGENT')
        )
        
        # Define the specific stocks we're interested in
        self.target_stocks = {'ORCL', 'NKE', 'CVX', 'ADBE', 'FB', 'MCD'}
    
    def extract_stock_symbols(self, text):
        """Extract only the specified stock symbols from text"""
        if not text:
            return ''
        # Look for words that are all uppercase and preceded by optional $
        pattern = r'\$?[A-Z]+\b'
        matches = re.findall(pattern, text)
        # Filter only our target stocks and remove $ if present
        symbols = [symbol.replace('$', '') for symbol in matches]
        filtered_symbols = [s for s in symbols if s in self.target_stocks]
        return ','.join(filtered_symbols) if filtered_symbols else ''
    
    def scrape_subreddits(self, limit=1000):
        """Scrape posts from r/stocks"""
        print("Scraping r/stocks for specific stocks:", ', '.join(self.target_stocks))
        
        posts_data = []
        subreddit = self.reddit.subreddit('stocks')
        
        # Get posts from different time periods
        for time_filter in ['day', 'week', 'month', 'year']:
            print(f"Searching posts from past {time_filter}...")
            
            for post in subreddit.top(time_filter=time_filter, limit=limit):
                # Check both title and post content for stock mentions
                title_stocks = self.extract_stock_symbols(post.title)
                selftext_stocks = self.extract_stock_symbols(post.selftext if hasattr(post, 'selftext') else '')
                
                # Combine unique stocks mentioned
                all_stocks = set((title_stocks + ',' + selftext_stocks).split(',')) - {''}
                stocks_mentioned = ','.join(all_stocks)
                
                if stocks_mentioned:  # Only add if stocks were mentioned
                    posts_data.append({
                        'title': post.title,
                        'score': post.score,
                        'num_comments': post.num_comments,
                        'created_utc': datetime.fromtimestamp(post.created_utc),
                        'upvote_ratio': post.upvote_ratio,
                        'stocks_mentioned': stocks_mentioned
                    })
                    print(f"Found mention of {stocks_mentioned}")
        
        df = pd.DataFrame(posts_data)
        print(f"\nCollected {len(df)} relevant posts")
        return df
    
    def save_data(self, df, filename='data/reddit_data.csv'):
        """Save the scraped data to a CSV file"""
        if not df.empty:
            df.to_csv(filename, index=False)
            print(f"Saved {len(df)} posts to {filename}")
        else:
            print("No data to save - DataFrame is empty")