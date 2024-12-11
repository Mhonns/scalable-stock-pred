from typing import Tuple, Set, Optional
import http.client
import json
import os
import pandas as pd
import daft
import yfinance as yf
import requests
from bs4 import BeautifulSoup
from dataclasses import dataclass
from datetime import datetime
import shutil
from pathlib import Path

@dataclass
class NewsData:
    daft_frame: daft.DataFrame
    ticker_symbols: Set[str]

class FinanceDataPipeline:
    def __init__(self, rapid_api_key: str):
        self.rapid_api_key = rapid_api_key
        self.data_dir = Path("../data")
        self.latest_dir = self.data_dir / "latest"
        self.headers = {
            'accept': '*/*',
            'accept-encoding': 'gzip, deflate, br',
            'accept-language': 'en-US,en;q=0.9',
            'referer': 'https://www.google.com',
            'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/85.0.4183.83 Safari/537.36 Edg/85.0.564.44'
        }
        # List of symbols to exclude (e.g., currency pairs, known delisted stocks)
        self.excluded_symbols = {'USD=X', 'CELG-RT'}
        
        # Create necessary directories
        self.data_dir.mkdir(exist_ok=True)
        self.latest_dir.mkdir(exist_ok=True)

    def _archive_existing_file(self, filename: str):
        """Archive existing file with timestamp before saving new data."""
        latest_file = self.latest_dir / filename
        if latest_file.exists():
            # Create timestamp directory if it doesn't exist
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            timestamp_dir = self.data_dir / timestamp
            timestamp_dir.mkdir(exist_ok=True)
            
            # Move the file to timestamped directory
            shutil.move(str(latest_file), str(timestamp_dir / filename))
            print(f"Archived previous {filename} to {timestamp_dir}")

    def save_data(self, data: Optional[daft.DataFrame], filename: str):
        """Save DataFrame to parquet format with versioning."""
        if data is None:
            print(f"No data to save to {filename}")
            return
            
        if not filename.endswith('.parquet'):
            filename += '.parquet'
            
        # Archive existing file if it exists
        self._archive_existing_file(filename)
        
        # Save new file to latest directory
        output_path = self.latest_dir / filename
        data.write_parquet(str(output_path))
        print(f"Successfully saved data to {output_path}")

    def _get_article_content(self, url: str) -> Optional[str]:
        """Extract main text content from article URL."""
        try:
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'lxml')
            
            content = []
            article_containers = soup.find_all(
                ['article', 'div'], 
                class_=lambda x: x and any(term in x.lower() for term in ['article', 'content', 'story', 'body'])
            )
            
            for container in article_containers:
                paragraphs = container.find_all('p')
                content.extend(p.get_text(strip=True) for p in paragraphs if len(p.get_text(strip=True)) > 50)
            
            return '\n'.join(content) if content else None
            
        except Exception as e:
            print(f"Error scraping {url}: {str(e)}")
            return None

    def fetch_news(self, symbols: str = "AAPL,GOOGL,TSLA") -> Optional[dict]:
        """Fetch financial news from Yahoo Finance API."""
        conn = http.client.HTTPSConnection("yahoo-finance166.p.rapidapi.com")
        headers = {
            'x-rapidapi-key': self.rapid_api_key,
            'x-rapidapi-host': "yahoo-finance166.p.rapidapi.com"
        }
        
        try:
            conn.request("GET", f"/api/news/list-by-symbol?s={symbols}&region=US&snippetCount=500", headers=headers)
            res = conn.getresponse()
            return json.loads(res.read().decode('utf-8'))
        except Exception as e:
            print(f"Error fetching news: {str(e)}")
            return None
        finally:
            conn.close()

    def process_news_data(self, json_data: dict) -> NewsData:
        """Process JSON news data into a structured format."""
        all_tickers = set()
        extracted_data = []
        
        for item in json_data['data']['main']['stream']:
            content = item['content']
            
            image_urls = []
            if content.get('thumbnail') and content['thumbnail'].get('resolutions'):
                image_urls = [res['url'] for res in content['thumbnail']['resolutions']]
            
            ticker_symbols = []
            if content.get('finance') and content['finance'].get('stockTickers'):
                ticker_symbols = [
                    ticker['symbol'] 
                    for ticker in content['finance']['stockTickers']
                    if ticker['symbol'] not in self.excluded_symbols
                ]
                all_tickers.update(ticker_symbols)

            url = ''
            if content.get('clickThroughUrl'):
                if isinstance(content['clickThroughUrl'], dict):
                    url = content['clickThroughUrl'].get('url', '')
            
            news_content = self._get_article_content(url) if url else ''
            
            extracted_data.append({
                'title': content.get('title', ''),
                'url': url,
                'pubDate': content.get('pubDate', ''),
                'imageUrls': image_urls,
                'tickerSymbols': ticker_symbols,
                'provider': content.get('provider', {}).get('displayName', ''),
                'content': news_content
            })
        
        df = daft.from_pandas(pd.DataFrame(extracted_data))
        # Materialize the DataFrame to avoid len() issues
        df = df.collect()
        return NewsData(daft_frame=df, ticker_symbols=all_tickers)

    def fetch_stock_data(self, symbols: Set[str], period: str = "3mo") -> Optional[daft.DataFrame]:
        """Fetch historical stock data for given symbols."""
        all_data = []
        valid_symbols = {sym for sym in symbols if sym not in self.excluded_symbols}
        
        for symbol in valid_symbols:
            try:
                ticker = yf.Ticker(symbol)
                data = ticker.history(period=period)
                
                # Skip if no data was returned
                if data.empty:
                    print(f"No data available for {symbol}")
                    continue
                    
                data = data.reset_index()
                data['Symbol'] = symbol
                all_data.append(data)
            except Exception as e:
                print(f"Error fetching data for {symbol}: {str(e)}")
                continue
        
        if not all_data:
            print("No valid stock data was fetched")
            return None
            
        combined_df = pd.concat(all_data, axis=0)
        combined_df['Date'] = combined_df['Date'].astype(str)
        df = daft.from_pandas(combined_df).sort(['Symbol', 'Date'])
        # Materialize the DataFrame to avoid len() issues
        return df.collect()

    def run_pipeline(self) -> Tuple[Optional[daft.DataFrame], Optional[daft.DataFrame]]:
        """Execute the complete pipeline."""
        # Step 1: Fetch news data
        print("Fetching financial news...")
        news_json = self.fetch_news()
        if not news_json:
            print("Failed to fetch news data")
            return None, None

        # Step 2: Process news data
        print("Processing news data...")
        try:
            news_data = self.process_news_data(news_json)
            self.save_data(news_data.daft_frame, 'finance_news.parquet')
        except Exception as e:
            print(f"Error processing news data: {str(e)}")
            return None, None

        # Step 3: Fetch stock data
        print("Fetching stock data...")
        try:
            stock_data = self.fetch_stock_data(news_data.ticker_symbols)
            if stock_data is not None:
                self.save_data(stock_data, 'stock_data.parquet')
        except Exception as e:
            print(f"Error fetching stock data: {str(e)}")
            return news_data.daft_frame, None

        return news_data.daft_frame, stock_data

# Example usage
if __name__ == "__main__":
    pipeline = FinanceDataPipeline(rapid_api_key=os.environ['RAPID_API_KEY'])
    news_df, stock_df = pipeline.run_pipeline()
    
    if news_df is not None:
        print("\nNews Data Preview:")
        print(news_df.show())
    
    if stock_df is not None:
        print("\nStock Data Preview:")
        print(stock_df.show())