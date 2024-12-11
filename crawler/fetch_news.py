import http.client
import json
import os
import pandas as pd
import daft
import yfinance as yf
from datetime import datetime
import requests
from bs4 import BeautifulSoup
import shutil
from pathlib import Path

headers = {
    'accept': '*/*',
    'accept-encoding': 'gzip, deflate, br',
    'accept-language': 'en-US,en;q=0.9',
    'referer': 'https://www.google.com',
    'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/85.0.4183.83 Safari/537.36 Edg/85.0.564.44'
}

def get_content(url):
    """
    Extracts only the main text content from a given article URL.
    Returns a single string containing all the article text.
    Returns None if extraction fails.
    """
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'lxml')
        
        # Find all text content
        content = []
        
        # Look for common article content containers
        article_containers = soup.find_all(['article', 'div'], class_=lambda x: x and any(term in x.lower() for term in ['article', 'content', 'story', 'body']))
        
        for container in article_containers:
            # Get all paragraphs within the container
            paragraphs = container.find_all('p')
            for p in paragraphs:
                # Skip if paragraph is too short (likely not main content)
                text = p.get_text(strip=True)
                if len(text) > 50:
                    content.append(text)
        
        # Join all paragraphs with newlines
        if content:
            return '\n'.join(content)
        return None
    
    except Exception as e:
        print(f"Error scraping {url}: {str(e)}")
        return None
    
def fetch_finance_news():
    # Create connection
    conn = http.client.HTTPSConnection("yahoo-finance166.p.rapidapi.com")
    
    # Define headers
    headers = {
        'x-rapidapi-key': os.environ['RAPID_API_KEY'],
        'x-rapidapi-host': "yahoo-finance166.p.rapidapi.com"
    }
    
    try:
        # Make request
        conn.request("GET", "/api/news/list-by-symbol?s=AAPL%2CGOOGL%2CTSLA&region=US&snippetCount=500", headers=headers)
        
        # Get response
        res = conn.getresponse()
        data = res.read()
        
        # Parse JSON response
        json_data = json.loads(data.decode('utf-8'))
        
        # # Write to JSON file
        # with open('finance_news.json', 'w', encoding='utf-8') as f:
        #     json.dump(json_data, f, indent=4, ensure_ascii=False)
            
        # print("Data successfully written to finance_news.json")
        return json_data
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return None
    finally:
        conn.close()

def convert_json_to_daft(data):
    """
    Convert JSON news data to a Daft DataFrame with specific fields.
    Also tracks all unique ticker symbols encountered.
    
    Args:
        data (dict): Input JSON data
        
    Returns:
        tuple: (daft.DataFrame, set): Tuple containing:
            - Daft DataFrame containing extracted information
            - Set of all unique ticker symbols encountered
    """
    all_tickers = set()

    def convert_json_to_df(data):
        extracted_data = []
        
        for item in data['data']['main']['stream']:
            content = item['content']
            
            image_urls = []
            if content.get('thumbnail') and content['thumbnail'].get('resolutions'):
                image_urls = [res['url'] for res in content['thumbnail']['resolutions']]
            
            ticker_symbols = []
            if content.get('finance') and content['finance'].get('stockTickers'):
                ticker_symbols = [ticker['symbol'] for ticker in content['finance']['stockTickers']]
                all_tickers.update(ticker_symbols)

            url = ''
            if content.get('clickThroughUrl'):
                if isinstance(content['clickThroughUrl'], dict):
                    url = content['clickThroughUrl'].get('url', '')
            
            news_content = ''
            if url:
                news_content = get_content(url)
            
            entry = {
                'title': content.get('title', ''),
                'url': url,
                'pubDate': content.get('pubDate', ''),
                'imageUrls': image_urls,
                'tickerSymbols': ticker_symbols,
                'provider': content.get('provider', {}).get('displayName', ''),
                'content': news_content
            }
            
            extracted_data.append(entry)
        
        return pd.DataFrame(extracted_data)
    
    # Convert to pandas first
    pdf = convert_json_to_df(data)
    
    # Convert pandas DataFrame to Daft DataFrame
    ddf = daft.from_pandas(pdf)
    
    return ddf, all_tickers

def save_daft_dataframe(ddf, filename):
    """
    Save Daft DataFrame to parquet format.
    
    Args:
        ddf (daft.DataFrame): Daft DataFrame to save
        filename (str): Output filename (should end with .parquet)
    """
    if not filename.endswith('.parquet'):
        filename += '.parquet'
    
    ddf.write_parquet(filename)

def fetch_stock_data(symbols, output_file='stock_data.parquet'):
    """
    Fetch stock data for the past month for a list of symbols and save to a single file.
    
    Args:
        symbols (list): List of stock symbols to fetch data for
        output_file (str): Output file path (default: 'combined_stock_data.parquet')
        
    Returns:
        pandas.DataFrame: Combined DataFrame with multi-index (symbol, date)
    """
    # Dictionary to store DataFrames
    all_data = []
    
    for symbol in symbols:
        try:
            # Fetch data for the symbol
            ticker = yf.Ticker(symbol)
            data = ticker.history(period="1mo") 
            
            # Reset index to make date a column
            data = data.reset_index()
            
            # Add symbol column
            data['Symbol'] = symbol
            
            # Append to list
            all_data.append(data)
            
        except Exception as e:
            print(f"Error fetching data for {symbol}: {str(e)}")
            continue
    
    # Combine all DataFrames
    if all_data:
        # First combine with pandas
        combined_df = pd.concat(all_data, axis=0)
        
        # Convert datetime to string to ensure compatibility
        combined_df['Date'] = combined_df['Date'].astype(str)
        
        # Convert to Daft DataFrame
        daft_df = daft.from_pandas(combined_df)
        
        # Sort by Symbol and Date
        daft_df = daft_df.sort(['Symbol', 'Date'])
        
        # Save to parquet file
        if not output_file.endswith('.parquet'):
            output_file += '.parquet'
        
        daft_df.write_parquet(output_file)
        print(f"Saved combined data to {output_file}")
        
        return daft_df
    else:
        print("No data was fetched successfully")
        return None

if __name__ == "__main__":
    print("Fetching latest 200 Finance related new from Yahoo Finance")
    json_data = fetch_finance_news()
    print("Extracting content, symbols into Daft DataFrame..")
    ddf, ticker_symbols = convert_json_to_daft(json_data)

    # Save the Daft DataFrame
    save_daft_dataframe(ddf, 'finance_news.parquet')
    
    # Display the result
    print("First few rows of the Daft DataFrame:")
    print(ddf.show())

    print("Fetching stock data")
    daft_df = fetch_stock_data(ticker_symbols)
    
    if daft_df is not None:
        print("\nPreview of stock data:")
        print(daft_df.show())
        
