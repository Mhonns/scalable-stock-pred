import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
from rpy2.robjects.vectors import StrVector
import re
import json
import daft  # For reading Parquet files
from sentiment_analysis_bert import analyze_sentiment
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

companies_input_path = "../data/companies.csv"
news_input_path = "../data/latest/finance_news.parquet"
output_path = "final_artifacts/sentiment_analysis_full.json"

def read_input_data(filename=companies_input_path):
    """
    Read the first line of the CSV file containing ticker symbols
    """
    with open(filename, 'r') as file:
        first_line = file.readline().strip()
        tickers = [ticker.strip() for ticker in first_line.split(',')]
    return tickers

def read_news_data(filename=news_input_path):
    # Parquet news
    df = daft.read_parquet(filename)
    python_list = df.collect().to_pylist()
    pandas_df = df.to_pandas()
    # Add title to the content
    pandas_df["content"] = pandas_df["title"] + " " + pandas_df["content"]
    column_list = pandas_df["content"].tolist() 
    ticker_list = pandas_df["tickerSymbols"].tolist()
    return column_list, ticker_list

def preprocess_text(text):
    # Split text into sentences
    sentences = re.findall(r'[^.]+[.]', text)
    # Remove \n and \t
    sentences = [sentence.replace('\n', '').replace('\t', '') for sentence in sentences]
    return sentences

def save_sentiment_dict(sentiment_dict):
    with open(output_path, "w") as f:
        json.dump(sentiment_dict, f)

def main():
    # Read the companies and news data
    companies = read_input_data()
    all_news_data, ticker_list = read_news_data()

    # Initialize the bert sentiment analyzer
    analyzer = SentimentIntensityAnalyzer()

    # Dictionary to store the sentiment of each company
    sentiment_dict = {}
    for company in companies:
        sentiment_dict[company] = 0
    print(sentiment_dict)
    
    for i, news_data in enumerate(all_news_data):
        # Preprocess the news data
        sentences = preprocess_text(news_data)
        ticker_tagged = ticker_list[i]

        # Create lists to store both VADER and BERT results
        vader_results = []
        
        # Get VADER sentiment scores
        for sentence in sentences:
            vs = analyzer.polarity_scores(sentence)
            vader_results.append({
                'text': sentence,
                'vader_sentiment': 'positive' if vs['compound'] > 0 else 'negative' if vs['compound'] < 0 else 'neutral',
                'vader_score': vs['compound']
            })
        
        # Get BERT sentiment analysis
        bert_results = analyze_sentiment(sentences)

        cum_vader_score = 0
        cum_bert_score = 0
        for _, (vader, bert) in enumerate(zip(vader_results, bert_results.to_dict('records'))):
            # Print test each sentiment
            # print(f"\nText: {vader['text'][:100]}...")  # Show first 100 chars
            # print(f"VADER: {vader['vader_sentiment']} (score: {vader['vader_score']:.3f})")
            # print(f"BERT:  {bert['simple_sentiment']} (confidence: {bert['confidence']:.3f})")
            
            cum_vader_score += vader['vader_score']
            if bert['simple_sentiment'] == 'positive':
                cum_bert_score += bert['confidence']
            elif bert['simple_sentiment'] == 'negative':
                cum_bert_score -= bert['confidence']
            else:
                cum_bert_score += bert['confidence'] - 0.5

        # Update the sentiment dictionary
        if len(vader_results) > 0 and len(bert_results) > 0:
            for company in companies:
                if company in ticker_tagged:
                    sentiment_dict[company] += (cum_bert_score + cum_vader_score) / (2 * len(sentences) * len(news_data))
                    
    print(sentiment_dict)
    save_sentiment_dict(sentiment_dict)
    
if __name__ == "__main__":
    main()