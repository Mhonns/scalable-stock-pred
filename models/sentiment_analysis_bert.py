from transformers import pipeline
import pandas as pd

def analyze_sentiment(texts, batch_size=32):
    """
    Analyze sentiment of multiple texts using a pre-trained model.
    
    Args:
        texts (list): List of strings to analyze
        batch_size (int): Number of texts to process at once
    
    Returns:
        pandas.DataFrame: DataFrame containing text, sentiment label, and confidence scores
    """
    # Initialize sentiment pipeline with RoBERTa model
    sentiment_analyzer = pipeline(
        "sentiment-analysis",
        model="cardiffnlp/twitter-roberta-base-sentiment",
        tokenizer="cardiffnlp/twitter-roberta-base-sentiment"
    )
    
    # Process texts in batches
    results = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        batch_results = sentiment_analyzer(batch)
        results.extend(batch_results)
    
    # Create DataFrame with results
    df = pd.DataFrame({
        'text': texts,
        'sentiment': [r['label'] for r in results],
        'confidence': [r['score'] for r in results]
    })
    
    # Add simplified sentiment (positive/negative/neutral)
    df['simple_sentiment'] = df['sentiment'].map({
        'LABEL_0': 'negative',
        'LABEL_1': 'neutral',
        'LABEL_2': 'positive'
    })
    
    return df

# Example usage:
if __name__ == "__main__":
    sample_texts = [
        "I absolutely love this product!",
        "The service was terrible and I'm very disappointed.",
        "It's okay, nothing special."
    ]
    
    results_df = analyze_sentiment(sample_texts)
    print("\nSentiment Analysis Results:")
    print(results_df[['text', 'simple_sentiment', 'confidence']])