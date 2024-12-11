import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
from rpy2.robjects.vectors import StrVector
import sys
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

def install_if_missing(package_name):
    """Install R package if it's not already installed"""
    r = robjects.r
    utils = importr('utils')
    
    # Check if package is installed
    is_installed = r(f'require("{package_name}")')[0]
    
    if not is_installed:
        print(f"Installing {package_name}...")
        utils.install_packages(package_name)

def setup_r_environment():
    """Set up the R environment with required packages"""
    try:
        # First ensure pacman is installed
        robjects.r('''
            if (!require("pacman", quietly = TRUE)) {
                install.packages("pacman", repos="https://cloud.r-project.org", quiet=TRUE)
            }
        ''')
        
        # Use pacman to install and load packages
        robjects.r('''
            library(pacman)
            p_load(sentimentr, dplyr)
        ''')
        
        print("R packages loaded successfully")
        
    except Exception as e:
        print(f"Error setting up R environment: {str(e)}")
        sys.exit(1)
    
    # Import required R packages
    global sentimentr
    sentimentr = importr('sentimentr')

def analyze_sentiment(texts):
    """
    Analyze sentiment of given texts
    
    Args:
        texts: List of strings to analyze
    
    Returns:
        DataFrame with sentiment analysis results
    """
    # Convert Python list to R vector
    r_texts = StrVector(texts)
    
    # Get sentences using sentimentr
    sentences = sentimentr.get_sentences(r_texts)
    
    # Perform sentiment analysis
    sentiment_results = sentimentr.sentiment(sentences)
    
    # Convert R data frame to pandas DataFrame
    df = pd.DataFrame({
        'element_id': list(sentiment_results.rx2('element_id')),
        'sentence_id': list(sentiment_results.rx2('sentence_id')),
        'word_count': list(sentiment_results.rx2('word_count')),
        'sentiment': list(sentiment_results.rx2('sentiment'))
    })
    
    return df

def main(texts):
    # Set up R environment
    print("Setting up R environment...")
    setup_r_environment()
    
    # Example texts
    # texts = [
    #     "I love this product! It's amazing.",
    #     "The service was terrible. Would not recommend.",
    #     "Mixed feelings about this. Some good, some bad."
    # ]
    
    print("\nAnalyzing texts:")
    for i, text in enumerate(texts, 1):
        print(f"\nText {i}: {text}")
    
    # Analyze sentiment
    results = analyze_sentiment(texts)
    
    # Print detailed results
    print("\nDetailed Results:")
    print(results)
    
    # Calculate and print summary statistics
    print("\nSummary Statistics:")
    print(f"Average Sentiment: {results['sentiment'].mean():.3f}")
    print(f"Total Sentiment: {results['sentiment'].sum():.3f}")
    print(f"Number of Positive Sentences: {(results['sentiment'] > 0).sum()}")
    print(f"Number of Negative Sentences: {(results['sentiment'] < 0).sum()}")
    print(f"Number of Neutral Sentences: {(results['sentiment'] == 0).sum()}")

    return results
    
if __name__ == "__main__":
    main()