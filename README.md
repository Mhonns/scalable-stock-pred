# Stock Price Prediction with RNN and Sentiment Analysis

## Overview
This project combines technical analysis using Recurrent Neural Networks (RNN) and sentiment analysis of news headlines to predict stock price movements. The final prediction is made using an Artificial Neural Network (ANN) that integrates both technical and sentiment indicators.


## Project Structure
`
├── crawler/ 
│ └── # Data scraping scripts for Yahoo News
├── data/
│ └── # Stored scraped news data and stock price data
├── models/
│ ├── sentiment_analysis.py # Sentiment analysis model
│ ├── rnn_prediction.py # RNN model for technical analysis
│ ├── final_ann.py # Final ANN model combining both predictions
│ └── model_parameters/ # Stored model parameters and weights
└── final_ann_full.py # Main execution script
`


## Features
- **Technical Analysis**: Uses RNN to analyze historical stock price patterns
- **Sentiment Analysis**: Analyzes news headlines to gauge market sentiment
- **Combined Prediction**: Integrates both technical and sentiment indicators using ANN
- **Automated Data Collection**: Scrapes relevant news data from Yahoo Finance


## Model Components
1. **RNN Model**
   - Processes historical price data
   - Predicts technical indicators

2. **Sentiment Analysis**
   - Analyzes news headlines
   - Generates sentiment scores

3. **Final ANN**
   - Combines outputs from RNN and sentiment analysis
   - Produces final price movement prediction

## Data Collection
- Stock price data is collected from financial APIs
- News headlines are scraped from Yahoo Finance
- Data is stored in the `data/` directory

## To run the prediction
bash
`
python final_ann_full.py
`

## To run the crawler
bash
`
python crawler/pipeline.py
`

2. Install the required packages:
pip install -r requirements.txt
## Requirements
python>=3.7
tensorflow>=2.0
pandas
numpy
scikit-learn
beautifulsoup4
requests