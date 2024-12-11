from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    #note: depending on how you installed (e.g., using source code download versus pip install), you may need to import like this:
    #from vaderSentiment import SentimentIntensityAnalyzer

# --- examples -------
sentences = ["Elon Musk isn’t getting the $101 billion windfall he so desperately wanted. But he’s still among the richest people on the planet and poised to get much richer in the coming years.",  # positive sentence example
             """Nvidia stock (NVDA) has gained traction ever since the launch of ChatGPT, driving generative AI into the limelight and fueling the unprecedented demand for AI hardware. The stock has surged a remarkable 189% year-to-date, reflecting the company’s significant role in the AI industry and strong financial performance throughout the year.
Its Graphics Processing Units (GPUs), initially used for the video games market, now rake in significant revenue from AI customers. Despite some regulatory challenges, the company has been maintaining its trajectory of growth. Most analysts maintain a bullish outlook on Nvidia’s prospects. For instance, Melius Research analyst Ben Reitzes likens the company’s positioning to that of Apple Inc, highlighting enthusiasm for the Blackwell chips. Similarly, Harsh Kumar from Piper Sandler identifies Nvidia as their top large-cap choice in the artificial intelligence accelerator market.
However, history has shown that analysts are not always right. According to TipRanks data, at the end of 2023 34 different analysts have published 12-month projections for Nvidia stock, with the average price target of just over $661 per share implying a 33% upside over the next 12 months. At the time NVDA achieved record-breaking revenue of over $18 billion, a 206% year-over-year increase and 34% growth from the previous quarter. However, its valuation metrics presented mixed signals: its P/E ratio of 65 was relatively modest given its growth trajectory. However, its P/S ratio of 28 far exceeded competitors like AMD and Intel, suggesting the stock "was priced for perfection". We now know that those predictions were extremely pessimistic as NVDA's stock price nearly tripled in 2024.
Our research director shared his views on NVDA’s earnings results here. He thinks NVDA stock can reach $170 within 3 months. While we acknowledge the potential of NVDA as an investment, our conviction lies in the belief that some AI stocks hold greater promise for delivering higher returns and doing so within a shorter time frame. If you are looking for an AI stock that is more promising than NVDA but that trades at less than 5 times its earnings, check out our report about the cheapest AI stock."""]

analyzer = SentimentIntensityAnalyzer()
for sentence in sentences:
    vs = analyzer.polarity_scores(sentence)
    print("{:-<65} {}".format(sentence, str(vs)))