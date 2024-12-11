import daft  # For reading Parquet files

# Parquet news
df = daft.read_parquet("../data/latest/finance_news.parquet")
python_list = df.collect().to_pylist()
pandas_df = df.to_pandas()
column_list = pandas_df["tickerSymbols"].tolist()
print(column_list[1])

# Parquet prices
# df = daft.read_parquet("../crawler/stock_data.parquet")
# python_list = df.collect().to_pylist()
# pandas_df = df.to_pandas()
# column_list = pandas_df["Open"].tolist()
# print(column_list)