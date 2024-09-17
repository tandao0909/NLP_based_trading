import os
import sys
import pandas as pd
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from preprocessing.yahoo import historical_stocks_data

ticker_data:pd.DataFrame = historical_stocks_data(["AAPL", "MSFT", "NVDA", "GOOG", "AMZN", "META", "BRK-B", "TSM", "LLY", "TSLA", "AVGO"])
ticker_data.to_csv("./data/ticker_data.csv")
