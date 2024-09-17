# pylint: disable=redefined-outer-name
import os
import pdb

import pandas as pd
from .yahoo import historical_stocks_data
from .process_zip_file import prepare_news_data_zip

# 10 largest market cap companies in the USA
MAIN_TICKERS = ["AAPL", "MSFT", "NVDA", "GOOG", "AMZN", "META", "BRK-B", "TSM", "LLY", "TSLA", "AVGO"]

CURRENT_DIRECTORY = os.path.dirname(os.path.abspath(__file__))

def prepare_ticker_data(stock_list: list[str]=None) -> pd.DataFrame:
    if stock_list is None:
        stock_list = MAIN_TICKERS

    filepath = f"{CURRENT_DIRECTORY}/../data/close_price.csv"
    if os.path.exists(filepath):
        ticker_data = pd.read_csv(filepath, index_col=0, parse_dates=True, date_format="%Y-%m-%d")
    else:
        os.makedirs(f"{CURRENT_DIRECTORY}/../data", exist_ok=True)
        ticker_data = historical_stocks_data(stock_list, years_to_look_back=20)
        ticker_data.to_csv(filepath)

    ticker_data = ticker_data.pct_change()
    ticker_data = ticker_data.shift(1) + ticker_data + ticker_data.shift(-1)
    return ticker_data

def prepare_news_data(stock_list: list[str]=None, filepath:str=None) -> pd.DataFrame:
    if stock_list is None:
        stock_list = MAIN_TICKERS
    if filepath is None:
        filepath = f"{CURRENT_DIRECTORY}/../data/raw_partner_headlines.csv"

    news_data = pd.read_csv(filepath, index_col=0, parse_dates=True, date_format="%Y-%m-%d %H:%M:%S")

    news_data = news_data[news_data["stock"].isin(stock_list)]
    news_data.index = pd.to_datetime(news_data["date"])
    news_data.index = news_data.index.strftime("%Y-%m-%d")
    news_data.index = pd.to_datetime(news_data["date"])
    news_data = news_data[["stock", "headline"]]
    return news_data

def prepare_train_data(
        ticker_data:pd.DataFrame=None,
        news_data:pd.DataFrame=None,
        stock_list: list[str]=None
        ) -> pd.DataFrame:
    
    filepath = f"{CURRENT_DIRECTORY}/../data/train_data.csv"

    if os.path.exists(filepath):
        merged_data = pd.read_csv(filepath, index_col=0, parse_dates=True, date_format="%Y-%m-%d")
    else:
        os.makedirs(f"{CURRENT_DIRECTORY}/../data", exist_ok=True)

        if stock_list is None:
            stock_list = MAIN_TICKERS

        if ticker_data is None:
            ticker_data = prepare_ticker_data()
        if news_data is None:
            news_data = prepare_news_data()

        news_data.index = news_data.index.astype(str)
        ticker_data.index = ticker_data.index.astype(str)
        news_data = news_data[news_data["stock"].isin(stock_list)]

        merged_data = pd.merge(ticker_data, news_data, how="right", left_index=True, right_index=True).dropna()

        def select_ticker(row):
            return row[row["stock"]]
        merged_data["event_return"] = merged_data.apply(select_ticker, axis=1)
        merged_data = merged_data.drop(columns=stock_list)

        if "date" in merged_data.columns:
            merged_data = merged_data.drop(columns=["date"])
        merged_data.to_csv(filepath)
    return merged_data

if __name__ == "__main__":
    news_data_zip = prepare_news_data_zip()
    news_data = prepare_news_data()
    news_data = pd.concat([news_data, news_data_zip]).dropna()
    merged_data = prepare_train_data(news_data=news_data)
    print(merged_data.head(5))
