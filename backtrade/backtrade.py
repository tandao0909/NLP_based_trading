import os
import datetime

import pandas as pd
import matplotlib.pyplot as plt

import backtrader as bt

CURRENT_DIRECTORY = os.path.dirname(os.path.abspath(__file__))

def prepare_data_sentiment():
    data_sentiment = pd.read_csv(
        f"{CURRENT_DIRECTORY}/../data/trained_data.csv",
        index_col=0,
    )
    data_sentiment = data_sentiment["sentiment_lexicon"]
    data_sentiment = data_sentiment.to_dict()
    data_sentiment = {
        datetime.datetime.strptime(key, "%Y-%m-%d").date(): value for key, value in data_sentiment.items()
    }
    return data_sentiment

def prepare_ticker_data(stock, start, end) -> pd.DataFrame:
    filepath = f"{CURRENT_DIRECTORY}/../data/ticker_data.csv"
    ticker_data = pd.read_csv(filepath, index_col=0, parse_dates=True)
    if isinstance(start, str):
        start = datetime.datetime.strptime(start, "%Y-%m-%d")
    if isinstance(end, str):
        end = datetime.datetime.strptime(end, "%Y-%m-%d")
        
    start = pd.Timestamp(start).tz_localize("UTC")
    end = pd.Timestamp(end).tz_localize("UTC")
    ticker_data_stock = ticker_data[ticker_data["stock"] == stock]
    return ticker_data_stock[start:end]

class SentimentStrat(bt.Strategy):
    def log(self, txt, dt=None, doprint=False):
        if self.params["printlog"] or doprint:
            dt = dt or self.datas[0].datetime.date(0)
            print(f"{dt.isoformat()}, {txt}")
    
    def __init__(self):
        self.params = {
            'period': 15,
            'printlog': True,
        }
        self.dataclose = self.datas[0].close
        self.order = None
        self.buyprice = None
        self.buycomm = None
        self.mas = bt.indicators.MovingAverageSimple(
            self.datas[0], period=self.params["period"]
        )
        self.date = self.data.datetime
        self.sentiment = None
        self.plotinfo.plot = False
        self.data_sentiment = prepare_data_sentiment()
        self.bar_executed = 0

    def notify_order(self, order:bt.Order):
        if order.status in [order.Submitted, order.Accepted]:
            # Buy/Sell order submitted/accepted to/by broker, do nothing
            return
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(
                    f"BUY EXECUTED, Price : {order.executed.price}, Cost: {order.executed.value}, Commission {order.executed.comm}"
                )
                self.buyprice = order.executed.price
                self.buycomm = order.executed.comm
            else: # Buy
                self.log(
                    f"BUY EXECUTED, Price : {order.executed.price}, Cost: {order.executed.value}, Commission {order.executed.comm}"
                )
            self.bar_executed = len(self)
        
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log("Order Canceled/Margin/Rejected")
        
        self.order = None
    
    def notify_trade(self, trade:bt.Trade):
        if not trade.isclosed:
            return

        self.log(
            f"Operation Profit, Gross: {trade.pnl}, Net: {trade.pnlcomm}"
        )

    def next(self):
        date = bt.num2date(self.date[0]).date()
        prev_sentiment = self.sentiment
        if date in self.data_sentiment:
            self.sentiment = self.data_sentiment[date]
        
        # Check if there is an order pending. If yes, we cannot send a 2nd one
        if self.order:
            return
        # If not in the market and the previous sentiment is not None
        if not self.position and prev_sentiment:
            # buy if current close more than mas and sentiment increased by >= 0.5
            if self.dataclose[0] > self.mas[0] and self.sentiment - prev_sentiment >= 0.5:
                self.log(f"Previous Sentiment {prev_sentiment} New Sentiment {self.sentiment} BUY CREATE {self.dataclose[0]}")
                self.order = self.buy()
        # Already in the market and the previous sentiment is not None
        elif prev_sentiment:
            # sell if current close less than mas and sentiment decreased by >= 0.5
            if self.dataclose[0] < self.mas[0] and prev_sentiment - self.sentiment >= 0.5:
                self.log(f"Previous Sentiment {prev_sentiment} New Sentiment {self.sentiment} SELL CREATE {self.dataclose[0]}")
                self.order = self.sell()

    def stop(self):
        self.log(
            f"MA period {15} Ending Value {self.broker.getvalue()}",
            doprint=True)

def run_strategy(ticker, start, end):
    print(ticker)
    df_ticker = prepare_ticker_data(ticker, start, end)
    os.makedirs(f"{ticker}", exist_ok=True)
    cerebro = bt.Cerebro()
    # Add the data
    cerebro.addstrategy(SentimentStrat)
    data = bt.feeds.PandasData(dataname=df_ticker)
    cerebro.adddata(data)
    start_amount = 100000
    cerebro.broker.setcash(start_amount)
    cerebro.addsizer(bt.sizers.FixedSize, stake=100)
    print(f'Starting Portfolio Value: {start_amount}')
    plt.rcParams['figure.figsize']=[10, 8]
    plt.rcParams["font.size"]="12"
    cerebro.run()
    figs = cerebro.plot(volume=False, iplot=True, plotname=ticker, show_fig=False)
    end_amount = cerebro.broker.getvalue()
    print(f'Start Portfolio value: {start_amount}\nFinal Portfolio Value: {end_amount}\nProfit: {end_amount-start_amount}\n')
    return figs[0][0], (end_amount - start_amount)

if __name__ == "__main__":
    data_sentiment = pd.read_csv(
        f"{CURRENT_DIRECTORY}/../data/trained_data.csv",
        index_col=0,
    )
    stock_list = data_sentiment["stock"].unique()
    best_profit = 0
    best_stock = None
    for stock in stock_list:
        fig, profit = run_strategy(stock, start='2012-05-22', end='2020-06-03')
        if best_profit < profit:
            best_profit = profit
            best_stock = stock
        fig.savefig(f"{stock}/{stock}_backtest.png")
    print(best_profit, best_stock)