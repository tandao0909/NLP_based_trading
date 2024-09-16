import os

import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

import backtrader as bt
import backtrader.indicators as btind
import backtrader.analyzers as btanalyzers

CURRENT_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
data_sentiment = None

class Sentiment(bt.Indicator):
    lines = ('sentiment',)
    plotinfo = dict(
        plotymargin=0.5,
        plothlines=[0],
        plotyticks=[1.0, 0, -1.0])
    
    def next(self):
        self.sentiment = 0.5
        self.date = self.data.datetime
        date = bt.num2date(self.date[0]).date()
        prev_sentiment = self.sentiment
        global data_sentiment  
        if date in data_sentiment:
            self.sentiment = data_sentiment[date]
        self.lines.sentiment[0] = self.sentiment

class SentimentStrat(bt.Strategy):
    params = (
        ('period', 15),
        ('printlog', True),
    )

    def log(self, txt, dt=None, doprint=False):
        if self.params.printlog or doprint:
            dt = dt or self.datas[0].datetime.date(0)
            print(f"{dt.isoformat()}, {txt}")
    
    def __init__(self):
        self.dataclose = self.datas[0].close
        self.order = None
        self.buyprice = None
        self.buycomm = None
        self.mas = bt.indicators.MovingAverageSimple(
            self.datas[0], period=self.params.period
        )
        self.date = self.data.datetime
        self.sentiment = None
        Sentiment(self.data)
        self.plotinfo.plot = False

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
        global data_sentiment
        if date in data_sentiment:
            self.sentiment = data_sentiment[date]
        
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
        self.log(f"MA period {self.params.period} Ending Value {self.broker.getvalue()}", doprint=True)

def run_strategy(ticker, start, end):
    print(ticker)
    ticker = yf.Ticker(ticker)
    df_ticker = ticker.history(start = start, end = end)
    
    cerebro = bt.Cerebro()
    # Add the data
    cerebro.addstrategy(SentimentStrat)
    data = bt.feeds.PandasData(dataname=df_ticker)
    cerebro.adddata(data)
    start = 100000
    cerebro.broker.setcash(start)
    cerebro.addsizer(bt.sizers.FixedSize, stake=100)
    print(f'Starting Portfolio Value: {start}')
    plt.rcParams['figure.figsize']=[10,6]
    plt.rcParams["font.size"]="12"
    cerebro.run()
    cerebro.plot(volume=False, iplot=True, plotname= ticker)
    end = cerebro.broker.getvalue()
    print(f'Start Portfolio value: {start}\nFinal Portfolio Value: {end}\nProfit: {end-start}\n')
    return float(df_ticker['Close'][0]), (end - start)

if __name__ == "__main__":
    import datetime

    data_sentiment = pd.read_csv(
        f"{CURRENT_DIRECTORY}/../data/trained_data.csv",
        index_col=0,
    )
    stock_list = data_sentiment["stock"].unique()
    data_sentiment = data_sentiment["sentiment_lexicon"]
    data_sentiment = data_sentiment.to_dict()
    data_sentiment = {
        datetime.datetime.strptime(key, "%Y-%m-%d").date(): value for key, value in data_sentiment.items()
    }
    for stock in ["AAPL"]:
        run_strategy(stock, start='2012-05-22', end='2020-06-03')
