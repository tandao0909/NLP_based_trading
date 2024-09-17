import os
import datetime

import streamlit as st
from PIL import Image

from backtrade.backtrade import run_strategy

if 'check' not in st.session_state:
    st.session_state.check = False

st.write("Please choose the date so that the start date is before the end date.")

start_date = st.date_input("Select your date to start trade",
              min_value=datetime.date(2012, 5, 22),
              max_value=datetime.date(2020, 3, 3),
              value=datetime.date(2012, 5, 22))
end_date = st.date_input("Select your date to end trade",
              min_value=start_date,
              max_value=datetime.date(2020, 6, 3),
              value=datetime.date(2020, 6, 3))

stock = st.selectbox("Please choose the ticker of your company:", ["AAPL", "MSFT", "NVDA", "GOOG", "AMZN", "META", "BRK-B", "TSM", "LLY", "TSLA", "AVGO"], index=1)

if st.button("I want to see the trading chart!"):
    st.session_state.check = True

if st.session_state.check:
    back_test_image, profit = run_strategy(stock, start_date, end_date)
    st.title("Back test")
    st.text("""
    In the chart below, there are 3 panels:
    - The top panel is the cash value observer. It keeps track of the cash and \nthe total portfolio value during the life of the back-testing run. In this run, \nwe always started with $100,000.
    - The next panel is the trade observer. It shows the realized profit/loss \nof each trade. A trade is defined as opening a position and taking the position \nback to zero (directly or crossing over from long to short or short to long). \nThe blue marks are profitable trade while the red ones are not.
    - The last panel is buy sell observer. It indicates where buy and sell operations \nhave taken place. In general, we see that the buy action takes place when the stock \nprice is increasing, and the sell action takes place when the stock price has started
declining.
    """)
    st.pyplot(back_test_image)
    st.write(f"The profit is {profit:.2f}$, or the nominal profit is {profit/10**5*100:.5f}%")
