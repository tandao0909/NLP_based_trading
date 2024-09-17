# pylint: disable=invalid-name
import sys
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import en_core_web_sm
from textblob import TextBlob
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import mean_squared_error, confusion_matrix, classification_report

from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import pad_sequences
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Add the parent directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from preprocessing.process import prepare_train_data

CURRENT_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
SCORING = "neg_mean_squared_error"
tokenizer = None

def add_text_blob(train_data:pd.DataFrame) -> pd.DataFrame:
    train_data["sentiment_textblob"] = [TextBlob(text).sentiment.polarity for text in train_data["headline"]]

    plt.figure(figsize=(8, 8))
    plt.scatter(train_data["sentiment_textblob"], train_data["event_return"])
    plt.xlabel("Sentiment")
    plt.ylabel("Event return")

    plt.savefig(f"{CURRENT_DIRECTORY}/../images/TextBlob.png")

    return train_data

def prepare_sentiment_data(sentiment_data_path:str=None) -> pd.DataFrame:
    if sentiment_data_path is None:
        sentiment_data = pd.read_csv(f"{CURRENT_DIRECTORY}/../data/stock_news.csv")
    else:
        sentiment_data = pd.read_csv(sentiment_data_path)
    
    sentiment_data["label"] = sentiment_data["label"].map({
        "Negative": 0,
        "Neutral": 0.5,
        "Positive": 1
        })
    return sentiment_data

def prepare_train_data_for_supervised_models(
        sentiment_data:pd.DataFrame,
        valid_size=0.2,
        seed=42
        ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    nlp = en_core_web_sm.load()
    all_vectors = np.array(
        [np.array(
            [token.vector for token in nlp(headline)]
        ).mean(axis=0) for headline in sentiment_data["headline"]]
    )
    encoder = LabelEncoder()

    X = pd.DataFrame(all_vectors)
    y = sentiment_data["label"]
    y = encoder.fit_transform(y)

    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=valid_size, random_state=seed
    )
    return X_train, X_valid, y_train, y_valid
    
def prepare_model_list() -> list[tuple[str, BaseEstimator]]:    
    models = []

    models.append(('LR', LogisticRegression()))
    models.append(('KNN', KNeighborsClassifier()))
    models.append(('CART', DecisionTreeClassifier()))
    models.append(('SVM', SVC()))

    # Neural Network
    models.append(('NN', MLPClassifier()))

    # Ensemble Model
    models.append(('RF', RandomForestClassifier()))
    return models

def plot_model_comparison(
        names:list[int],
        train_results:list[float],
        valid_results:list[float]
        ) -> None:

    plt.figure(figsize=(15, 8))
    index = np.arange(len(names))
    width = 0.3
    plt.bar(index - width / 2, train_results, width=width, label="Train MSE")
    plt.bar(index + width / 2, valid_results, width=width, label="Valid MSE")

    plt.legend()
    plt.xticks(ticks=range(len(names)), labels=names)

    plt.savefig(f"{CURRENT_DIRECTORY}/../images/model_comparison.png")

def train_supervised_models(sentiment_data: pd.DataFrame, num_folds:int=10, seed:int=42, train_LSTM=True):
    models = prepare_model_list()
    X_train, X_valid, y_train, y_valid = prepare_train_data_for_supervised_models(sentiment_data)

    results = []
    names = []
    valid_results = []
    train_results = []

    for name, model in models:
        k_fold = KFold(num_folds, shuffle=True, random_state=seed)
        cv_results = cross_val_score(model, X_train, y_train, cv=k_fold, scoring=SCORING)
        results.append(cv_results)
        names.append(name)

        result = model.fit(X_train, y_train)
        train_result = mean_squared_error(result.predict(X_train), y_train)
        train_results.append(train_result)

        valid_result = mean_squared_error(result.predict(X_valid), y_valid)
        valid_results.append(valid_result)

        print(f"{name}: ")
        print(confusion_matrix(result.predict(X_valid), y_valid))
        print(classification_report(result.predict(X_valid), y_valid))
    if train_LSTM:
        LSTM_model = train_LSTM_model()
        X_train_LSTM, X_valid_LSTM, y_train_LSTM, y_valid_LSTM = prepare_LSTM_data(sentiment_data)


        train_result_LSTM = mean_squared_error(LSTM_model.predict(X_train_LSTM), y_train_LSTM)
        valid_result_LSTM = mean_squared_error(LSTM_model.predict(X_valid_LSTM), y_valid_LSTM)

        results.append(None)
        train_results.append(train_result_LSTM)
        valid_results.append(valid_result_LSTM)

        names.append("LSTM")

    return results, names, train_results, valid_results

def prepare_LSTM_data(sentiment_data:pd.DataFrame, vocabulary_size=20000, valid_size=0.2, seed=42):
    global tokenizer 
    tokenizer = Tokenizer(num_words=vocabulary_size)
    tokenizer.fit_on_texts(sentiment_data["headline"])
    sequences = tokenizer.texts_to_sequences(sentiment_data["headline"])
    X_LSTM = pad_sequences(sequences, maxlen=max([len(x) for x in sequences]))

    encoder = LabelEncoder()

    y = sentiment_data["label"]
    y = encoder.fit_transform(y)

    X_train_LSTM, X_valid_LSTM, y_train_LSTM, y_valid_LSTM = train_test_split(X_LSTM, y, test_size=valid_size, shuffle=True, random_state=seed)
    return X_train_LSTM, X_valid_LSTM, y_train_LSTM, y_valid_LSTM

def train_LSTM_model(vocabulary_size=20000):
    X_train_LSTM, _, y_train_LSTM, _ = prepare_LSTM_data(prepare_sentiment_data())

    LSTM_model = Sequential(
        [
            Embedding(vocabulary_size, 300),
            LSTM(100, dropout=0.2, recurrent_dropout=0.2),
            Dense(1, activation="sigmoid")
        ]
    )
    LSTM_model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    LSTM_model.fit(X_train_LSTM, y_train_LSTM, epochs=3)
    return LSTM_model

def add_lstm_sentiment(train_data) -> pd.DataFrame:
    global tokenizer
    LSTM_model = train_LSTM_model()
    sequences_LSTM = tokenizer.texts_to_sequences(train_data["headline"])
    X_LSTM = pad_sequences(sequences_LSTM, maxlen=max([len(x) for x in sequences_LSTM]))
    y_LSTM = LSTM_model.predict(X_LSTM)
    train_data["sentiment_lstm"] = y_LSTM
    return train_data

def add_sia_sentiment(train_data) -> pd.DataFrame:
    sia = SentimentIntensityAnalyzer()
    stock_lex = pd.read_csv(f"{CURRENT_DIRECTORY}/../data/LexiconData.csv")
    stock_lex["sentiment"] = (stock_lex["Aff_Score"] + stock_lex["Neg_Score"]) / 2
    stock_lex = dict(zip(stock_lex["Item"], stock_lex["sentiment"]))
    stock_lex = {k:v for k,v in stock_lex.items() if len(k.split(" ")) == 1}
    stock_lex_scaled = {}
    for k, v in stock_lex.items():
        if v > 0:
            stock_lex_scaled[k] = v / max(stock_lex.values()) * 4
        else:
            stock_lex_scaled[k] = v / min(stock_lex.values()) * -4

    final_lex = {}
    final_lex.update(stock_lex_scaled)
    final_lex.update(sia.lexicon)
    sia.lexicon = final_lex
    vader_sentiment = np.array([sia.polarity_scores(headline)["compound"] for headline in train_data["headline"]])
    train_data["sentiment_lexicon"] = vader_sentiment
    return train_data

def plot_compare_correlation_sentiment_methods(trained_data):
    correlation = trained_data[["sentiment_textblob", "sentiment_lstm", "sentiment_lexicon", "event_return"]].dropna(axis=0).corr()
    plt.figure(figsize=(8, 8))
    sns.heatmap(correlation[["event_return"]], vmax=1, annot=True)
    plt.savefig(f"{CURRENT_DIRECTORY}/../images/compare_correlation_sentiment_methods.png")
    return None

def plot_compare_sentiment_methods(trained_data):
    correlation_data = []
    for ticker in trained_data["stock"].unique():
        trained_data_stock = trained_data[trained_data["stock"] == ticker]
        corr_textblob = trained_data_stock["event_return"].corr(trained_data_stock["sentiment_textblob"])
        corr_lstm = trained_data_stock["event_return"].corr(trained_data_stock["sentiment_lstm"])
        corr_lexicon = trained_data_stock["event_return"].corr(trained_data_stock["sentiment_lexicon"])
        correlation_data.append([ticker, corr_textblob, corr_lstm, corr_lexicon])

    correlation_df = pd.DataFrame(correlation_data, columns=["ticker", "corr_textblob", "corr_lstm", "corr_lexicon"])
    correlation_df.set_index("ticker", inplace=True)
    correlation_df.fillna(0, inplace=True)

    correlation_df.plot.bar(figsize=(10, 8))
    plt.savefig(f"{CURRENT_DIRECTORY}/../images/compare_sentiment_methods.png")
    return None
    
if __name__ == "__main__":
    train_data = prepare_train_data()
    print("Complete prepare train data")
    train_data = add_text_blob(train_data)
    print("Complete add text blob to train data")
    results, names, train_results, valid_results = train_supervised_models(prepare_sentiment_data())
    plot_model_comparison(names, train_results, valid_results)
    train_data = add_lstm_sentiment(train_data)
    print("Complete add lstm to train data")
    train_data = add_sia_sentiment(train_data)
    print("Complete add sia to train data")
    train_data.to_csv(f"{CURRENT_DIRECTORY}/../data/trained_data.csv")
    print("Complete add trained data to data/ directory")
    plot_compare_correlation_sentiment_methods(train_data)
    plot_compare_sentiment_methods(train_data)
