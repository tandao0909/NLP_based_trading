# pylint: disable=redefined-outer-name

import os
from io import StringIO
from datetime import date

import zipfile
import json
from lxml import etree
import pandas as pd

CURRENT_DIRECTORY = os.path.dirname(os.path.abspath(__file__))

def json_parser(json_data:dict) -> tuple[list[str], list[str]]:
    xml_data = json_data["content"]

    tree = etree.parse(StringIO(xml_data), parser=etree.HTMLParser())

    headlines = tree.xpath("//h4[contains(@class, 'media-heading')]/a/text()")
    assert len(headlines) == json_data["count"]

    main_tickers = [x.replace("/symbol/", "") 
                    for x in tree.xpath("//div[contains(@class, 'media-left')]//a/@href")]
    assert len(main_tickers) == json_data["count"]

    final_headlines = [''.join(f.xpath(".//text()"))
                       for f in tree.xpath("//div[contains(@class, 'media-body')]")]
    final_headlines = [f.replace(h, "").split("\xa0")[0].strip() 
                       for f, h in zip(final_headlines, headlines)]

    return main_tickers, final_headlines

def process_news_data_zip(zipfile_path:str) -> pd.DataFrame:
    news_data = pd.DataFrame()
    with zipfile.ZipFile(zipfile_path, "r") as z:
        for filename in z.namelist():
            with z.open(filename) as file:
                data = file.read()
                json_data = json.loads(data)
            if json_data.get("count", 0) > 0:
                main_tickers, final_headlines = json_parser(json_data)
                if len(final_headlines) != json_data["count"]:
                    continue
                file_date = filename.split("/")[-1].replace(".json", "")
                file_date = date(int(file_date[:4]), int(file_date[5:7]), int(file_date[8:]))

                df = pd.DataFrame({
                    "stock": main_tickers,
                    "headlines": final_headlines,
                    "date": [file_date] * len(main_tickers)
                })
                news_data = pd.concat([news_data, df])
    news_data = news_data.dropna()
    return news_data

def prepare_news_data_zip(news_data_zip:pd.DataFrame=None) -> pd.DataFrame:
    filepath = f"{CURRENT_DIRECTORY}/../data/zip_news_data.csv"

    if news_data_zip is None:
        # Not given any data, but has on disk, then read from disk
        if os.path.exists(filepath):
            news_data_zip = pd.read_csv(filepath, index_col=0, parse_dates=True, date_format="%Y-%m-%d")
        # If not on disk, process from scratch
        else:
            news_data_zip = process_news_data_zip(f"{CURRENT_DIRECTORY}/../data/Raw Headline Data.zip")

            news_data_zip["date"] = pd.to_datetime(news_data_zip["date"])
            news_data_zip = news_data_zip.set_index("date")
            news_data_zip = news_data_zip.rename({"headlines": "headline"}, axis=1)
            news_data_zip.to_csv(filepath)
    return news_data_zip

if __name__ == "__main__":
    news_data_zip = prepare_news_data_zip()
    print(news_data_zip.head(5))
