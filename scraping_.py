"""
Scraping Module - Week 1
Drug Consumption Data Collection
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime
import os

def scrape_drug_news(url="https://news.google.com/search?q=drug%20consumption&hl=en-US&gl=US&ceid=US:en"):
    """Scrape news about drug consumption"""
    headers = {"User-Agent": "Mozilla/5.0"}
    print(f"Fetching news from {url}...")
    
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        print("Failed to retrieve page")
        return None
    
    soup = BeautifulSoup(response.content, "html.parser")
    news_items = []
    
    for article in soup.find_all("article"):
        title_tag = article.find("h3")
        if title_tag:
            headline = title_tag.text.strip()
            link_tag = title_tag.find("a")
            link = "https://news.google.com" + link_tag["href"][1:] if link_tag else None
            time_tag = article.find("time")
            timestamp = time_tag["datetime"] if time_tag and time_tag.has_attr("datetime") else datetime.now().strftime("%Y-%m-%d")
            news_items.append({"headline": headline, "timestamp": timestamp, "link": link})
    
    return pd.DataFrame(news_items)

if __name__ == "__main__":
    df = scrape_drug_news()
    if df is not None and not df.empty:
        print(f"Scraped {len(df)} articles")
        os.makedirs("data", exist_ok=True)
        df.to_csv("data/drug_news.csv", index=False)
        print("Saved to data/drug_news.csv")
    else:
        print("No articles found")