import requests
import os
from dotenv import load_dotenv

load_dotenv()

def search_brave_news(query, api_key, count=5, country="us", lang="en"):
    url = "https://api.search.brave.com/res/v1/news/search"
    headers = {
        "Accept": "application/json",
        "Accept-Encoding": "gzip",
        "X-Subscription-Token": api_key
    }
    params = {
        "q": query,
        "count": count,
        "country": country,
        "search_lang": lang,
        "spellcheck": 1
    }

    response = requests.get(url, headers=headers, params=params)

    if response.status_code == 200:
        return response.json()  # Return JSON response
    else:
        print(f"Error: {response.status_code} - {response.text}")
        return None

# Example usage
API_KEY = os.getenv("BRAVE_API_KEY")
query = "Cody Ko news"
news_results = search_brave_news(query, API_KEY)

if news_results:
    print(news_results)
    breakpoint()
