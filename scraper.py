import requests
from urllib.parse import quote
import yfinance as yf
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
import time
from transformers import pipeline
from langdetect import detect

# Download VADER lexicon if not already downloaded
nltk.download('vader_lexicon')

# Initialize Sentiment Analyzer
sia = SentimentIntensityAnalyzer()

# Initialize Hugging Face zero-shot classification pipeline
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# Headers to mimic a browser visit
headers = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_2) AppleWebKit/601.3.9 (KHTML, like Gecko) Version/9.0.2 Safari/601.3.9'
}

# Check if stock ticker is valid
def is_valid_ticker(ticker):
    try:
        stock = yf.Ticker(ticker.upper())
        return stock.info.get('symbol', '').upper() == ticker.upper()
    except:
        return False

# Fetch posts from a subreddit
def fetch_reddit_posts(stock, time_filter, subreddit):
    url = f'https://www.reddit.com/r/{subreddit}/search.json?q={quote(stock)}&sort=top&t={time_filter}&limit=50'
    try:
        response = requests.get(url, headers=headers)
        if response.status_code != 200:
            print(f"Failed to get posts from {subreddit}: {response.status_code}")
            return []
        return response.json().get('data', {}).get('children', [])
    except:
        print(f"Error getting posts from {subreddit}: {e}")
        return []

# Filter posts for relevance using Hugging Face zero-shot classification
def is_relevant_post(content, stock):
    labels = [f"related to {stock} stock analysis", "not related to stock analysis"]
    result = classifier(content, labels)
    return result["labels"][0] == f"related to {stock} stock analysis"

# Detect if content is in English
def is_english(content):
    try:
        return detect(content) == "en"
    except:
        return False

# Remove duplicate posts based on URL
def remove_duplicates(posts):
    unique_urls = set()
    unique_posts = []
    for post in posts:
        if post['url'] not in unique_urls:
            unique_urls.add(post['url'])
            unique_posts.append(post)
    return unique_posts

# Main function
if __name__ == "__main__":
    stock = input("Enter the stock symbol or keyword to search: ").strip().upper()
    time_filter = input("Enter time filter (day, week, month, year, all): ").strip().lower()

    if time_filter not in ["day", "week", "month", "year", "all"] or not is_valid_ticker(stock):
        print("Invalid input. Please try again.")
        exit()

    subreddits = [
        'wallstreetbets', 'pennystocks', 'valueinvesting',
        'investing', 'stockmarket', 'stocksandtrading',
        'robinhoodpennystocks', 'wallstreetbetselite', 
        'shortsqueeze', 'dividends'
    ]
    posts_data = []

    for subreddit in subreddits:
        print(f"\nSearching in r/{subreddit}...")
        posts = fetch_reddit_posts(stock, time_filter, subreddit)

        for post in posts:
            post_data = post.get('data', {})
            title = post_data.get('title', 'No Title')
            post_url = f"https://www.reddit.com{post_data.get('permalink', '')}"
            content = post_data.get('selftext', 'No Content')

            #skip posts with no content or less than 50 words
            if content == 'No Content' or len(content.split()) < 50:
                continue

            #skip non-English 
            if not is_english(content):
                print(f"...")
                continue

            #Skip posts that are not relevant to the stock using Hugging Face model (transformers)
            if not is_relevant_post(content, stock):
                print(f"...")
                continue

            #analyze content sentiment for post that made it.
            content_sentiment = sia.polarity_scores(content)

            #append data
            posts_data.append({
                'subreddit': subreddit,
                'title': title,
                'url': post_url,
                'content_sentiment': content_sentiment
            })

        time.sleep(2) #so i dont get blocked.

    #remove dupes
    posts_data = remove_duplicates(posts_data)

    #display
    if posts_data:
        print("\n--- Analysis Results ---\n")
        for post in posts_data:
            print(f"Subreddit: r/{post['subreddit']}")
            print(f"Title: {post['title']}")
            print(f"Post URL: {post['url']}")
            print(f"Content Sentiment: {post['content_sentiment']}")
            print("-" * 80)
    else:
        print("\nNo valid posts found.")
