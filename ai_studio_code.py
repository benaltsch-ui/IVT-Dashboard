import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from duckduckgo_search import DDGS
from newspaper import Article
import datetime

# --- CONFIGURATION ---
st.set_page_config(page_title="Invicta Holdings Live Dashboard", layout="wide", page_icon="🏭")

# --- CACHING & MODEL LOADING ---
@st.cache_resource
def load_sentiment_resources():
    nltk.download('vader_lexicon', quiet=True)
    nltk.download('punkt', quiet=True) # Required for newspaper3k
    sia = SentimentIntensityAnalyzer()
    
    # Financial Dictionary (The "Brain" Upgrade)
    financial_lexicon = {
        'shoot': 2.0, 'surged': 3.0, 'jumped': 2.5, 'climbed': 2.0, 'soared': 3.0, 'green': 1.5,
        'plunged': -3.0, 'tumbled': -2.5, 'slumped': -2.5, 'red': -1.5,
        'dividend': 2.0, 'earnings': 1.5, 'profit': 2.0, 'growth': 2.0,
        'resilient': 2.0, 'strong': 2.0, 'up': 1.0, 'down': -1.0,
        'acquisition': 1.5, 'buyback': 1.5, 'challenging': -1.0, 'headwinds': -1.5
    }
    sia.lexicon.update(financial_lexicon)
    return sia

sia = load_sentiment_resources()

# --- NEW SCRAPER (Newspaper3k) ---
def get_article_content(url):
    """
    Uses newspaper3k to download and parse the article.
    Newspaper3k is smarter at extracting just the text body.
    """
    try:
        article = Article(url)
        article.download()
        article.parse()
        
        # If the body text is too short, the download likely failed or was blocked
        if len(article.text) < 150:
            return None
            
        return article.text
    except Exception as e:
        return None

# --- SENTIMENT LOGIC ---
def analyze_content(text, method="Snippet"):
    # 1. Score
    score = sia.polarity_scores(text)['compound']
    
    # 2. Label
    if score >= 0.05: label, icon = "Positive", "🟢"
    elif score <= -0.05: label, icon = "Negative", "🔴"
    else: label, icon = "Neutral", "⚪"
    
    # 3. Explain Drivers
    words = text.lower().split()
    drivers = []
    for word in words:
        clean_word = word.strip('.,!?"\'()')
        if clean_word in sia.lexicon:
            val = sia.lexicon[clean_word]
            if abs(val) >= 1.0: 
                drivers.append((clean_word, val))
    
    drivers.sort(key=lambda x: abs(x[1]), reverse=True)
    unique_drivers = list(set([x[0] for x in drivers]))[:5]
    
    explanation = f"Based on {method}. Key drivers: " + ", ".join([f"**{w}**" for w in unique_drivers]) if unique_drivers else f"Based on {method}."
        
    return score, label, icon, explanation

@st.cache_data(ttl=900)
def get_market_data(ticker):
    stock = yf.Ticker(ticker)
    try:
        return stock.history(period="1y"), stock.info
    except:
        return pd.DataFrame(), {}

@st.cache_data(ttl=3600)
def get_live_news_duckduckgo(query):
    """
    Uses DuckDuckGo to get DIRECT links (bypassing Google Redirects).
    """
    results = DDGS().news(keywords=f"{query} South Africa", region="za-en", safesearch="off", max_results=5)
    
    news_items = []
    
    # Progress Bar
    progress_text = "Fetching direct articles..."
    my_bar = st.progress(0, text=progress_text)
    
    if not results:
        return []

    for i, result in enumerate(results):
        my_bar.progress((i + 1) / len(results), text=f"Reading: {result['title'][:20]}...")
        
        # DuckDuckGo gives us a 'url' key which is the REAL link.
        direct_url = result['url']
        
        # 1. Try to Scrape Full Text
        full_text = get_article_content(direct_url)
        
        if full_text:
            score, label, icon, expl = analyze_content(full_text, "Full Article Text")
            snippet = full_text[:300] + "..."
            method = "✅ Full Text Read"
        else:
            # Fallback to the snippet DDG provides (usually better than RSS)
            fallback_text = f"{result['title']}. {result['body']}"
            score, label, icon, expl = analyze_content(fallback_text, "Search Snippet")
            snippet = result['body']
            method = "⚠️ Snippet Only (Blocked)"

        news_items.append({
