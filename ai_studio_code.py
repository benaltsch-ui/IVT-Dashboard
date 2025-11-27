import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import feedparser
import requests
import trafilatura
from email.utils import parsedate_to_datetime

# --- CONFIGURATION ---
st.set_page_config(page_title="Invicta Holdings Pro Dashboard", layout="wide", page_icon="🏭")

# --- CACHING & RESOURCES ---
@st.cache_resource
def load_sentiment_resources():
    try:
        nltk.data.find('sentiment/vader_lexicon.zip')
    except LookupError:
        nltk.download('vader_lexicon', quiet=True)
    
    sia = SentimentIntensityAnalyzer()
    
    # Financial Dictionary
    financial_lexicon = {
        'shoot': 2.0, 'surged': 3.0, 'jumped': 2.5, 'climbed': 2.0, 'soared': 3.0, 'green': 1.5,
        'plunged': -3.0, 'tumbled': -2.5, 'slumped': -2.5, 'red': -1.5,
        'dividend': 2.0, 'earnings': 1.5, 'profit': 2.0, 'growth': 2.0,
        'resilient': 2.0, 'strong': 2.0, 'up': 1.0, 'down': -1.0,
        'acquisition': 1.5, 'buyback': 1.5, 'challenging': -1.0, 'headwinds': -1.5
    }
    sia.lexicon.update(financial_lexicon)
    return sia

try:
    sia = load_sentiment_resources()
except Exception as e:
    st.error(f"Error loading AI: {e}")
    st.stop()

# --- HELPER FUNCTIONS ---
def format_large_number(num):
    if num is None: return "N/A"
    if num >= 1e9:
        return f"R {num/1e9:.2f} B"
    elif num >= 1e6:
        return f"R {num/1e6:.2f} M"
    else:
        return f"R {num:,.2f}"

def get_final_url(url):
    try:
        session = requests.Session()
        session.headers.update({"User-Agent": "Mozilla/5.0"})
        response = session.head(url, allow_redirects=True, timeout=5)
        return response.url
    except:
        return url

def get_article_content(url):
    try:
        downloaded = trafilatura.fetch_url(url)
        if downloaded:
            text = trafilatura.extract(downloaded)
            if text and len(text) > 200: return text
    except:
        pass
    return None

def analyze_content(text, method="Snippet"):
    score = sia.polarity_scores(text)['compound']
    
    if score >= 0.05: label, icon = "Positive", "🟢"
    elif score <= -0.05: label, icon = "Negative", "🔴"
    else: label, icon = "Neutral", "⚪"
    
    words = text.lower().split()
    drivers = []
    for word in words:
        clean_word = word.strip('.,!?"\'()')
        if clean_word in sia.lexicon:
            val = sia.lexicon[clean_word]
            if abs(val) >= 1.0: drivers.append((clean_word, val))
    
    drivers.sort(key=lambda x: abs(x[1]), reverse=True)
    unique_drivers = list(set([x[0] for x in drivers]))[:5]
    
    explanation = f"Based on {method}. Key drivers: " + ", ".join([f"**{w}**" for w in unique_drivers]) if unique_drivers else f"Based on {method}."
    return score, label, icon, explanation

# --- DATA FETCHING ---
@st.cache_data(ttl=900)
def get_market_data(ticker, period="1y"):
    try:
        stock = yf.Ticker(ticker)
        # Fetch longer history for 200 SMA calculation
        history = stock.history(period="2y") 
        info = stock.info
        financials = stock.financials
        balance_sheet = stock.balance_sheet
        return history, info, financials, balance_sheet
    except:
        return pd.DataFrame(), {}, pd.DataFrame(), pd.DataFrame()

@st.cache_data(ttl=3600) 
def get_competitor_financials():
    tickers = {"Invicta (IVT)": "IVT.JO", "Hudaco (HDC)": "HDC.JO", "Barloworld (BAW)": "BAW.JO"}
    metrics = []
    history_df = pd.DataFrame()
    
    for name, sym in tickers.items():
        try:
            stock = yf.Ticker(sym)
            hist = stock.history(period="1y")
            if not hist.empty:
                start_price = hist['Close'].iloc[0]
                hist['Growth'] = ((hist['Close'] - start_price) / start_price) * 100
                history_df[name] = hist['Growth']
                current_price = hist['Close'].iloc[-1]
            else:
                current_price = 0

            info = stock.info
            metrics.append({
                "Company": name,
                "Price": current_price,
                "P/E Ratio": info.get('trailingPE', 0),
                "Div Yield (%)": (info.get('dividendYield', 0) or 0) * 100,
                "Market Cap (B)": (info.get('marketC
