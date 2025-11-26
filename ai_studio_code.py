import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import feedparser
import datetime
import requests
from bs4 import BeautifulSoup
import re

# --- CONFIGURATION ---
st.set_page_config(page_title="Invicta Holdings Live Dashboard", layout="wide", page_icon="🏭")

# --- CACHING & MODEL LOADING ---
@st.cache_resource
def load_sentiment_resources():
    nltk.download('vader_lexicon', quiet=True)
    sia = SentimentIntensityAnalyzer()
    return sia

sia = load_sentiment_resources()

# --- WEB SCRAPER ENGINE ---
def scrape_full_text(url):
    """
    Visits the URL and attempts to extract the main article body.
    Includes headers to mimic a real browser to avoid being blocked.
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    
    try:
        # Timeout set to 3 seconds to keep app snappy
        response = requests.get(url, headers=headers, timeout=3, allow_redirects=True)
        
        # If successfully downloaded
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove script and style elements (junk)
            for script in soup(["script", "style", "nav", "footer", "header"]):
                script.decompose()
            
            # Extract text from paragraph tags (most articles use <p>)
            paragraphs = [p.get_text() for p in soup.find_all('p')]
            full_text = ' '.join(paragraphs)
            
            # Clean up whitespace
            full_text = ' '.join(full_text.split())
            
            # Validation: If text is too short, it's likely a captcha or paywall
            if len(full_text) < 200:
                return None
                
            return full_text
            
    except Exception as e:
        # If scraping fails, return None to trigger fallback
        return None
        
    return None

# --- SENTIMENT LOGIC ---
def analyze_content(text, method="Snippet"):
    """
    Analyzes text and returns: Score, Label, Icon, Explanation
    """
    # VADER Scoring
    score = sia.polarity_scores(text)['compound']
    
    # Labeling
    if score >= 0.05: label, icon = "Positive", "🟢"
    elif score <= -0.05: label, icon = "Negative", "🔴"
    else: label, icon = "Neutral", "⚪"
    
    # Keyword Extraction (Why is it Pos/Neg?)
    words = text.lower().split()
    drivers = []
    
    # Check words against VADER dictionary
    for word in words:
        clean_word = word.strip('.,!?"\'')
        if clean_word in sia.lexicon:
            val = sia.lexicon[clean_word]
            # We filter for 'strong' emotional words to make the summary useful
            if abs(val) > 1.0: 
                drivers.append((clean_word, val))
    
    # Sort by impact
    drivers.sort(key=lambda x: abs(x[1]), reverse=True)
    
    # Create Explanation String
    unique_drivers = list(set([x[0] for x in drivers]))[:5] # Top 5 unique words
    if unique_drivers:
        explanation = f"Based on {method}. Key drivers: " + ", ".join([f"**{w}**" for w in unique_drivers])
    else:
        explanation = f"Based on {method}. No strong emotional keywords found."
        
    return score, label, icon, explanation

@st.cache_data(ttl=900)
def get_market_data(ticker):
    stock = yf.Ticker(ticker)
    try:
        history = stock.history(period="1y")
        info = stock.info
    except:
        history = pd.DataFrame()
        info = {}
    return history, info

@st.cache_data(ttl=3600)
def get_live_news_deep(query):
    encoded_query = query.replace(" ", "%20")
    rss_url = f"https://news.google.com/rss/search?q={encoded_query}+South+Africa&hl=en-ZA&gl=ZA&ceid=ZA:en"
    
    feed = feedparser.parse(rss_url)
    news_items = []
    
    if not feed.entries:
        return []

    # Process Top 8 articles (Reduced from 12 to save time on scraping)
    progress_bar = st.progress(0)
    total = min(len(feed.entries), 8)
    
    for i, entry in enumerate(feed.entries[:8]):
        # Update progress bar
        progress_bar.progress((i + 1) / total)
        
        # 1. Try to scrape Full Text
        full_text = scrape_full_text(entry.link)
        
        # 2. Analyze
        if full_text:
            # If scrape worked, analyze full text
            score, label, icon, explanation = analyze_content(full_text, method="Full Article Text")
            snippet = full_text[:300] + "..." # Show first 300 chars as preview
        else:
            # Fallback to RSS snippet
            raw_summary = entry.get('description', '') or entry.title
            clean_summary = re.sub('<.*?>', '', raw_summary)
            score, label, icon, explanation = analyze_content(clean_summary, method="Headline/Snippet")
            snippet = clean_summary

        news_items.append({
            "date": entry.published[:16],
            "title": entry.title,
            "snippet": snippet,
            "link": entry.link,
            "source": entry.source.title if hasattr(entry, 'source') else "Google News",
            "Sentiment": label,
            "Icon": icon,
            "Score": score,
            "Explanation": explanation
        })
    
    progress_bar.empty() # Remove bar when done
    return news_items

# --- MAIN APP ---
def main():
    st.title("🏭 Invicta Holdings (IVT) | Deep Sentiment Monitor")
    st.markdown("This tool uses **AI Web Scraping** to read full news articles for deeper sentiment context.")
    st.divider()

    # Sidebar
    st.sidebar.header("Configuration")
    ticker = st.sidebar.text_input("JSE Ticker", "IVT.JO")
    
    # 1. FETCH DATA
    history, info = get_market_data(ticker)
    
    st.write("🔄 **Scanning and reading news articles... (This takes a few seconds)**")
    
    # We run the deep search here
    news_data = get_live_news_deep("Invicta Holdings Limited")
    if not news_data:
            st.warning("No direct news found. Scanning sector peers...")
            news_data = get_live_news_deep("JSE Industrial Engineering")

    # 2. PROCESS DATAFRAME
    if news_data:
        df_news = pd.DataFrame(news_data)
        avg_sentiment = df_news['Score'].mean()
    else:
        df_news = pd.DataFrame()
        avg_sentiment = 0

    # 3. KPI DISPLAY
    if not history.empty:
        current_price = history['Close'].iloc[-1]
        prev_price = history['Close'].iloc[-2]
        delta = ((current_price - prev_price) / prev_price) * 100
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Share Price (ZAR)", f"{current_price:.2f}", f"{delta:.2f}%")
        col2.metric("Market Cap", f"R {info.get('marketCap', 0)/1e9:.2f} B")
        col3.metric("PE Ratio", f"{info.get('trailingPE', 'N/A')}")
        
        sent_label = "Bullish" if avg_sentiment > 0.05 else "Bearish" if avg_sentiment < -0.05 else "Neutral"
        col4.metric("Deep Sentiment", sent_label, f"{avg_sentiment:.2f} Score")
    else:
        st.error("Check Ticker Symbol.")

    # 4. CHARTS
    c1, c2 = st.columns([2, 1])
    with c1:
        st.subheader("📈 Price History (1 Year)")
        if not history.empty:
            fig = go.Figure(data=[go.Candlestick(x=history.index,
                            open=history['Open'], high=history['High'],
                            low=history['Low'], close=history['Close'])])
            fig.update_layout(height=350, margin=dict(l=0,r=0,t=20,b=0))
            st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.subheader("🧠 Sentiment Split")
        if not df_news.empty:
            counts = df_news['Sentiment'].value_counts()
            fig_pie = go.Figure(data=[go.Pie(labels=counts.index, values=counts.values, hole=.4)])
            fig_pie.update_traces(marker=dict(colors=['#00CC96', '#EF553B', '#AB63FA']))
            fig_pie.update_layout(height=350, margin=dict(l=0,r=0,t=20,b=0))
            st.plotly_chart(fig_pie, use_container_width=True)

    # 5. DEEP NEWS FEED
    st.subheader("📰 Deep Article Analysis")
    if not df_news.empty:
        for index, row in df_news.iterrows():
            with st.expander(f"{row['Icon']} {row['title']}"):
                st.caption(f"**Source:** {row['source']} | **Date:** {row['date']}")
                
                # Show the Logic
                st.info(f"💡 **AI Logic:** {row['Explanation']}")
                
                # Show the preview text (Scraped or Snippet)