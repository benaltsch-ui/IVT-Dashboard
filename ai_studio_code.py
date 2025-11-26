import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import feedparser
import datetime
import requests
import re

# --- CONFIGURATION ---
st.set_page_config(page_title="Invicta Holdings Live Dashboard", layout="wide", page_icon="🏭")

# --- CACHING & MODEL LOADING ---
@st.cache_resource
def load_sentiment_resources():
    nltk.download('vader_lexicon', quiet=True)
    sia = SentimentIntensityAnalyzer()
    
    # Financial Dictionary Update (Teach AI that 'shoot up' is good)
    financial_lexicon = {
        'shoot': 2.0, 'surged': 3.0, 'jumped': 2.5, 'climbed': 2.0, 'soared': 3.0, 'green': 1.5,
        'plunged': -3.0, 'tumbled': -2.5, 'slumped': -2.5, 'red': -1.5,
        'dividend': 2.0, 'earnings': 1.5, 'profit': 2.0, 'growth': 2.0,
        'resilient': 2.0, 'strong': 2.0, 'up': 1.0, 'down': -1.0,
        'challenging': -1.0, 'headwinds': -1.5, 'uncertainty': -1.0
    }
    sia.lexicon.update(financial_lexicon)
    return sia

sia = load_sentiment_resources()

# --- NEW SCRAPER (JINA AI PROXY) ---
def get_full_article_text(url):
    """
    Prefixes URL with https://r.jina.ai/ to use their reader engine.
    This bypasses many standard bot blocks.
    """
    try:
        api_url = f"https://r.jina.ai/{url}"
        headers = {"User-Agent": "Mozilla/5.0"}
        
        # Timeout increased to 10s because Jina renders the page
        response = requests.get(api_url, headers=headers, timeout=10)
        
        if response.status_code == 200:
            text = response.text
            # Basic cleaning of the Markdown response
            if "Just a moment" in text or "Access denied" in text:
                return None
            if len(text) > 400: # Ensure we got a real article
                return text
    except:
        pass
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
        clean_word = word.strip('.,!?"\'*#') # Strip markdown chars too
        if clean_word in sia.lexicon:
            val = sia.lexicon[clean_word]
            if abs(val) >= 1.0: 
                drivers.append((clean_word, val))
    
    drivers.sort(key=lambda x: abs(x[1]), reverse=True)
    unique_drivers = list(set([x[0] for x in drivers]))[:5]
    
    if unique_drivers:
        explanation = f"Based on {method}. Key drivers: " + ", ".join([f"**{w}**" for w in unique_drivers])
    else:
        explanation = f"Based on {method}. (No strong keywords detected)"
        
    return score, label, icon, explanation

@st.cache_data(ttl=900)
def get_market_data(ticker):
    stock = yf.Ticker(ticker)
    try:
        return stock.history(period="1y"), stock.info
    except:
        return pd.DataFrame(), {}

@st.cache_data(ttl=3600)
def get_live_news_deep(query):
    encoded = query.replace(" ", "%20")
    rss_url = f"https://news.google.com/rss/search?q={encoded}+South+Africa&hl=en-ZA&gl=ZA&ceid=ZA:en"
    
    feed = feedparser.parse(rss_url)
    if not feed.entries: return []
    
    news_items = []
    
    # Limit to 5 articles because Jina takes a moment to process
    progress_text = "AI is reading full articles..."
    my_bar = st.progress(0, text=progress_text)
    total = min(len(feed.entries), 5) 
    
    for i, entry in enumerate(feed.entries[:5]):
        my_bar.progress((i + 1) / total, text=f"Analyzing: {entry.title[:30]}...")
        
        # 1. Try Jina Proxy
        full_text = get_full_article_text(entry.link)
        
        if full_text:
            score, label, icon, expl = analyze_content(full_text, "Full Text")
            # Create a clean preview (remove markdown links)
            clean_preview = re.sub(r'\[.*?\]\(.*?\)', '', full_text)[:300].replace('\n', ' ') + "..."
            method_badge = "✅ Full Article (Jina AI)"
        else:
            # Fallback
            raw_summary = entry.get('description', '') or entry.title
            clean_summary = re.sub('<.*?>', '', raw_summary)
            score, label, icon, expl = analyze_content(clean_summary, "Headline Only")
            clean_preview = clean_summary
            method_badge = "⚠️ Summary Only (Blocked)"

        news_items.append({
            "title": entry.title,
            "link": entry.link,
            "source": entry.source.title if hasattr(entry, 'source') else "News",
            "date": entry.published[:16],
            "snippet": clean_preview,
            "Sentiment": label,
            "Icon": icon,
            "Score": score,
            "Explanation": expl,
            "Method": method_badge
        })
        
    my_bar.empty()
    return news_items

# --- MAIN APP ---
def main():
    st.title("🏭 Invicta Holdings (IVT) | Deep Sentiment")
    
    # 1. Market Data
    history, info = get_market_data("IVT.JO")
    if not history.empty:
        curr = history['Close'].iloc[-1]
        pct = ((curr - history['Close'].iloc[-2]) / history['Close'].iloc[-2]) * 100
        c1, c2, c3 = st.columns(3)
        c1.metric("Price (ZAR)", f"{curr:.2f}", f"{pct:.2f}%")
        c2.metric("PE Ratio", f"{info.get('trailingPE', 'N/A')}")
        
        fig = go.Figure(data=[go.Candlestick(x=history.index, open=history['Open'], high=history['High'], low=history['Low'], close=history['Close'])])
        fig.update_layout(height=300, margin=dict(l=0,r=0,t=10,b=0))
        st.plotly_chart(fig, use_container_width=True)
    
    # 2. News Data
    st.divider()
    st.subheader("📰 Deep News Reader")
    st.caption("Using Jina AI to bypass basic paywalls and read full context.")
    
    news = get_live_news_deep("Invicta Holdings Limited")
    if not news:
        news = get_live_news_deep("JSE Industrial Engineering")
        
    if news:
        df = pd.DataFrame(news)
        avg = df['Score'].mean()
        st.metric("Sentiment Score", f"{avg:.2f}", delta="Bullish" if avg > 0.05 else "Bearish" if avg < -0.05 else "Neutral")
        
        for i, row in df.iterrows():
            with st.expander(f"{row['Icon']} {row['title']}"):
                if "Full Article" in row['Method']:
                    st.success(row['Method'])
                else:
                    st.warning(row['Method'])
                
                st.info(f"💡 {row['Explanation']}")
                st.markdown(f"**Preview:** {row['snippet']}")
                st.markdown(f"[Read Source]({row['link']})")

if __name__ == "__main__":
    main()
