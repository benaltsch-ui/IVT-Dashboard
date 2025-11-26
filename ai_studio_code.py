import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import feedparser
from newspaper import Article
import requests
import datetime
from email.utils import parsedate_to_datetime

# --- CONFIGURATION ---
st.set_page_config(page_title="Invicta Holdings Live Dashboard", layout="wide", page_icon="🏭")

# --- CACHING & RESOURCES ---
@st.cache_resource
def load_sentiment_resources():
    try:
        nltk.data.find('sentiment/vader_lexicon.zip')
        nltk.data.find('tokenizers/punkt.zip')
    except LookupError:
        nltk.download('vader_lexicon', quiet=True)
        nltk.download('punkt', quiet=True)
    
    sia = SentimentIntensityAnalyzer()
    
    # Financial Lexicon
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
except:
    st.error("Error loading AI models.")
    st.stop()

# --- HELPER: RESOLVE GOOGLE LINKS ---
def get_final_url(url):
    """
    Follows the Google Redirect to get the actual clean URL (e.g., iol.co.za)
    This fixes broken links and allows the scraper to work better.
    """
    try:
        # We use a session with a browser header to follow the redirect
        session = requests.Session()
        session.headers.update({"User-Agent": "Mozilla/5.0"})
        response = session.head(url, allow_redirects=True, timeout=5)
        return response.url
    except:
        return url

# --- HELPER: ARTICLE SCRAPER ---
def get_article_content(url):
    try:
        article = Article(url)
        article.download()
        article.parse()
        if len(article.text) < 150:
            return None
        return article.text
    except:
        return None

# --- SENTIMENT LOGIC ---
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
            if abs(val) >= 1.0: 
                drivers.append((clean_word, val))
    
    drivers.sort(key=lambda x: abs(x[1]), reverse=True)
    unique_drivers = list(set([x[0] for x in drivers]))[:5]
    
    explanation = f"Based on {method}. Key drivers: " + ", ".join([f"**{w}**" for w in unique_drivers]) if unique_drivers else f"Based on {method}."
    return score, label, icon, explanation

@st.cache_data(ttl=900)
def get_market_data(ticker):
    try:
        stock = yf.Ticker(ticker)
        return stock.history(period="1y"), stock.info
    except:
        return pd.DataFrame(), {}

@st.cache_data(ttl=3600)
def get_google_news(query):
    # Google News RSS (Best for coverage and dates)
    encoded = query.replace(" ", "%20")
    rss_url = f"https://news.google.com/rss/search?q={encoded}+South+Africa&hl=en-ZA&gl=ZA&ceid=ZA:en"
    
    feed = feedparser.parse(rss_url)
    if not feed.entries: return []
    
    news_items = []
    # Increase limit to 10 articles
    limit = min(len(feed.entries), 10)
    
    progress_text = "Resolving links and analyzing..."
    my_bar = st.progress(0, text=progress_text)
    
    for i, entry in enumerate(feed.entries[:limit]):
        my_bar.progress((i + 1) / limit, text=f"Processing: {entry.title[:20]}...")
        
        # 1. Get Date
        try:
            # Parse the messy RSS date into a clean string
            dt = parsedate_to_datetime(entry.published)
            clean_date = dt.strftime("%a, %d %b %Y") # e.g. Wed, 26 Nov 2025
        except:
            clean_date = "Unknown Date"

        # 2. Resolve URL (Fix broken links)
        real_url = get_final_url(entry.link)
        
        # 3. Scrape
        full_text = get_article_content(real_url)
        
        if full_text:
            score, label, icon, expl = analyze_content(full_text, "Full Article Text")
            snippet = full_text[:300] + "..."
            method = "✅ Full Text"
        else:
            # Fallback to description from RSS
            raw_desc = entry.get('description', entry.title)
            # Clean HTML tags
            import re
            clean_desc = re.sub('<.*?>', '', raw_desc)
            score, label, icon, expl = analyze_content(clean_desc, "Headline/Snippet")
            snippet = clean_desc
            method = "⚠️ Snippet"

        news_items.append({
            "title": entry.title,
            "link": real_url, # The clean link
            "source": entry.source.title if hasattr(entry, 'source') else "News",
            "date": clean_date,
            "snippet": snippet,
            "Sentiment": label,
            "Icon": icon,
            "Score": score,
            "Explanation": expl,
            "Method": method
        })
        
    my_bar.empty()
    return news_items

# --- MAIN APP ---
def main():
    st.title("🏭 Invicta Holdings (IVT) | Deep Sentiment")
    
    # 1. MARKET DATA
    history, info = get_market_data("IVT.JO")
    
    if not history.empty:
        curr = history['Close'].iloc[-1]
        prev = history['Close'].iloc[-2]
        pct = ((curr - prev) / prev) * 100
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Price", f"R {curr:.2f}", f"{pct:.2f}%")
        c2.metric("PE Ratio", f"{info.get('trailingPE', 'N/A')}")
        c3.metric("Market Cap", f"R {info.get('marketCap', 0)/1e9:.2f} B")
        
        # Charts
        st.subheader("Price Performance & Volume")
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.7, 0.3])
        fig.add_trace(go.Candlestick(x=history.index, open=history['Open'], high=history['High'], low=history['Low'], close=history['Close'], name="Price"), row=1, col=1)
        colors = ['red' if row['Open'] - row['Close'] > 0 else 'green' for index, row in history.iterrows()]
        fig.add_trace(go.Bar(x=history.index, y=history['Volume'], marker_color=colors, name="Volume"), row=2, col=1)
        fig.update_layout(height=500, margin=dict(l=20, r=20, t=20, b=20), xaxis_rangeslider_visible=False, showlegend=False, hovermode='x unified')
        fig.update_yaxes(title_text="<b>Price (ZAR)</b>", tickprefix="R", row=1, col=1)
        fig.update_yaxes(title_text="<b>Volume</b>", row=2, col=1)
        fig.update_xaxes(title_text="<b>Date</b>", row=2, col=1)
        st.plotly_chart(fig, use_container_width=True)
    
    # 2. NEWS
    st.divider()
    st.subheader("📰 Deep News Reader")
    st.caption("Latest 10 Articles | Dates & Links verified")
    
    news = get_google_news("Invicta Holdings Limited")
    if not news:
        st.warning("No direct news found. Checking Sector...")
        news = get_google_news("JSE Industrial Engineering")
        
    if news:
        df = pd.DataFrame(news)
        avg = df['Score'].mean()
        st.metric("Sentiment Score", f"{avg:.2f}", delta="Bullish" if avg > 0.05 else "Bearish" if avg < -0.05 else "Neutral")
        
        for i, row in df.iterrows():
            with st.expander(f"{row['Icon']} {row['title']}"):
                # Top Row: Source and Date
                m1, m2 = st.columns([3, 1])
                m1.caption(f"**Source:** {row['source']} | **Published:** {row['date']}")
                m2.caption(f"**{row['Method']}**")
                
                # Content
                st.info(f"💡 {row['Explanation']}")
                st.markdown(f"**Preview:** {row['snippet']}")
                st.markdown(f"🔗 [**Click to Read Full Article**]({row['link']})")
    else:
        st.write("No news found.")

if __name__ == "__main__":
    main()
