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
st.set_page_config(page_title="Invicta Holdings Live Dashboard", layout="wide", page_icon="🏭")

# --- CACHING & RESOURCES ---
@st.cache_resource
def load_sentiment_resources():
    try:
        nltk.data.find('sentiment/vader_lexicon.zip')
    except LookupError:
        nltk.download('vader_lexicon', quiet=True)
    
    sia = SentimentIntensityAnalyzer()
    
    # Financial Dictionary (Teaching the AI market slang)
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

# --- HELPER: RESOLVE GOOGLE LINKS ---
def get_final_url(url):
    """
    Follows the Google Redirect to get the actual clean URL.
    This ensures links open properly when clicked.
    """
    try:
        session = requests.Session()
        session.headers.update({"User-Agent": "Mozilla/5.0"})
        # We allow redirects to find the real destination
        response = session.head(url, allow_redirects=True, timeout=5)
        return response.url
    except:
        return url

# --- HELPER: STABLE SCRAPER ---
def get_article_content(url):
    """
    Uses Trafilatura (More stable than Newspaper3k).
    """
    try:
        # Download
        downloaded = trafilatura.fetch_url(url)
        if downloaded:
            # Extract
            text = trafilatura.extract(downloaded)
            if text and len(text) > 200:
                return text
    except:
        pass
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
    # Google News RSS
    encoded = query.replace(" ", "%20")
    rss_url = f"https://news.google.com/rss/search?q={encoded}+South+Africa&hl=en-ZA&gl=ZA&ceid=ZA:en"
    
    feed = feedparser.parse(rss_url)
    if not feed.entries: return []
    
    news_items = []
    limit = min(len(feed.entries), 8) # Limit to 8 for speed
    
    progress_text = "Analyzing news stream..."
    my_bar = st.progress(0, text=progress_text)
    
    for i, entry in enumerate(feed.entries[:limit]):
        my_bar.progress((i + 1) / limit, text=f"Processing: {entry.title[:20]}...")
        
        # 1. Date Formatting
        try:
            dt = parsedate_to_datetime(entry.published)
            clean_date = dt.strftime("%d %b %Y") # 26 Nov 2025
        except:
            clean_date = "Recent"

        # 2. Link Resolution (Fixing broken links)
        real_url = get_final_url(entry.link)
        
        # 3. Scraping
        full_text = get_article_content(real_url)
        
        if full_text:
            score, label, icon, expl = analyze_content(full_text, "Full Article Text")
            snippet = full_text[:300] + "..."
            method = "✅ Full Text"
        else:
            # Fallback
            raw_desc = entry.get('description', entry.title)
            import re
            clean_desc = re.sub('<.*?>', '', raw_desc)
            score, label, icon, expl = analyze_content(clean_desc, "Headline/Snippet")
            snippet = clean_desc
            method = "⚠️ Snippet"

        news_items.append({
            "title": entry.title,
            "link": real_url,
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
        
        # --- CHARTS SECTION (IMPROVED) ---
        st.subheader("Price Performance & Trading Activity")
        
        # Calculate 30-Day Moving Average for Volume
        history['Vol_Avg'] = history['Volume'].rolling(window=30).mean()

        fig = make_subplots(
            rows=2, cols=1, 
            shared_xaxes=True, 
            vertical_spacing=0.1, 
            row_heights=[0.7, 0.3]
        )

        # Row 1: Price
        fig.add_trace(go.Candlestick(
            x=history.index, open=history['Open'], high=history['High'],
            low=history['Low'], close=history['Close'], name="Price"
        ), row=1, col=1)

        # Row 2: Volume Bars (Green=Up Day, Red=Down Day)
        colors = ['#EA4335' if row['Open'] - row['Close'] > 0 else '#34A853' for index, row in history.iterrows()]
        fig.add_trace(go.Bar(
            x=history.index, y=history['Volume'],
            marker_color=colors, name="Shares Traded",
            hovertemplate="<b>Date:</b> %{x}<br><b>Shares:</b> %{y:,.0f}<extra></extra>"
        ), row=2, col=1)

        # Row 2: Volume Average Line
        fig.add_trace(go.Scatter(
            x=history.index, y=history['Vol_Avg'],
            mode='lines', name="30-Day Avg Volume",
            line=dict(color='orange', width=2)
        ), row=2, col=1)

        fig.update_layout(height=650, margin=dict(l=20, r=20, t=20, b=20), xaxis_rangeslider_visible=False, showlegend=True, legend=dict(orientation="h", y=1.02, x=0), hovermode='x unified')
        fig.update_yaxes(title_text="<b>Price (ZAR)</b>", tickprefix="R", row=1, col=1)
        fig.update_yaxes(title_text="<b>No. Shares Traded</b>", row=2, col=1)
        fig.update_xaxes(title_text="<b>Date</b>", row=2, col=1)
        
        st.plotly_chart(fig, use_container_width=True)

        # Explanation Box
        st.info("""
        **📊 How to read the Volume Chart (Bottom Graph):**
        *   **Bars:** Represent the total number of shares traded that day.
        *   **Orange Line:** The 30-day average. Bars spiking above this line indicate **high conviction** events (e.g., Earnings, M&A News).
        *   **Color:** 🟢 **Green** = Buyers were dominant (Price closed higher) | 🔴 **Red** = Sellers were dominant (Price closed lower).
        """)
        
    # 2. NEWS SECTION
    st.divider()
    st.subheader("📰 Deep News Reader")
    st.caption("Latest Articles | Verified Links & Dates")
    
    # Primary Search
    news = get_google_news("Invicta Holdings Limited")
    # Secondary Search if empty
    if not news:
        st.warning("No direct news found. Checking Sector...")
        news = get_google_news("JSE Industrial Engineering")
        
    if news:
        df = pd.DataFrame(news)
        avg = df['Score'].mean()
        st.metric("Sentiment Score", f"{avg:.2f}", delta="Bullish" if avg > 0.05 else "Bearish" if avg < -0.05 else "Neutral")
        
        for i, row in df.iterrows():
            with st.expander(f"{row['Icon']} {row['title']}"):
                m1, m2 = st.columns([3, 1])
                m1.caption(f"**Source:** {row['source']} | **Published:** {row['date']}")
                m2.caption(f"**{row['Method']}**")
                
                st.info(f"💡 {row['Explanation']}")
                st.markdown(f"**Preview:** {row['snippet']}")
                st.markdown(f"🔗 [**Click to Read Full Article**]({row['link']})")
    else:
        st.write("No news found.")

if __name__ == "__main__":
    main()
