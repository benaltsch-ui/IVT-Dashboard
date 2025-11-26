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
    
    # Financial Dictionary (Teaching the AI market terms)
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
    
    if unique_drivers:
        explanation = f"Based on {method}. Key drivers: " + ", ".join([f"**{w}**" for w in unique_drivers])
    else:
        explanation = f"Based on {method}."
        
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
    Uses DuckDuckGo to get DIRECT links.
    """
    # DDGS allows us to search for news
    try:
        results = DDGS().news(keywords=f"{query} South Africa", region="za-en", safesearch="off", max_results=5)
    except Exception as e:
        st.error(f"Search Error: {e}")
        return []
    
    news_items = []
    
    # Progress Bar
    progress_text = "Fetching direct articles..."
    my_bar = st.progress(0, text=progress_text)
    
    if not results:
        return []

    for i, result in enumerate(results):
        my_bar.progress((i + 1) / len(results), text=f"Reading: {result.get('title', 'Article')[:20]}...")
        
        # DuckDuckGo gives us a 'url' key which is the REAL link.
        direct_url = result.get('url', '')
        
        # 1. Try to Scrape Full Text
        full_text = None
        if direct_url:
            full_text = get_article_content(direct_url)
        
        if full_text:
            score, label, icon, expl = analyze_content(full_text, "Full Article Text")
            snippet = full_text[:300] + "..."
            method = "✅ Full Text Read"
        else:
            # Fallback to the snippet
            fallback_text = f"{result.get('title', '')}. {result.get('body', '')}"
            score, label, icon, expl = analyze_content(fallback_text, "Search Snippet")
            snippet = result.get('body', 'No preview available')
            method = "⚠️ Snippet Only (Blocked)"

        # --- THIS WAS THE BLOCK CAUSING ERRORS BEFORE ---
        news_items.append({
            "title": result.get('title', 'Unknown Title'),
            "link": direct_url,
            "source": result.get('source', 'News'),
            "date": result.get('date', ''),
            "snippet": snippet,
            "Sentiment": label,
            "Icon": icon,
            "Score": score,
            "Explanation": expl,
            "Method": method
        })
        # ------------------------------------------------
        
    my_bar.empty()
    return news_items

# --- MAIN APP ---
def main():
    st.title("🏭 Invicta Holdings (IVT) | Deep Sentiment")
    
    # 1. Market Data
    history, info = get_market_data("IVT.JO")
    
    if not history.empty:
        # Calculate Metrics
        curr = history['Close'].iloc[-1]
        prev = history['Close'].iloc[-2]
        pct = ((curr - prev) / prev) * 100
        
        # Top KPI Row
        c1, c2, c3 = st.columns(3)
        c1.metric("Current Price", f"R {curr:.2f}", f"{pct:.2f}%")
        c2.metric("PE Ratio", f"{info.get('trailingPE', 'N/A')}")
        c3.metric("Market Cap", f"R {info.get('marketCap', 0)/1e9:.2f} B")
        
        # --- IMPROVED CHARTING SECTION ---
        st.subheader("Price Performance & Volume")
        
        # Create a chart with 2 rows (Price on top, Volume on bottom)
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                            vertical_spacing=0.05, 
                            row_heights=[0.7, 0.3]) # Price gets 70% space, Volume 30%

        # Row 1: Candlestick Chart (Price)
        fig.add_trace(go.Candlestick(
            x=history.index,
            open=history['Open'], high=history['High'],
            low=history['Low'], close=history['Close'],
            name="Price"
        ), row=1, col=1)

        # Row 2: Bar Chart (Volume)
        # We color volume bars based on price movement (Green if up, Red if down)
        colors = ['red' if row['Open'] - row['Close'] > 0 else 'green' for index, row in history.iterrows()]
        fig.add_trace(go.Bar(
            x=history.index, 
            y=history['Volume'],
            marker_color=colors,
            name="Volume"
        ), row=2, col=1)

        # --- LABELS AND STYLING ---
        fig.update_layout(
            height=600, # Taller chart to fit labels
            margin=dict(l=20, r=20, t=30, b=20),
            xaxis_rangeslider_visible=False, # Hide the bottom slider to save space
            showlegend=False,
            hovermode='x unified' # Shows all data when you hover over a date
        )

        # Y-Axis Labels
        fig.update_yaxes(title_text="<b>Price (ZAR)</b>", tickprefix="R", row=1, col=1)
        fig.update_yaxes(title_text="<b>Volume</b>", row=2, col=1)
        
        # X-Axis Label
        fig.update_xaxes(title_text="<b>Date</b>", row=2, col=1)
        
        # Render
        st.plotly_chart(fig, use_container_width=True)
        # ---------------------------------
    
    # 2. News Data
    st.divider()
    st.subheader("📰 Deep News Reader")
    st.caption("Powered by DuckDuckGo Direct Links & Newspaper3k")
    
    news = get_live_news_duckduckgo("Invicta Holdings Limited")
    
    if not news:
        st.warning("No direct news found. Checking Industrial Sector...")
        news = get_live_news_duckduckgo("JSE Industrial Engineering")
        
    if news:
        df = pd.DataFrame(news)
        avg = df['Score'].mean()
        st.metric("Sentiment Score", f"{avg:.2f}", delta="Bullish" if avg > 0.05 else "Bearish" if avg < -0.05 else "Neutral")
        
        for i, row in df.iterrows():
            with st.expander(f"{row['Icon']} {row['title']}"):
                if "Full Text" in row['Method']:
                    st.success(row['Method'])
                else:
                    st.warning(row['Method'])
                
                st.info(f"💡 {row['Explanation']}")
                st.markdown(f"**Preview:** {row['snippet']}")
                st.markdown(f"[Read Source]({row['link']})")
