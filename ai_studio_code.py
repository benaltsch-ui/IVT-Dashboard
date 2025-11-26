import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import feedparser
import datetime
import trafilatura # <--- Specialized News Scraper

# --- CONFIGURATION ---
st.set_page_config(page_title="Invicta Holdings Live Dashboard", layout="wide", page_icon="🏭")

# --- CACHING & MODEL LOADING ---
@st.cache_resource
def load_sentiment_resources():
    nltk.download('vader_lexicon', quiet=True)
    sia = SentimentIntensityAnalyzer()
    
    # --- TEACH THE AI FINANCIAL CONTEXT ---
    # We update the dictionary so it understands stock market slang
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

# --- ADVANCED SCRAPER ---
def get_full_article_text(url):
    """
    Uses Trafilatura to fetch article content. 
    It is much better at bypassing bot detection than standard requests.
    """
    try:
        # Download the HTML
        downloaded = trafilatura.fetch_url(url)
        
        if downloaded:
            # Extract only the main article text (removes ads, menus, popups)
            text = trafilatura.extract(downloaded)
            if text and len(text) > 200:
                return text
    except:
        pass
    return None

# --- SENTIMENT LOGIC ---
def analyze_content(text, method="Snippet"):
    # 1. Calculate Score
    score = sia.polarity_scores(text)['compound']
    
    # 2. Assign Label
    if score >= 0.05: label, icon = "Positive", "🟢"
    elif score <= -0.05: label, icon = "Negative", "🔴"
    else: label, icon = "Neutral", "⚪"
    
    # 3. Extract Drivers (Why did it get this score?)
    words = text.lower().split()
    drivers = []
    
    for word in words:
        clean_word = word.strip('.,!?"\'')
        # Check if the word is in our dictionary
        if clean_word in sia.lexicon:
            val = sia.lexicon[clean_word]
            # Only pick "Strong" words to show the user
            if abs(val) >= 1.0: 
                drivers.append((clean_word, val))
    
    # Sort by impact (High sentiment words first)
    drivers.sort(key=lambda x: abs(x[1]), reverse=True)
    
    # Get top 5 unique keywords
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
    # Google News RSS Search
    encoded = query.replace(" ", "%20")
    rss_url = f"https://news.google.com/rss/search?q={encoded}+South+Africa&hl=en-ZA&gl=ZA&ceid=ZA:en"
    
    feed = feedparser.parse(rss_url)
    
    if not feed.entries: return []
    
    news_items = []
    
    # Progress Bar Setup
    progress_text = "Reading articles... Please wait."
    my_bar = st.progress(0, text=progress_text)
    total_articles = min(len(feed.entries), 6) # Limit to 6 for speed
    
    for i, entry in enumerate(feed.entries[:6]):
        # Update progress
        my_bar.progress((i + 1) / total_articles, text=f"Reading article {i+1} of {total_articles}")
        
        # 1. Attempt to Read Full Article
        full_text = get_full_article_text(entry.link)
        
        if full_text:
            # SUCCESS: We have the body text
            score, label, icon, expl = analyze_content(full_text, "Full Text")
            snippet = full_text[:300] + "..." # Longer preview
            source_method = "Full Article Read"
        else:
            # FAIL: Fallback to RSS description
            raw_summary = entry.get('description', '') or entry.title
            # Clean HTML from summary
            import re
            clean_summary = re.sub('<.*?>', '', raw_summary)
            score, label, icon, expl = analyze_content(clean_summary, "Headline/Snippet")
            snippet = clean_summary
            source_method = "Snippet (Site Blocked)"

        news_items.append({
            "title": entry.title,
            "link": entry.link,
            "source": entry.source.title if hasattr(entry, 'source') else "News",
            "date": entry.published[:16],
            "snippet": snippet,
            "Sentiment": label,
            "Icon": icon,
            "Score": score,
            "Explanation": expl,
            "Method": source_method
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
        pct = ((curr - history['Close'].iloc[-2]) / history['Close'].iloc[-2]) * 100
        c1, c2, c3 = st.columns(3)
        c1.metric("Price (ZAR)", f"{curr:.2f}", f"{pct:.2f}%")
        c2.metric("PE Ratio", f"{info.get('trailingPE', 'N/A')}")
        
        fig = go.Figure(data=[go.Candlestick(x=history.index, open=history['Open'], high=history['High'], low=history['Low'], close=history['Close'])])
        fig.update_layout(height=300, margin=dict(l=0,r=0,t=10,b=0))
        st.plotly_chart(fig, use_container_width=True)
    
    # 2. NEWS
    st.divider()
    st.subheader("📰 Deep News Reader")
    st.info("The system now specifically scans for financial terms (e.g., 'shoot up' = Positive).")
    
    news = get_live_news_deep("Invicta Holdings Limited")
    if not news:
        st.warning("No direct news found. Scanning sector peers...")
        news = get_live_news_deep("JSE Industrial Engineering")
    
    if news:
        df = pd.DataFrame(news)
        avg = df['Score'].mean()
        
        st.metric("Aggregate Sentiment Score", f"{avg:.2f}", delta="Bullish" if avg > 0.05 else "Bearish" if avg < -0.05 else "Neutral")
        
        for i, row in df.iterrows():
            with st.expander(f"{row['Icon']} {row['title']}"):
                
                # Dynamic Badge based on scrape success
                if "Full Article" in row['Method']:
                    st.success(f"✅ AI successfully read the full article.")
                else:
                    st.warning(f"⚠️ Access Denied by Publisher. AI fell back to analyzing the summary.")
                
                st.info(f"💡 {row['Explanation']}")
                st.markdown(f"_{row['snippet']}_")
                st.markdown(f"[Read Source]({row['link']})")

if __name__ == "__main__":
    main()
