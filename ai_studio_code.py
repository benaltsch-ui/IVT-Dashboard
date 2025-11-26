import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import feedparser
import datetime

# --- CONFIGURATION ---
st.set_page_config(page_title="Invicta Holdings Live Dashboard", layout="wide", page_icon="🏭")

# --- CACHING (SPEED & PERFORMANCE) ---
@st.cache_resource
def load_sentiment_model():
    nltk.download('vader_lexicon', quiet=True)
    return SentimentIntensityAnalyzer()

sia = load_sentiment_model()

@st.cache_data(ttl=900) # Cache data for 15 minutes to prevent blocking
def get_market_data(ticker):
    stock = yf.Ticker(ticker)
    # Get 1 year history
    history = stock.history(period="1y")
    # Get fundamentals (handle potential missing keys safely)
    try:
        info = stock.info
    except:
        info = {}
    return history, info

@st.cache_data(ttl=3600) # Cache news for 1 hour
def get_live_news(query):
    # Google News RSS Feed (Free & Reliable)
    encoded_query = query.replace(" ", "%20")
    rss_url = f"https://news.google.com/rss/search?q={encoded_query}+South+Africa&hl=en-ZA&gl=ZA&ceid=ZA:en"
    
    feed = feedparser.parse(rss_url)
    
    news_items = []
    # If no news found, provide a fallback message
    if not feed.entries:
        return []

    for entry in feed.entries[:10]: # Limit to top 10 recent
        news_items.append({
            "date": entry.published[:16], # Truncate time for cleaner look
            "title": entry.title,
            "link": entry.link,
            "source": entry.source.title if hasattr(entry, 'source') else "Google News"
        })
    return news_items

def analyze_sentiment(text):
    score = sia.polarity_scores(text)['compound']
    if score >= 0.05: return "Positive", "🟢", score
    elif score <= -0.05: return "Negative", "🔴", score
    else: return "Neutral", "⚪", score

# --- MAIN APP LAYOUT ---
def main():
    st.title("🏭 Invicta Holdings (IVT) | Live Sentiment Monitor")
    st.markdown(f"*Data refreshed: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')} SAST*")
    st.divider()

    # Sidebar
    st.sidebar.header("Configuration")
    ticker = st.sidebar.text_input("JSE Ticker", "IVT.JO")
    
    # 1. FETCH DATA
    with st.spinner('Connecting to JSE & Google News...'):
        history, info = get_market_data(ticker)
        
        # We search for "Invicta Holdings" AND "JSE" to filter noise
        news_data = get_live_news("Invicta Holdings Limited")
        
        # Fallback if no news found specifically for Invicta (Common for mid-cap stocks)
        if not news_data:
             st.warning("No immediate news found for Invicta. Showing sector peers (Industrial Engineering).")
             news_data = get_live_news("JSE Industrial Engineering sector")

    # 2. PROCESS SENTIMENT
    if news_data:
        df_news = pd.DataFrame(news_data)
        df_news[['Sentiment', 'Icon', 'Score']] = df_news['title'].apply(
            lambda x: pd.Series(analyze_sentiment(x))
        )
        avg_sentiment = df_news['Score'].mean()
    else:
        df_news = pd.DataFrame(columns=['date', 'title', 'source', 'Sentiment', 'Icon', 'Score'])
        avg_sentiment = 0

    # 3. KPI DISPLAY
    current_price = history['Close'].iloc[-1]
    prev_price = history['Close'].iloc[-2]
    delta = ((current_price - prev_price) / prev_price) * 100
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Share Price (ZAR)", f"{current_price:.2f}", f"{delta:.2f}%")
    col2.metric("Market Cap", f"R {info.get('marketCap', 0)/1e9:.2f} B") # Convert to Billions
    col3.metric("PE Ratio", f"{info.get('trailingPE', 'N/A')}")
    
    # Sentiment KPI
    sent_label = "Bullish" if avg_sentiment > 0.05 else "Bearish" if avg_sentiment < -0.05 else "Neutral"
    col4.metric("Media Sentiment", sent_label, f"{avg_sentiment:.2f} Score")

    # 4. CHARTS
    c1, c2 = st.columns([2, 1])
    
    with c1:
        st.subheader("📈 Price History (1 Year)")
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
            fig_pie.update_colors(marker=dict(colors=['#00CC96', '#EF553B', '#AB63FA']))
            fig_pie.update_layout(height=350, margin=dict(l=0,r=0,t=20,b=0), showlegend=True)
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.info("No news data available to chart.")

    # 5. NEWS FEED
    st.subheader("📰 Live News & Sentiment Analysis")
    if not df_news.empty:
        for index, row in df_news.iterrows():
            with st.expander(f"{row['Icon']} {row['title']}"):
                st.write(f"**Source:** {row['source']} | **Date:** {row['date']}")
                st.write(f"**Sentiment Score:** {row['Score']:.2f} ({row['Sentiment']})")
                st.markdown(f"[Read Article]({row['link']})")
    else:
        st.write("No recent news articles found.")

if __name__ == "__main__":
    main()