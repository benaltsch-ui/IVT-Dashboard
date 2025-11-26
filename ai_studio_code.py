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

# --- UNIVERSAL NEWS FETCHER ---
@st.cache_data(ttl=3600)
def fetch_news_score(query, article_limit=5):
    """
    Fetches news for ANY company and returns the average sentiment score + articles.
    """
    encoded = query.replace(" ", "%20")
    rss_url = f"https://news.google.com/rss/search?q={encoded}+South+Africa&hl=en-ZA&gl=ZA&ceid=ZA:en"
    feed = feedparser.parse(rss_url)
    
    if not feed.entries:
        return 0.0, [] # Neutral score, no articles
    
    news_items = []
    limit = min(len(feed.entries), article_limit)
    
    for entry in feed.entries[:limit]:
        real_url = get_final_url(entry.link)
        full_text = get_article_content(real_url)
        
        if full_text:
            score, label, icon, expl = analyze_content(full_text, "Full Text")
            snippet = full_text[:300] + "..."
            method = "✅ Full Text"
        else:
            raw_desc = entry.get('description', entry.title)
            import re
            clean_desc = re.sub('<.*?>', '', raw_desc)
            score, label, icon, expl = analyze_content(clean_desc, "Snippet")
            snippet = clean_desc
            method = "⚠️ Snippet"
            
        # Clean Date
        try:
            dt = parsedate_to_datetime(entry.published)
            clean_date = dt.strftime("%d %b %Y")
        except:
            clean_date = "Recent"

        news_items.append({
            "title": entry.title,
            "link": real_url,
            "source": entry.source.title if hasattr(entry, 'source') else "News",
            "date": clean_date,
            "snippet": snippet,
            "Sentiment": label,
            "Icon": icon,
            "Score": score, # The specific impact of this article
            "Explanation": expl,
            "Method": method
        })
        
    # Calculate Average
    if news_items:
        df = pd.DataFrame(news_items)
        avg_score = df['Score'].mean()
    else:
        avg_score = 0.0
        
    return avg_score, news_items

@st.cache_data(ttl=900)
def get_market_data(ticker):
    try:
        stock = yf.Ticker(ticker)
        return stock.history(period="1y"), stock.info
    except:
        return pd.DataFrame(), {}

# --- MAIN APP ---
def main():
    st.title("🏭 Invicta Holdings (IVT) | Deep Sentiment")
    
    # 1. MARKET DATA
    history, info = get_market_data("IVT.JO")
    
    if not history.empty:
        curr = history['Close'].iloc[-1]
        pct = ((curr - history['Close'].iloc[-2]) / history['Close'].iloc[-2]) * 100
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Price", f"R {curr:.2f}", f"{pct:.2f}%")
        c2.metric("PE Ratio", f"{info.get('trailingPE', 'N/A')}")
        c3.metric("Market Cap", f"R {info.get('marketCap', 0)/1e9:.2f} B")
        
        # Charts
        st.subheader("Price Performance & Trading Volume")
        history['Vol_Avg'] = history['Volume'].rolling(window=30).mean()
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1, row_heights=[0.7, 0.3])
        fig.add_trace(go.Candlestick(x=history.index, open=history['Open'], high=history['High'], low=history['Low'], close=history['Close'], name="Price"), row=1, col=1)
        colors = ['#EA4335' if row['Open'] - row['Close'] > 0 else '#34A853' for index, row in history.iterrows()]
        fig.add_trace(go.Bar(x=history.index, y=history['Volume'], marker_color=colors, name="Shares Traded"), row=2, col=1)
        fig.add_trace(go.Scatter(x=history.index, y=history['Vol_Avg'], mode='lines', name="30-Day Avg Volume", line=dict(color='orange', width=2)), row=2, col=1)
        fig.update_layout(height=600, margin=dict(l=20, r=20, t=20, b=20), xaxis_rangeslider_visible=False, showlegend=True, legend=dict(orientation="h", y=1.02, x=0), hovermode='x unified')
        fig.update_yaxes(title_text="<b>Price (ZAR)</b>", tickprefix="R", row=1, col=1)
        fig.update_yaxes(title_text="<b>Shares Traded</b>", row=2, col=1)
        fig.update_xaxes(title_text="<b>Date</b>", row=2, col=1)
        st.plotly_chart(fig, use_container_width=True)
    
    # 2. INVICTA NEWS
    st.divider()
    st.subheader("📰 Invicta News & Impact Analysis")
    
    with st.spinner("Analyzing Invicta News..."):
        # We fetch up to 8 articles for Invicta
        ivt_score, ivt_news = fetch_news_score("Invicta Holdings Limited", article_limit=8)
    
    if ivt_news:
        st.metric("Invicta Total Sentiment", f"{ivt_score:.2f}", delta="Bullish" if ivt_score > 0 else "Bearish")
        
        for item in ivt_news:
            with st.expander(f"{item['Icon']} {item['title']}"):
                # Layout
                col_txt, col_imp = st.columns([3, 1])
                
                with col_txt:
                    st.caption(f"**Source:** {item['source']} | **Date:** {item['date']}")
                    st.info(f"💡 {item['Explanation']}")
                    st.markdown(f"**Preview:** {item['snippet']}")
                    st.markdown(f"[Read Article]({item['link']})")
                
                with col_imp:
                    st.markdown("##### Sentiment Impact")
                    # Visual Impact Bar
                    # Normalizing score (-1 to 1) to (0 to 100) for progress bar
                    norm_score = (item['Score'] + 1) / 2 
                    st.progress(norm_score)
                    
                    if item['Score'] > 0:
                        st.success(f"Positive Impact\n(+{item['Score']:.2f})")
                    elif item['Score'] < 0:
                        st.error(f"Negative Impact\n({item['Score']:.2f})")
                    else:
                        st.warning("Neutral Impact")
    else:
        st.write("No news found for Invicta.")

    # 3. COMPETITOR BATTLE
    st.divider()
    st.subheader("🏆 Sentiment Battle: IVT vs Competitors")
    st.caption("AI is reading news for competitors to compare Brand Reputation.")
    
    with st.spinner("Scouting Competitors (Hudaco & Barloworld)..."):
        # Fetch less articles (4) for competitors to keep app fast
        hdc_score, _ = fetch_news_score("Hudaco Industries", article_limit=4)
        baw_score, _ = fetch_news_score("Barloworld Limited", article_limit=4)
    
    # Create Comparison Data
    comp_data = {
        'Company': ['Invicta (IVT)', 'Hudaco (HDC)', 'Barloworld (BAW)'],
        'Sentiment Score': [ivt_score, hdc_score, baw_score],
        'Color': ['#1f77b4', '#d62728', '#2ca02c'] # Blue, Red, Green
    }
    
    df_comp = pd.DataFrame(comp_data)
    
    # 1. Bar Chart Comparison
    fig_comp = go.Figure(go.Bar(
        x=df_comp['Sentiment Score'],
        y=df_comp['Company'],
        orientation='h',
        marker_color=['#00CC96', '#636EFA', '#EF553B'], # Custom Colors
        text=df_comp['Sentiment Score'].apply(lambda x: f"{x:.2f}"),
        textposition='auto'
    ))
    
    fig_comp.update_layout(
        title="Real-time Media Sentiment Comparison",
        xaxis_title="Sentiment Score (-1.0 to +1.0)",
        height=300,
        margin=dict(l=0, r=0, t=30, b=0)
    )
    
    st.plotly_chart(fig_comp, use_container_width=True)
    
    # 2. Verdict
    winner = df_comp.loc[df_comp['Sentiment Score'].idxmax()]
    st.success(f"🏆 **Current Leader:** {winner['Company']} has the most positive media sentiment today ({winner['Sentiment Score']:.2f}).")

if __name__ == "__main__":
    main()
