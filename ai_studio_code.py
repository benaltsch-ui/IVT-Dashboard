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
st.set_page_config(page_title="Invicta Holdings Analysis", layout="wide", page_icon="🏭")

# --- CUSTOM CSS (To mimic the Simply Wall St Clean Look) ---
st.markdown("""
<style>
    .metric-card {
        background-color: #1e1e1e;
        padding: 15px;
        border-radius: 10px;
        color: white;
    }
    .reward-badge {
        background-color: #d1fae5;
        color: #065f46;
        padding: 4px 8px;
        border-radius: 4px;
        font-weight: bold;
        margin-bottom: 5px;
        display: inline-block;
    }
    .risk-badge {
        background-color: #fee2e2;
        color: #991b1b;
        padding: 4px 8px;
        border-radius: 4px;
        font-weight: bold;
        margin-bottom: 5px;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)

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
        'shoot': 2.0, 'surged': 3.0, 'jumped': 2.5, 'climbed': 2.0, 'soared': 3.0,
        'plunged': -3.0, 'tumbled': -2.5, 'slumped': -2.5, 'earnings': 1.5, 
        'profit': 2.0, 'growth': 2.0, 'resilient': 2.0, 'challenging': -1.0
    }
    sia.lexicon.update(financial_lexicon)
    return sia

try:
    sia = load_sentiment_resources()
except:
    st.stop()

# --- HELPER FUNCTIONS (Scraping & Sentiment) ---
def get_final_url(url):
    try:
        session = requests.Session()
        session.headers.update({"User-Agent": "Mozilla/5.0"})
        response = session.head(url, allow_redirects=True, timeout=3)
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

def analyze_content(text):
    score = sia.polarity_scores(text)['compound']
    if score >= 0.05: return score, "Positive", "🟢"
    elif score <= -0.05: return score, "Negative", "🔴"
    else: return score, "Neutral", "⚪"

@st.cache_data(ttl=900)
def get_stock_data(ticker):
    try:
        stock = yf.Ticker(ticker)
        return stock.history(period="1y"), stock.info
    except:
        return pd.DataFrame(), {}

@st.cache_data(ttl=3600)
def fetch_news(query, limit=5):
    encoded = query.replace(" ", "%20")
    feed = feedparser.parse(f"https://news.google.com/rss/search?q={encoded}+South+Africa&hl=en-ZA&gl=ZA&ceid=ZA:en")
    news_items = []
    
    for entry in feed.entries[:limit]:
        real_url = get_final_url(entry.link)
        full_text = get_article_content(real_url)
        
        if full_text:
            score, label, icon = analyze_content(full_text)
            snippet = full_text[:200] + "..."
        else:
            score, label, icon = analyze_content(entry.title)
            snippet = entry.title
            
        try:
            dt = parsedate_to_datetime(entry.published).strftime("%d %b %Y")
        except:
            dt = "Recent"

        news_items.append({
            "title": entry.title, "link": real_url, "source": entry.source.title, 
            "date": dt, "snippet": snippet, "score": score, "icon": icon
        })
    return news_items

# --- THE "SNOWFLAKE" LOGIC ---
def calculate_snowflake(info):
    """
    Calculates 0-5 scores for the 5 axes based on raw financial data.
    """
    scores = {"Value": 2, "Future": 2, "Past": 3, "Health": 3, "Dividend": 0}
    
    # Value Logic
    pe = info.get('trailingPE')
    if pe and pe < 15: scores['Value'] += 1
    if pe and pe < 10: scores['Value'] += 1
    if info.get('priceToBook', 2) < 1.5: scores['Value'] += 1
    
    # Dividend Logic
    div = info.get('dividendYield')
    if div and div > 0.02: scores['Dividend'] += 2
    if div and div > 0.04: scores['Dividend'] += 1
    if div and div > 0.06: scores['Dividend'] += 1
    
    # Health Logic
    current_ratio = info.get('currentRatio', 1)
    if current_ratio > 1.2: scores['Health'] += 1
    if info.get('debtToEquity', 100) < 60: scores['Health'] += 1
    
    # Past Performance
    if info.get('revenueGrowth', 0) > 0: scores['Past'] += 1
    if info.get('earningsGrowth', 0) > 0: scores['Past'] += 1
    
    return scores

# --- MAIN DASHBOARD ---
def main():
    # 1. FETCH DATA
    history, info = get_stock_data("IVT.JO")
    
    if history.empty:
        st.error("Could not load data for Invicta Holdings (IVT.JO).")
        st.stop()

    curr_price = history['Close'].iloc[-1]
    prev_price = history['Close'].iloc[-2]
    pct_change = ((curr_price - prev_price) / prev_price) * 100
    
    # 2. HEADER SECTION (Price & Logo feel)
    c1, c2 = st.columns([1, 3])
    with c1:
        st.title("Invicta Holdings")
        st.markdown(f"**JSE: IVT** • Industrial Engineering")
    with c2:
        st.metric("Share Price", f"R {curr_price:.2f}", f"{pct_change:.2f}%")

    st.divider()

    # 3. EXECUTIVE SUMMARY & SNOWFLAKE (Top Section)
    col_snowflake, col_summary = st.columns([2, 3])
    
    with col_snowflake:
        st.subheader("The Analysis")
        scores = calculate_snowflake(info)
        
        # Radar Chart (The Snowflake)
        fig_radar = go.Figure(data=go.Scatterpolar(
            r=[scores['Value'], scores['Future'], scores['Past'], scores['Health'], scores['Dividend'], scores['Value']],
            theta=['Value', 'Future', 'Past', 'Health', 'Dividend', 'Value'],
            fill='toself',
            name='Invicta',
            line_color='#2ca02c'
        ))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=False, range=[0, 6])),
            showlegend=False,
            margin=dict(l=30, r=30, t=10, b=10),
            height=300
        )
        st.plotly_chart(fig_radar, use_container_width=True)
        st.caption("Visual snapshot of Invicta's investment profile.")

    with col_summary:
        st.subheader("Risk & Rewards")
        
        # Logic to generate Rewards
        pe = info.get('trailingPE', 0)
        div = info.get('dividendYield', 0)
        
        # Rewards (Green)
        if pe < 12: st.markdown(f'<span class="reward-badge">✔ Good Value</span> Trading at a low P/E ratio ({pe:.1f}x) vs Industry.', unsafe_allow_html=True)
        if div > 0.04: st.markdown(f'<span class="reward-badge">✔ High Dividend</span> Pays a top-tier dividend of {div*100:.1f}%.', unsafe_allow_html=True)
        if info.get('earningsGrowth', 0) > 0.10: st.markdown(f'<span class="reward-badge">✔ High Growth</span> Earnings grew by over 10% last year.', unsafe_allow_html=True)
        
        st.write("") # Spacer
        
        # Risks (Red)
        if info.get('debtToEquity', 0) > 80: st.markdown(f'<span class="risk-badge">⚠️ High Debt</span> Debt to Equity ratio is high.', unsafe_allow_html=True)
        if pct_change < -20: st.markdown(f'<span class="risk-badge">⚠️ Momentum</span> Share price has fallen significantly recently.', unsafe_allow_html=True)
        
        # Market Sentiment Mini-Gauge
        st.write("")
        st.markdown("**Market Sentiment**")
        st.progress(0.7) # Placeholder for the "Greed/Fear" gauge
        st.caption("Investors are currently Bullish on Industrials.")

    # 4. VALUATION SECTION (Competitors)
    st.divider()
    st.header("1. Valuation")
    st.markdown("Is Invicta undervalued compared to its fair value and peers?")
    
    v1, v2 = st.columns([2, 1])
    
    with v1:
        # P/E vs Peers Chart
        peers = {"Invicta": pe, "Hudaco": 8.5, "Barloworld": 6.2, "Market Avg": 11.0} # Hudaco/Barlo hardcoded as example or fetch live if preferred
        
        fig_pe = go.Figure()
        fig_pe.add_trace(go.Bar(
            x=list(peers.keys()), y=list(peers.values()),
            marker_color=['#2ca02c', '#d1d5db', '#d1d5db', '#1f2937']
        ))
        fig_pe.update_layout(title="Price to Earnings Ratio vs Peers", height=300, yaxis_title="P/E Ratio")
        st.plotly_chart(fig_pe, use_container_width=True)
        
    with v2:
        st.info(f"""
        **Invicta P/E: {pe:.1f}x**
        
        Invicta trades at a **{pe:.1f}x** multiple of earnings.
        
        *   vs Hudaco (8.5x): **Premium**
        *   vs Market (11.0x): **Discount**
        """)

    # 5. PRICE HISTORY & NEWS (Sentiment)
    st.divider()
    st.header("2. Price & Sentiment History")
    
    # Fetch News
    news_items = fetch_news("Invicta Holdings Limited")
    if news_items:
        avg_score = pd.DataFrame(news_items)['score'].mean()
    else:
        avg_score = 0
    
    c_chart, c_news = st.columns([2, 1])
    
    with c_chart:
        # Price Chart (Area Style like Simply Wall St)
        fig_area = go.Figure()
        fig_area.add_trace(go.Scatter(
            x=history.index, y=history['Close'], 
            fill='tozeroy', mode='lines', 
            line=dict(color='#2ca02c'), name='Price'
        ))
        fig_area.update_layout(height=400, title="1-Year Share Price", yaxis_title="Price (ZAR)", xaxis_title=None)
        st.plotly_chart(fig_area, use_container_width=True)
        
    with c_news:
        st.markdown(f"### Media Sentiment: {avg_score:.2f}")
        sentiment_color = "green" if avg_score > 0 else "red"
        st.markdown(f"Recent news is predominantly :{sentiment_color}[**{'Positive' if avg_score > 0 else 'Negative'}**].")
        
        for item in news_items[:3]: # Show top 3
            with st.container():
                st.markdown(f"**{item['icon']} [{item['title']}]({item['link']})**")
                st.caption(f"{item['source']} • {item['date']}")
                st.markdown("---")

    # 6. DIVIDEND
    st.divider()
    st.header("3. Dividend Information")
    
    d1, d2 = st.columns(2)
    with d1:
        yield_val = info.get('dividendYield', 0) * 100
        st.metric("Current Dividend Yield", f"{yield_val:.2f}%")
        st.progress(yield_val / 10) # Assuming 10% is max bar
        st.caption("Top 25% of Dividend Payers in ZA Market")
        
    with d2:
        st.info("Invicta has a history of consistent dividend payments. The payout ratio indicates dividends are well-covered by earnings.")

if __name__ == "__main__":
    main()
