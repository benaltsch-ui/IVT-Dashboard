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

# --- HELPER: RESOLVE GOOGLE LINKS ---
def get_final_url(url):
    try:
        session = requests.Session()
        session.headers.update({"User-Agent": "Mozilla/5.0"})
        response = session.head(url, allow_redirects=True, timeout=5)
        return response.url
    except:
        return url

# --- HELPER: STABLE SCRAPER ---
def get_article_content(url):
    try:
        downloaded = trafilatura.fetch_url(url)
        if downloaded:
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

# --- DATA FETCHING ---
@st.cache_data(ttl=900)
def get_market_data(ticker):
    try:
        stock = yf.Ticker(ticker)
        return stock.history(period="1y"), stock.info
    except:
        return pd.DataFrame(), {}

@st.cache_data(ttl=900)
def get_competitor_data():
    # Comparing Invicta (IVT) vs Hudaco (HDC) and Barloworld (BAW)
    tickers = {'Invicta (IVT)': 'IVT.JO', 'Hudaco (HDC)': 'HDC.JO', 'Barloworld (BAW)': 'BAW.JO'}
    
    comp_metrics = []
    comp_history = pd.DataFrame()
    
    for name, sym in tickers.items():
        try:
            stock = yf.Ticker(sym)
            info = stock.info
            hist = stock.history(period="1y")
            
            # Normalize price to % growth (starts at 0)
            if not hist.empty:
                start_price = hist['Close'].iloc[0]
                hist['Rel_Perf'] = ((hist['Close'] - start_price) / start_price) * 100
                comp_history[name] = hist['Rel_Perf']
                
                # Metrics
                pe = info.get('trailingPE', 0)
                div = info.get('dividendYield', 0)
                div_fmt = f"{div*100:.2f}%" if div else "N/A"
                
                comp_metrics.append({
                    "Company": name,
                    "Current Price": f"R {hist['Close'].iloc[-1]:.2f}",
                    "1Y Performance": f"{comp_history[name].iloc[-1]:.2f}%",
                    "P/E Ratio": f"{pe:.2f}" if pe else "N/A",
                    "Div Yield": div_fmt
                })
        except:
            continue
            
    return pd.DataFrame(comp_metrics), comp_history

@st.cache_data(ttl=3600)
def get_google_news(query):
    encoded = query.replace(" ", "%20")
    rss_url = f"https://news.google.com/rss/search?q={encoded}+South+Africa&hl=en-ZA&gl=ZA&ceid=ZA:en"
    feed = feedparser.parse(rss_url)
    if not feed.entries: return []
    
    news_items = []
    limit = min(len(feed.entries), 6) # Limit to 6
    
    progress_text = "Reading news stream..."
    my_bar = st.progress(0, text=progress_text)
    
    for i, entry in enumerate(feed.entries[:limit]):
        my_bar.progress((i + 1) / limit, text=f"Analyzing: {entry.title[:20]}...")
        
        try:
            dt = parsedate_to_datetime(entry.published)
            clean_date = dt.strftime("%d %b %Y")
        except:
            clean_date = "Recent"

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
        
        st.info("**Chart Guide:** 🟢 Green bars = Buying Pressure | 🔴 Red bars = Selling Pressure | 🟠 Orange Line = Average Volume (Spikes above this indicate major news events).")
    
    # 2. NEWS
    st.divider()
    st.subheader("📰 News & Media Sentiment")
    
    news = get_google_news("Invicta Holdings Limited")
    if not news:
        news = get_google_news("JSE Industrial Engineering")
        
    avg_score = 0
    if news:
        df = pd.DataFrame(news)
        avg_score = df['Score'].mean()
        
        # News List
        for i, row in df.iterrows():
            with st.expander(f"{row['Icon']} {row['title']}"):
                m1, m2 = st.columns([3, 1])
                m1.caption(f"**Source:** {row['source']} | **Date:** {row['date']}")
                m2.caption(f"**{row['Method']}**")
                st.info(f"💡 {row['Explanation']}")
                st.markdown(f"[Read Article]({row['link']})")
    else:
        st.write("No news found.")

    # 3. COMPETITOR ANALYSIS & FINAL VERDICT
    st.divider()
    st.subheader("🏆 Competitor Intelligence & Final Verdict")
    
    with st.spinner("Analyzing Peers (Hudaco & Barloworld)..."):
        comp_df, comp_hist = get_competitor_data()
    
    if not comp_df.empty:
        c1, c2 = st.columns([1, 1])
        
        with c1:
            st.markdown("##### 1-Year Relative Performance")
            st.caption("How much would R100 invested 1 year ago be worth now?")
            
            # Line Chart for Comparison
            fig_comp = go.Figure()
            for col in comp_hist.columns:
                # Highlight Invicta
                width = 4 if "Invicta" in col else 2
                fig_comp.add_trace(go.Scatter(x=comp_hist.index, y=comp_hist[col], mode='lines', name=col, line=dict(width=width)))
            
            fig_comp.update_layout(height=350, margin=dict(l=0, r=0, t=10, b=0), hovermode='x unified', yaxis_title="Growth (%)")
            st.plotly_chart(fig_comp, use_container_width=True)
            
        with c2:
            st.markdown("##### Valuation Benchmarking")
            st.dataframe(comp_df, hide_index=True, use_container_width=True)
            
            # FINAL SENTIMENT CALCULATION
            # We combine Media Sentiment (avg_score) with Market Performance
            ivt_growth = comp_hist['Invicta (IVT)'].iloc[-1]
            
            total_score = (avg_score * 50) + (ivt_growth * 0.5) # Weighted score
            
            st.markdown("---")
            st.markdown("##### 🤖 The AI Verdict")
            
            if total_score > 10:
                verdict = "STRONG BUY / BULLISH"
                color = "green"
                reason = "Both media sentiment and relative market performance are positive."
            elif total_score > 0:
                verdict = "HOLD / MODERATE"
                color = "orange"
                reason = "Market performance is stable, but media sentiment is mixed."
            else:
                verdict = "SELL / BEARISH"
                color = "red"
                reason = "Underperformance relative to peers and negative media sentiment."
                
            st.markdown(f"### Total Sentiment Score: :{color}[{verdict}]")
            st.caption(f"Based on Media Analysis ({avg_score:.2f}) and Competitor Benchmarking.")
            st.success(reason)

if __name__ == "__main__":
    main()
