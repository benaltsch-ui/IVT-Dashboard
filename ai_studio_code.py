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
st.set_page_config(page_title="Invicta Holdings Pro Dashboard", layout="wide", page_icon="🏭")

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

# --- DATA FETCHING ---
@st.cache_data(ttl=900)
def get_market_data(ticker, period="1y"):
    try:
        stock = yf.Ticker(ticker)
        # Fetch longer history for 200 SMA calculation if needed, then slice back
        history = stock.history(period="2y") 
        info = stock.info
        return history, info
    except:
        return pd.DataFrame(), {}

@st.cache_data(ttl=3600) 
def get_competitor_financials():
    tickers = {"Invicta (IVT)": "IVT.JO", "Hudaco (HDC)": "HDC.JO", "Barloworld (BAW)": "BAW.JO"}
    metrics = []
    history_df = pd.DataFrame()
    
    for name, sym in tickers.items():
        try:
            stock = yf.Ticker(sym)
            hist = stock.history(period="1y")
            if not hist.empty:
                start_price = hist['Close'].iloc[0]
                hist['Growth'] = ((hist['Close'] - start_price) / start_price) * 100
                history_df[name] = hist['Growth']
                current_price = hist['Close'].iloc[-1]
            else:
                current_price = 0

            info = stock.info
            metrics.append({
                "Company": name,
                "Price": current_price,
                "P/E Ratio": info.get('trailingPE', 0),
                "Div Yield (%)": (info.get('dividendYield', 0) or 0) * 100,
                "Market Cap (B)": (info.get('marketCap', 0) or 0) / 1e9,
                "1Y Return (%)": history_df[name].iloc[-1] if not hist.empty else 0
            })
        except:
            continue
            
    return pd.DataFrame(metrics), history_df

@st.cache_data(ttl=3600)
def fetch_news_score(query, article_limit=5):
    encoded = query.replace(" ", "%20")
    rss_url = f"https://news.google.com/rss/search?q={encoded}+South+Africa&hl=en-ZA&gl=ZA&ceid=ZA:en"
    feed = feedparser.parse(rss_url)
    
    if not feed.entries: return 0.0, []
    
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
            "Score": score,
            "Explanation": expl,
            "Method": method
        })
        
    avg_score = pd.DataFrame(news_items)['Score'].mean() if news_items else 0.0
    return avg_score, news_items

# --- MAIN APP ---
def main():
    # --- SIDEBAR CONTROLS ---
    st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/8/87/JSE_Logo.png/640px-JSE_Logo.png", width=100)
    st.sidebar.title("Controls")
    # We fetch 2y data by default to calculate 200SMA, but we display based on this selection
    display_period = st.sidebar.selectbox("Chart View", ["3mo", "6mo", "1y", "2y"], index=2)
    
    st.title("🏭 Invicta Holdings (IVT)")
    st.caption("Strategic Intelligence Dashboard")

    # 1. FETCH MAIN DATA
    # Fetch 2y history to ensure we have enough data points for the 200 SMA
    history_full, info = get_market_data("IVT.JO", period="2y")
    
    if not history_full.empty:
        # --- CALCULATE TECHNICAL INDICATORS ---
        # 1. Moving Averages
        history_full['SMA_50'] = history_full['Close'].rolling(window=50).mean()
        history_full['SMA_200'] = history_full['Close'].rolling(window=200).mean()
        
        # 2. RSI (14-Day)
        delta = history_full['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        history_full['RSI'] = 100 - (100 / (1 + rs))
        
        # 3. Volume Average
        history_full['Vol_Avg'] = history_full['Volume'].rolling(window=30).mean()

        # --- SLICE DATA FOR DISPLAY ---
        # Filter the dataframe based on the user selection in sidebar
        if display_period == "3mo": slice_days = 90
        elif display_period == "6mo": slice_days = 180
        elif display_period == "1y": slice_days = 365
        else: slice_days = 730
        
        history = history_full.tail(slice_days).copy()
        
        # Metrics
        curr = history['Close'].iloc[-1]
        prev = history['Close'].iloc[-2]
        pct = ((curr - prev) / prev) * 100
        current_rsi = history['RSI'].iloc[-1]
        
        # --- TOP LEVEL METRICS ---
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Share Price", f"R {curr:.2f}", f"{pct:.2f}%")
        m2.metric("P/E Ratio", f"{info.get('trailingPE', 'N/A')}")
        m3.metric("RSI (14-Day)", f"{current_rsi:.1f}", "Overbought" if current_rsi > 70 else "Oversold" if current_rsi < 30 else "Neutral")
        m4.metric("Market Cap", f"R {info.get('marketCap', 0)/1e9:.2f} B")

        # --- TABS ---
        tab_market, tab_comp, tab_sent = st.tabs([
            "📈 Market & Technicals", 
            "📊 Competitor Benchmarks", 
            "📰 AI Sentiment"
        ])

        # --- TAB 1: MARKET & TECHNICALS (ENHANCED) ---
        with tab_market:
            c_main, c_sidebar = st.columns([3, 1])
            
            with c_main:
                st.subheader("Technical Price Analysis")
                fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1, row_heights=[0.7, 0.3])
                
                # Candlestick
                fig.add_trace(go.Candlestick(x=history.index, open=history['Open'], high=history['High'], low=history['Low'], close=history['Close'], name="Price"), row=1, col=1)
                
                # Moving Averages
                fig.add_trace(go.Scatter(x=history.index, y=history['SMA_50'], line=dict(color='royalblue', width=1.5), name="50-Day SMA"), row=1, col=1)
                fig.add_trace(go.Scatter(x=history.index, y=history['SMA_200'], line=dict(color='firebrick', width=1.5), name="200-Day SMA"), row=1, col=1)
                
                # Volume
                colors = ['#EA4335' if row['Open'] - row['Close'] > 0 else '#34A853' for index, row in history.iterrows()]
                fig.add_trace(go.Bar(x=history.index, y=history['Volume'], marker_color=colors, name="Volume"), row=2, col=1)
                fig.add_trace(go.Scatter(x=history.index, y=history['Vol_Avg'], mode='lines', name="Avg Vol", line=dict(color='orange', dash='dot')), row=2, col=1)
                
                fig.update_layout(height=600, xaxis_rangeslider_visible=False, showlegend=True, legend=dict(orientation="h", y=1.02, x=0))
                fig.update_yaxes(title_text="Price (ZAR)", row=1, col=1)
                fig.update_yaxes(title_text="Volume", row=2, col=1)
                st.plotly_chart(fig, use_container_width=True)
                
                st.info("💡 **Legend:** 🔵 **Blue Line** = 50-Day Avg (Short Term Trend) | 🔴 **Red Line** = 200-Day Avg (Long Term Trend). If Blue crosses above Red, it's a 'Golden Cross' (Bullish).")

            with c_sidebar:
                # TRADING STATISTICS SECTION
                st.subheader("Trading Data")
                
                # 52 Week High/Low
                high_52 = history_full['High'].tail(252).max()
                low_52 = history_full['Low'].tail(252).min()
                st.metric("52-Week High", f"R {high_52:.2f}")
                st.metric("52-Week Low", f"R {low_52:.2f}")
                
                st.markdown("---")
                
                # Volatility
                volatility = history['Close'].pct_change().std() * (252**0.5) * 100
                st.metric("Annualized Volatility", f"{volatility:.1f}%")
                
                # Trend Status
                st.markdown("---")
                st.markdown("**Trend Status**")
                sma_50_val = history['SMA_50'].iloc[-1]
                sma_200_val = history['SMA_200'].iloc[-1]
                
                if curr > sma_200_val:
                    st.success("Long-Term: Bullish (Above 200 SMA)")
                else:
                    st.error("Long-Term: Bearish (Below 200 SMA)")
                    
                if curr > sma_50_val:
                    st.success("Short-Term: Bullish (Above 50 SMA)")
                else:
                    st.warning("Short-Term: Bearish (Below 50 SMA)")

            # CURRENCY SECTION (Bottom Row of Tab 1)
            st.divider()
            c_forex, c_info = st.columns([2, 1])
            with c_forex:
                st.subheader("🇿🇦 Currency Correlation (USD/ZAR)")
                with st.spinner("Loading Forex..."):
                    forex = yf.Ticker("ZAR=X").history(period=display_period)
                if not forex.empty:
                    # Normalize
                    h_norm = (history['Close'] / history['Close'].iloc[0]) * 100
                    f_norm = (forex['Close'] / forex['Close'].iloc[0]) * 100
                    fig_c = go.Figure()
                    fig_c.add_trace(go.Scatter(x=history.index, y=h_norm, name='Invicta (Share Price)', line=dict(width=2)))
                    fig_c.add_trace(go.Scatter(x=forex.index, y=f_norm, name='USD/ZAR (Exchange Rate)', line=dict(dash='dot', color='orange')))
                    fig_c.update_layout(height=350, yaxis_title="Relative Performance (%)", hovermode="x unified")
                    st.plotly_chart(fig_c, use_container_width=True)
            with c_info:
                st.info("""
                **Why this matters?**
                Invicta imports engineering components (Bearings, Belts, Hydraulics).
                
                *   **Orange Line Up (Weak Rand):** Import costs rise, potentially squeezing margins.
                *   **Orange Line Down (Strong Rand):** Imports become cheaper, boosting margins.
                """)

        # --- TAB 2: COMPETITORS ---
        with tab_comp:
            st.markdown("### ⚔️ Invicta vs. Hudaco vs. Barloworld")
            with st.spinner("Analyzing Peers..."):
                comp_metrics, comp_history = get_competitor_financials()
            
            if not comp_metrics.empty:
                col_c1, col_c2 = st.columns([3, 2])
                with col_c1:
                    st.markdown("**1-Year Relative Performance**")
                    fig_rel = go.Figure()
                    colors = {"Invicta (IVT)": "#1f77b4", "Hudaco (HDC)": "#d62728", "Barloworld (BAW)": "#2ca02c"}
                    for col in comp_history.columns:
                        width = 4 if "Invicta" in col else 2
                        fig_rel.add_trace(go.Scatter(x=comp_history.index, y=comp_history[col], mode='lines', name=col, line=dict(width=width, color=colors.get(col, "gray"))))
                    fig_rel.update_layout(height=350, margin=dict(l=0, r=0, t=30, b=0), yaxis_title="Growth %")
                    st.plotly_chart(fig_rel, use_container_width=True)
                with col_c2:
                    st.markdown("**Valuation Table**")
                    # Clean formatting
                    styled_df = comp_metrics.copy()
                    styled_df['Price'] = styled_df['Price'].apply(lambda x: f"R {x:.2f}")
                    styled_df['P/E Ratio'] = styled_df['P/E Ratio'].apply(lambda x: f"{x:.2f}")
                    styled_df['Div Yield (%)'] = styled_df['Div Yield (%)'].apply(lambda x: f"{x:.2f}%")
                    styled_df['Market Cap (B)'] = styled_df['Market Cap (B)'].apply(lambda x: f"R {x:.2f} B")
                    styled_df.drop(columns=['1Y Return (%)'], inplace=True) # Already in chart
                    st.dataframe(styled_df, hide_index=True, use_container_width=True)

        # --- TAB 3: SENTIMENT ---
        with tab_sent:
            with st.spinner("AI is reading the news..."):
                ivt_score, ivt_news = fetch_news_score("Invicta Holdings Limited", article_limit=6)
                hdc_score, _ = fetch_news_score("Hudaco Industries", article_limit=3)
                baw_score, _ = fetch_news_score("Barloworld Limited", article_limit=3)

            s1, s2 = st.columns([1, 2])
            with s1:
                st.markdown("#### Sentiment Battle")
                comp_data = {'Company': ['Invicta', 'Hudaco', 'Barloworld'], 'Score': [ivt_score, hdc_score, baw_score]}
                df_comp = pd.DataFrame(comp_data)
                fig_comp = go.Figure(go.Bar(x=df_comp['Score'], y=df_comp['Company'], orientation='h', marker_color=['#1f77b4', '#d62728', '#2ca02c']))
                fig_comp.update_layout(height=250, margin=dict(l=0, r=0, t=0, b=0), xaxis_title="Score (-1 to +1)")
                st.plotly_chart(fig_comp, use_container_width=True)
                
            with s2:
                st.markdown("#### Invicta News Feed")
                if ivt_news:
                    for item in ivt_news:
                        with st.expander(f"{item['Icon']} {item['title']}"):
                            c_t, c_i = st.columns([3, 1])
                            with c_t:
                                st.caption(f"{item['source']} | {item['date']}")
                                st.write(item['Explanation'])
                                st.markdown(f"[Read Article]({item['link']})")
                            with c_i:
                                st.progress((item['Score'] + 1) / 2)
                                st.caption(f"Impact: {item['Score']:.2f}")

if __name__ == "__main__":
    main()
