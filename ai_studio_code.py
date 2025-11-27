import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import feedparser
import requests
import trafilatura
from email.utils import parsedate_to_datetime
import datetime

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
    
    # Financial Dictionary (Customized)
    financial_lexicon = {
        'shoot': 2.0, 'surged': 3.0, 'jumped': 2.5, 'climbed': 2.0, 'soared': 3.0, 'green': 1.5,
        'plunged': -3.0, 'tumbled': -2.5, 'slumped': -2.5, 'red': -1.5,
        'dividend': 2.0, 'earnings': 1.5, 'profit': 2.0, 'growth': 2.0,
        'resilient': 2.0, 'strong': 2.0, 'up': 1.0, 'down': -1.0,
        'acquisition': 1.5, 'buyback': 1.5, 'challenging': -1.0, 'headwinds': -1.5,
        'ghost': 0.0, 'busy': 1.0, 'mixed': 0.0, 'flat': -0.5
    }
    sia.lexicon.update(financial_lexicon)
    return sia

try:
    sia = load_sentiment_resources()
except Exception as e:
    st.error(f"Error loading AI: {e}")
    st.stop()

# --- HELPER FUNCTIONS ---
def format_large_number(num):
    if num is None: return "N/A"
    if isinstance(num, str): return num
    if num >= 1e9: return f"R {num/1e9:.2f} B"
    elif num >= 1e6: return f"R {num/1e6:.2f} M"
    else: return f"R {num:,.2f}"

def calculate_change(current, previous):
    if previous is None or previous == 0: return 0.0
    return ((current - previous) / abs(previous)) * 100

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
        history = stock.history(period="2y") 
        info = stock.info
        financials = stock.financials
        quarterly_fin = stock.quarterly_financials
        balance_sheet = stock.balance_sheet
        
        # --- FIX: Convert Cents to Rands ---
        if not history.empty:
            current_price = history['Close'].iloc[-1]
            if current_price > 500: # Threshold to detect if data is in Cents (ZAc)
                cols_to_fix = ['Open', 'High', 'Low', 'Close']
                history[cols_to_fix] = history[cols_to_fix] / 100
        
        return history, info, financials, quarterly_fin, balance_sheet
    except:
        return pd.DataFrame(), {}, pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

@st.cache_data(ttl=3600) 
def get_macro_data(period="1y"):
    tickers = {
        "Invicta": "IVT.JO",
        "USD/ZAR": "ZAR=X",
        "JSE Industrials": "STXIND.JO"
        # REMOVED GOLD (GC=F) AS REQUESTED
    }
    data = pd.DataFrame()
    for name, ticker in tickers.items():
        try:
            hist = yf.Ticker(ticker).history(period=period)
            if not hist.empty:
                hist.index = hist.index.tz_localize(None)
                data[name] = hist['Close']
        except:
            continue
    
    # Normalize Invicta in macro data too if needed
    if 'Invicta' in data.columns and data['Invicta'].mean() > 500:
        data['Invicta'] = data['Invicta'] / 100
        
    return data.ffill().bfill()

@st.cache_data(ttl=3600) 
def get_competitor_financials():
    tickers = {"Invicta (IVT)": "IVT.JO", "Hudaco (HDC)": "HDC.JO", "Barloworld (BAW)": "BAW.JO"}
    metrics = []
    history_df = pd.DataFrame()
    
    for name, sym in tickers.items():
        try:
            stock = yf.Ticker(sym)
            hist = stock.history(period="1y")
            info = stock.info
            
            if not hist.empty:
                # Fix ZAc to ZAR for competitors
                if hist['Close'].iloc[-1] > 500:
                    hist['Close'] = hist['Close'] / 100
                
                start_price = hist['Close'].iloc[0]
                hist['Growth'] = ((hist['Close'] - start_price) / start_price) * 100
                history_df[name] = hist['Growth']
                current_price = hist['Close'].iloc[-1]
            else:
                current_price = 0

            pe = info.get('trailingPE', 0)
            m_cap = (info.get('marketCap', 0) or 0) / 1e9
            
            # Competitor Dividend Fix
            div_rate = info.get('dividendRate', 0)
            if div_rate and current_price > 0:
                calc_yield = (div_rate / current_price) * 100
            else:
                calc_yield = (info.get('dividendYield', 0) or 0) * 100

            metrics.append({
                "Company": name,
                "Price": current_price,
                "P/E Ratio": pe,
                "Div Yield (%)": calc_yield,
                "Market Cap (B)": m_cap,
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
    st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/8/87/JSE_Logo.png/640px-JSE_Logo.png", width=100)
    st.sidebar.title("Controls")
    display_period = st.sidebar.selectbox("Chart View", ["3mo", "6mo", "1y", "2y"], index=2)
    
    st.title("🏭 Invicta Holdings (IVT)")
    st.caption("Strategic Intelligence Dashboard")

    history_full, info, financials, quarterly_fin, balance_sheet = get_market_data("IVT.JO", period="2y")
    
    if not history_full.empty:
        # --- CALCULATIONS ---
        # Technicals
        history_full['SMA_50'] = history_full['Close'].rolling(window=50).mean()
        history_full['SMA_200'] = history_full['Close'].rolling(window=200).mean()
        delta = history_full['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        history_full['RSI'] = 100 - (100 / (1 + rs))
        history_full['Vol_Avg'] = history_full['Volume'].rolling(window=30).mean()

        # Slice for display
        if display_period == "3mo": slice_days = 90
        elif display_period == "6mo": slice_days = 180
        elif display_period == "1y": slice_days = 365
        else: slice_days = 730
        history = history_full.tail(slice_days).copy()
        
        # --- METRICS CALCULATION ---
        curr = history['Close'].iloc[-1]
        prev = history['Close'].iloc[-2]
        pct = ((curr - prev) / prev) * 100
        current_rsi = history['RSI'].iloc[-1]
        
        eps = info.get('trailingEps', 0)
        pe_ratio = info.get('trailingPE', 0)
        
        # --- DIVIDEND YIELD FIX ---
        div_rate = info.get('dividendRate', None)
        
        if div_rate is not None and curr > 0:
            calculated_yield = (div_rate / curr) * 100
        else:
            raw_yield = info.get('dividendYield', 0) or 0
            if raw_yield > 0.5:
                calculated_yield = raw_yield
            else:
                calculated_yield = raw_yield * 100

        # --- TOP LEVEL METRICS ROW ---
        m1, m2, m3, m4, m5 = st.columns(5)
        
        m1.metric("Share Price", f"R {curr:.2f}", f"{pct:.2f}%")
        m2.metric("EPS (TTM)", f"R {eps:.2f}" if eps else "N/A")
        m3.metric("P/E Ratio", f"{pe_ratio:.2f}" if pe_ratio else "N/A")
        m4.metric("Dividend Yield", f"{calculated_yield:.2f}%")
        m5.metric("Market Cap", f"R {info.get('marketCap', 0)/1e9:.2f} B")

        # --- TABS ---
        tab_market, tab_macro, tab_fin, tab_comp, tab_sent = st.tabs([
            "📈 Market & Technicals",
            "🔗 Macro & Correlations", 
            "📊 Financial Health",
            "📊 Competitor Benchmarks", 
            "📰 AI Sentiment"
        ])

        # --- TAB 1: MARKET & TECHNICALS ---
        with tab_market:
            c_main, c_sidebar = st.columns([3, 1])
            with c_main:
                st.subheader("Price & Volume Analysis")
                fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1, row_heights=[0.7, 0.3])
                fig.add_trace(go.Candlestick(x=history.index, open=history['Open'], high=history['High'], low=history['Low'], close=history['Close'], name="Price"), row=1, col=1)
                fig.add_trace(go.Scatter(x=history.index, y=history['SMA_50'], line=dict(color='royalblue', width=1.5), name="50-Day SMA"), row=1, col=1)
                fig.add_trace(go.Scatter(x=history.index, y=history['SMA_200'], line=dict(color='firebrick', width=1.5), name="200-Day SMA"), row=1, col=1)
                colors = ['#EA4335' if row['Open'] - row['Close'] > 0 else '#34A853' for index, row in history.iterrows()]
                fig.add_trace(go.Bar(x=history.index, y=history['Volume'], marker_color=colors, name="Volume"), row=2, col=1)
                fig.add_trace(go.Scatter(x=history.index, y=history['Vol_Avg'], mode='lines', name="Avg Vol", line=dict(color='orange', dash='dot')), row=2, col=1)
                fig.update_layout(height=600, xaxis_rangeslider_visible=False, showlegend=True, legend=dict(orientation="h", y=1.02, x=0))
                fig.update_yaxes(title_text="Price (ZAR)", row=1, col=1)
                fig.update_yaxes(title_text="Volume", row=2, col=1)
                st.plotly_chart(fig, use_container_width=True)
                st.info("💡 **Chart Guide:** 🔵 **50-Day SMA** (Short Trend) | 🔴 **200-Day SMA** (Long Trend). **Volume:** Green Bars = Buying Pressure, Red Bars = Selling Pressure.")

            with c_sidebar:
                st.subheader("Trading Stats")
                st.metric("RSI (14-Day)", f"{current_rsi:.1f}", "Overbought" if current_rsi > 70 else "Oversold" if current_rsi < 30 else "Neutral")
                st.markdown("---")
                high_52 = history_full['High'].tail(252).max()
                low_52 = history_full['Low'].tail(252).min()
                st.metric("52-Week High", f"R {high_52:.2f}")
                st.metric("52-Week Low", f"R {low_52:.2f}")
                st.markdown("---")
                volatility = history['Close'].pct_change().std() * (252**0.5) * 100
                st.metric("Annualized Volatility", f"{volatility:.1f}%")
                st.markdown("---")
                sma_200_val = history['SMA_200'].iloc[-1]
                if curr > sma_200_val: st.success("Trend: Bullish (Above 200 SMA)")
                else: st.error("Trend: Bearish (Below 200 SMA)")

        # --- TAB 2: MACRO ---
        with tab_macro:
            st.subheader("🔗 Macro-Economic Correlations")
            st.caption("Comparing Invicta against Currency (USD/ZAR) and Sector benchmarks.")
            
            with st.spinner("Analyzing macro data..."):
                macro_df = get_macro_data(period=display_period)
            
            if not macro_df.empty and 'Invicta' in macro_df.columns:
                c_charts, c_stats = st.columns([3, 1])
                with c_charts:
                    st.markdown("#### 🌍 Relative Performance Comparison")
                    norm_df = (macro_df / macro_df.iloc[0]) * 100 - 100
                    fig_macro = go.Figure()
                    # Updated Colors: Removed Gold
                    colors = {'Invicta': '#1f77b4', 'USD/ZAR': 'orange', 'JSE Industrials': 'gray'}
                    for col in norm_df.columns:
                        width = 3 if col == 'Invicta' else 1.5
                        dash = 'solid' if col == 'Invicta' else 'dot'
                        fig_macro.add_trace(go.Scatter(x=norm_df.index, y=norm_df[col], name=col, line=dict(color=colors.get(col, 'black'), width=width, dash=dash)))
                    fig_macro.update_layout(height=400, yaxis_title="Performance (%)", hovermode="x unified")
                    st.plotly_chart(fig_macro, use_container_width=True)
                    
                with c_stats:
                    st.markdown("#### 🔢 Correlation")
                    st.caption("Correlation with Invicta (1.0 = High, -1.0 = Inverse)")
                    corr_matrix = macro_df.pct_change().corr()['Invicta'].drop('Invicta')
                    st.dataframe(corr_matrix, use_container_width=True)
                    
                st.divider()
                st.markdown("#### 📉 Dynamic Relationship: Invicta vs USD/ZAR")
                if 'USD/ZAR' in macro_df.columns:
                    rolling_corr = macro_df['Invicta'].rolling(window=30).corr(macro_df['USD/ZAR'])
                    fig_roll = go.Figure()
                    fig_roll.add_trace(go.Scatter(x=macro_df.index, y=rolling_corr, fill='tozeroy', name='30-Day Correlation'))
                    fig_roll.add_hline(y=0, line_dash="dash", line_color="black")
                    fig_roll.update_layout(height=250, yaxis_title="Correlation", yaxis_range=[-1, 1])
                    st.plotly_chart(fig_roll, use_container_width=True)
            else:
                st.warning("Macro data temporarily unavailable. Please refresh.")

        # --- TAB 3: FINANCIAL HEALTH (REVISED) ---
        with tab_fin:
            st.subheader("📊 Financial Health Comparison")
            
            # --- SECTION 1: ANNUAL COMPARISON ---
            st.markdown("### 🗓️ Annual Results (Full Financial Year)")
            if not financials.empty and financials.shape[1] >= 2:
                # Get Most Recent Year (Col 0) and Previous Year (Col 1)
                latest_date = financials.columns[0]
                prev_date = financials.columns[1]
                
                # Extract metrics
                metrics_list = ['Total Revenue', 'Gross Profit', 'EBITDA', 'Net Income', 'Basic EPS']
                
                fin_data = []
                for m in metrics_list:
                    try:
                        val_curr = financials.loc[m, latest_date] if m in financials.index else 0
                        val_prev = financials.loc[m, prev_date] if m in financials.index else 0
                        change = calculate_change(val_curr, val_prev)
                        
                        # Formatting
                        fin_data.append({
                            "Metric": m,
                            f"{latest_date.strftime('%Y') } (Latest)": format_large_number(val_curr),
                            f"{prev_date.strftime('%Y')} (Prior)": format_large_number(val_prev),
                            "Change (%)": f"{change:+.2f}%"
                        })
                    except:
                        continue
                
                df_annual_comp = pd.DataFrame(fin_data)
                st.dataframe(df_annual_comp, use_container_width=True, hide_index=True)
            else:
                st.warning("Insufficient Annual Data available for comparison.")

            st.divider()

            # --- SECTION 2: INTERIM / RECENT COMPARISON ---
            st.markdown("### ⏱️ Year to Date / Recent Trend (Interim)")
            
            has_quarterly = not quarterly_fin.empty and quarterly_fin.shape[1] >= 2
            
            if has_quarterly:
                st.caption(f"Comparing most recent reported period vs prior period.")
                q_latest_date = quarterly_fin.columns[0]
                q_prev_date = quarterly_fin.columns[1] # Comparing to sequential previous
                
                q_metrics_list = ['Total Revenue', 'Net Income', 'Operating Income']
                q_data = []
                
                for m in q_metrics_list:
                    try:
                        q_curr = quarterly_fin.loc[m, q_latest_date] if m in quarterly_fin.index else 0
                        q_prev = quarterly_fin.loc[m, q_prev_date] if m in quarterly_fin.index else 0
                        q_change = calculate_change(q_curr, q_prev)
                        
                        q_data.append({
                            "Metric": m,
                            f"{q_latest_date.strftime('%b %Y')}": format_large_number(q_curr),
                            f"{q_prev_date.strftime('%b %Y')}": format_large_number(q_prev),
                            "Change (%)": f"{q_change:+.2f}%"
                        })
                    except:
                        continue
                
                df_q_comp = pd.DataFrame(q_data)
                st.dataframe(df_q_comp, use_container_width=True, hide_index=True)
            
            else:
                # FALLBACK: If quarterly data is missing (common for JSE stocks on Yahoo), show TTM vs Annual
                st.info("Detailed Interim/Quarterly data not available via API. Showing TTM vs Last Annual.")
                ttm_data = []
                
                # Revenue
                rev_ttm = info.get('totalRevenue', 0)
                rev_last = financials.iloc[0]['Total Revenue'] if not financials.empty else 0
                ttm_data.append({"Metric": "Revenue", "TTM (Current)": format_large_number(rev_ttm), "Last Annual": format_large_number(rev_last)})
                
                # EBITDA
                ebitda_ttm = info.get('ebitda', 0)
                ebitda_last = financials.iloc[0]['EBITDA'] if (not financials.empty and 'EBITDA' in financials.index) else 0
                ttm_data.append({"Metric": "EBITDA", "TTM (Current)": format_large_number(ebitda_ttm), "Last Annual": format_large_number(ebitda_last)})

                st.dataframe(pd.DataFrame(ttm_data), use_container_width=True, hide_index=True)

            # --- VISUALIZATION ---
            st.markdown("#### 📉 Performance Visualization")
            if not financials.empty:
                fin_T = financials.T.iloc[:4][::-1] # Last 4 years reversed
                fig_fin = go.Figure()
                fig_fin.add_trace(go.Bar(x=fin_T.index, y=fin_T['Total Revenue'], name='Revenue', marker_color='#1f77b4'))
                
                if 'Net Income' in fin_T.columns:
                    fig_fin.add_trace(go.Bar(x=fin_T.index, y=fin_T['Net Income'], name='Net Income', marker_color='#2ca02c'))
                    
                fig_fin.update_layout(barmode='group', height=350, title="Annual Revenue vs Net Income Trend")
                st.plotly_chart(fig_fin, use_container_width=True)


        # --- TAB 4: COMPETITORS ---
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
                    styled_df = comp_metrics.copy()
                    styled_df['Price'] = styled_df['Price'].apply(lambda x: f"R {x:.2f}")
                    styled_df['P/E Ratio'] = styled_df['P/E Ratio'].apply(lambda x: f"{x:.2f}")
                    styled_df['Div Yield (%)'] = styled_df['Div Yield (%)'].apply(lambda x: f"{x:.2f}%")
                    styled_df['Market Cap (B)'] = styled_df['Market Cap (B)'].apply(lambda x: f"R {x:.2f} B")
                    styled_df.drop(columns=['1Y Return (%)'], inplace=True)
                    st.dataframe(styled_df, hide_index=True, use_container_width=True)

        # --- TAB 5: SENTIMENT ---
        with tab_sent:
            with st.spinner("AI is reading the news..."):
                ivt_score, ivt_news = fetch_news_score("Invicta Holdings", article_limit=6)
                hdc_score, _ = fetch_news_score("Hudaco Industries", article_limit=3)
                baw_score, _ = fetch_news_score("Barloworld", article_limit=3)

            s1, s2 = st.columns([1, 2])
            with s1:
                st.markdown("#### Sentiment Battle")
                comp_data = {'Company': ['Invicta', 'Hudaco', 'Barloworld'], 'Score': [ivt_score, hdc_score, baw_score]}
                df_comp = pd.DataFrame(comp_data)
                fig_comp = go.Figure(go.Bar(x=df_comp['Score'], y=df_comp['Company'], orientation='h', marker_color=['#1f77b4', '#d62728', '#2ca02c']))
                fig_comp.update_layout(height=250, margin=dict(l=0, r=0, t=0, b=0), xaxis=dict(range=[-1.0, 1.0], title="Score (-1 to +1)"))
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
