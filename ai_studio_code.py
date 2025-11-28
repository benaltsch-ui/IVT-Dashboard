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
    try:
        val = float(num)
    except:
        return "N/A"
        
    if val >= 1e9: return f"R {val/1e9:.2f} B"
    elif val >= 1e6: return f"R {val/1e6:.2f} M"
    else: return f"R {val:,.2f}"

def calculate_change(current, previous):
    try:
        if previous is None or previous == 0: return 0.0
        return ((current - previous) / abs(previous)) * 100
    except:
        return 0.0

def safe_get_financial_value(df, metric_name, col_index=0):
    try:
        if df.empty: return 0
        if metric_name in df.index:
            return df.loc[metric_name].iloc[col_index]
        return 0
    except:
        return 0

def get_final_url(url):
    try:
        session = requests.Session()
        session.headers.update({"User-Agent": "Mozilla/5.0"})
        response = session.head(url, allow_redirects=True, timeout=5)
        return response.url
    except:
        return url

# 🔧 Improved: actually download HTML and extract body text
def get_article_content(url):
    try:
        session = requests.Session()
        session.headers.update({
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/122.0 Safari/537.36"
            )
        })
        resp = session.get(url, timeout=10)
        if resp.status_code == 200:
            extracted = trafilatura.extract(
                resp.text,
                include_comments=False,
                include_links=False,
                favor_recall=True
            )
            # Ignore cookie pages / very short junk
            if extracted and len(extracted) > 500:
                return extracted
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
    
    if 'Invicta' in data.columns and data['Invicta'].mean() > 500:
        data['Invicta'] = data['Invicta'] / 100
        
    return data.ffill().bfill()

@st.cache_data(ttl=3600) 
def get_competitor_financials():
    tickers = {
        "Invicta (IVT)": "IVT.JO",
        "Hudaco (HDC)": "HDC.JO",
        "Barloworld (BAW)": "BAW.JO",
        "Bell Equipment (BEL)": "BEL.JO",
        "Master Drilling (MDI)": "MDI.JO",
        "enX Group (ENX)": "ENX.JO"
    }
    metrics = []
    history_df = pd.DataFrame()
    
    for name, sym in tickers.items():
        try:
            stock = yf.Ticker(sym)
            hist = stock.history(period="1y")
            info = stock.info
            
            if not hist.empty:
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
            
            # Additional Ratios for Comparison
            gross_margin = info.get('grossMargins', 0)
            op_margin = info.get('operatingMargins', 0)
            roe = info.get('returnOnEquity', 0)
            
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
                "Gross Margin (%)": gross_margin * 100 if gross_margin else 0,
                "Op Margin (%)": op_margin * 100 if op_margin else 0,
                "ROE (%)": roe * 100 if roe else 0,
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
            score, label, icon, expl = analyze_content(full_text, "Full Text (article body)")
            snippet = full_text[:600] + "..."
            method = "✅ Full Text"
        else:
            raw_desc = entry.get('description', entry.title)
            import re
            clean_desc = re.sub('<.*?>', '', raw_desc)
            score, label, icon, expl = analyze_content(clean_desc, "Snippet (headline/description)")
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
            "snippet": snippet,          # 👈 text AI actually read
            "Sentiment": label,
            "Icon": icon,
            "Score": score,
            "Explanation": expl,
            "Method": method             # 👈 Full Text vs Snippet
        })
        
    avg_score = pd.DataFrame(news_items)['Score'].mean() if news_items else 0.0
    return avg_score, news_items

# --- MAIN APP ---
def main():
    st.sidebar.title("Controls")
    display_period = st.sidebar.selectbox("Chart View", ["3mo", "6mo", "1y", "2y"], index=2)
    
    st.title("🏭 Invicta Holdings (IVT)")
    st.caption("Strategic Intelligence Dashboard")

    history_full, info, financials, quarterly_fin, balance_sheet = get_market_data("IVT.JO", period="2y")
    
    if not history_full.empty:
        # --- TECHNICAL CALCULATIONS ---
        history_full['SMA_50'] = history_full['Close'].rolling(window=50).mean()
        history_full['SMA_200'] = history_full['Close'].rolling(window=200).mean()
        
        delta = history_full['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        history_full['RSI'] = 100 - (100 / (1 + rs))
        
        history_full['EMA_12'] = history_full['Close'].ewm(span=12, adjust=False).mean()
        history_full['EMA_26'] = history_full['Close'].ewm(span=26, adjust=False).mean()
        history_full['MACD'] = history_full['EMA_12'] - history_full['EMA_26']
        history_full['Signal_Line'] = history_full['MACD'].ewm(span=9, adjust=False).mean()

        history_full['Vol_Avg'] = history_full['Volume'].rolling(window=30).mean()

        if display_period == "3mo": slice_days = 90
        elif display_period == "6mo": slice_days = 180
        elif display_period == "1y": slice_days = 365
        else: slice_days = 730
        history = history_full.tail(slice_days).copy()
        
        curr = history['Close'].iloc[-1]
        prev = history['Close'].iloc[-2]
        pct = ((curr - prev) / prev) * 100
        current_rsi = history['RSI'].iloc[-1]
        sma_50_val = history['SMA_50'].iloc[-1]
        sma_200_val = history['SMA_200'].iloc[-1]
        macd_val = history['MACD'].iloc[-1]
        signal_val = history['Signal_Line'].iloc[-1]
        
        eps = info.get('trailingEps', 0)
        pe_ratio = info.get('trailingPE', 0)
        
        div_rate = info.get('dividendRate', None)
        if div_rate is not None and curr > 0:
            calculated_yield = (div_rate / curr) * 100
        else:
            raw_yield = info.get('dividendYield', 0) or 0
            if raw_yield > 0.5: calculated_yield = raw_yield
            else: calculated_yield = raw_yield * 100

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

        with tab_market:
            c_main, c_sidebar = st.columns([3, 1])
            with c_main:
                st.subheader("Price & Volume Analysis")
                fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1, row_heights=[0.7, 0.3])
                
                # Price Trace
                fig.add_trace(go.Candlestick(
                    x=history.index,
                    open=history['Open'],
                    high=history['High'],
                    low=history['Low'],
                    close=history['Close'],
                    name="Price"
                ), row=1, col=1)
                fig.add_trace(go.Scatter(
                    x=history.index,
                    y=history['SMA_50'],
                    line=dict(color='royalblue', width=1.5),
                    name="50-Day SMA"
                ), row=1, col=1)
                fig.add_trace(go.Scatter(
                    x=history.index,
                    y=history['SMA_200'],
                    line=dict(color='firebrick', width=1.5),
                    name="200-Day SMA"
                ), row=1, col=1)
                
                # Volume Trace
                colors = ['#EA4335' if row['Open'] - row['Close'] > 0 else '#34A853' for index, row in history.iterrows()]
                fig.add_trace(go.Bar(
                    x=history.index, 
                    y=history['Volume'], 
                    marker_color=colors, 
                    name="Volume",
                    hovertemplate='<b>Date</b>: %{x|%d %b %Y}<br><b>Volume</b>: %{y:,} shares<extra></extra>'
                ), row=2, col=1)
                
                fig.add_trace(go.Scatter(
                    x=history.index,
                    y=history['Vol_Avg'],
                    mode='lines',
                    name="Avg Vol",
                    line=dict(color='orange', dash='dot')
                ), row=2, col=1)
                
                fig.update_layout(
                    height=600,
                    xaxis_rangeslider_visible=False,
                    showlegend=True,
                    legend=dict(orientation="h", y=1.02, x=0)
                )
                fig.update_yaxes(title_text="Price (ZAR)", row=1, col=1)
                fig.update_yaxes(title_text="Volume (Shares)", tickformat=".2s", row=2, col=1)
                
                st.plotly_chart(fig, use_container_width=True)
                
                st.markdown("### 🔑 Quick Chart Legend")
                l1, l2, l3, l4 = st.columns(4)
                l1.info("**SMA (Trend)**\n\nPrice > Lines = 🟢 Bullish\n\nPrice < Lines = 🔴 Bearish")
                l2.info("**RSI (Momentum)**\n\n> 70 = High/Sell ⚠️\n\n< 30 = Low/Buy 🟢")
                l3.info("**MACD (Signal)**\n\nLine > Signal = 🟢 Positive\n\nLine < Signal = 🔴 Negative")
                l4.info("**Volume**\n\nGreen = Buying Pressure\n\nRed = Selling Pressure")

            with c_sidebar:
                st.subheader("💡 AI Technical Verdict")
                
                if curr > sma_50_val and curr > sma_200_val:
                    trend_msg = "Bullish"
                    trend_icon = "🟢"
                    trend_expl = f"Price (R {curr:.2f}) is trading above both the 50-day (R {sma_50_val:.2f}) and 200-day (R {sma_200_val:.2f}) averages."
                elif curr < sma_50_val and curr < sma_200_val:
                    trend_msg = "Bearish"
                    trend_icon = "🔴"
                    trend_expl = f"Price (R {curr:.2f}) is trading below both the 50-day (R {sma_50_val:.2f}) and 200-day (R {sma_200_val:.2f}) averages."
                else:
                    trend_msg = "Neutral"
                    trend_icon = "⚪"
                    trend_expl = f"Price is currently consolidating between the 50-day (R {sma_50_val:.2f}) and 200-day (R {sma_200_val:.2f}) averages."
                
                if current_rsi > 70:
                    mom_msg = "Overbought"
                    mom_icon = "⚠️"
                    mom_expl = f"RSI is {current_rsi:.1f} (>70), suggesting the stock may be due for a pullback."
                elif current_rsi < 30:
                    mom_msg = "Oversold"
                    mom_icon = "♻️"
                    mom_expl = f"RSI is {current_rsi:.1f} (<30), suggesting the stock may be undervalued/due for a bounce."
                else:
                    mom_msg = "Stable"
                    mom_icon = "✅"
                    mom_expl = f"RSI is {current_rsi:.1f}, sitting comfortably in the neutral zone (30-70)."
                    
                if macd_val > signal_val:
                    macd_msg = "Positive Divergence"
                    macd_icon = "🟢"
                    macd_expl = f"The MACD line ({macd_val:.2f}) is above the Signal line ({signal_val:.2f}), indicating bullish momentum."
                else:
                    macd_msg = "Negative Divergence"
                    macd_icon = "🔴"
                    macd_expl = f"The MACD line ({macd_val:.2f}) is below the Signal line ({signal_val:.2f}), indicating bearish momentum."

                st.markdown(f"**Trend:** {trend_icon} {trend_msg}")
                st.caption(trend_expl)
                
                st.markdown(f"**Momentum:** {mom_icon} {mom_msg}")
                st.caption(mom_expl)
                
                st.markdown(f"**MACD:** {macd_icon} {macd_msg}")
                st.caption(macd_expl)
                
                st.markdown("---")
                
                high_52 = history_full['High'].tail(252).max()
                low_52 = history_full['Low'].tail(252).min()
                st.metric("52-Week High", f"R {high_52:.2f}")
                st.metric("52-Week Low", f"R {low_52:.2f}")

        with tab_macro:
            st.subheader("🔗 Macro-Economic Correlations")
            with st.spinner("Analyzing macro data..."):
                macro_df = get_macro_data(period=display_period)
            
            if not macro_df.empty and 'Invicta' in macro_df.columns:
                c_charts, c_stats = st.columns([3, 1])
                with c_charts:
                    st.markdown("#### 🌍 Relative Performance Comparison")
                    norm_df = (macro_df / macro_df.iloc[0]) * 100 - 100
                    fig_macro = go.Figure()
                    colors = {'Invicta': '#1f77b4', 'USD/ZAR': 'orange', 'JSE Industrials': 'gray'}
                    for col in norm_df.columns:
                        width = 3 if col == 'Invicta' else 1.5
                        dash = 'solid' if col == 'Invicta' else 'dot'
                        fig_macro.add_trace(go.Scatter(
                            x=norm_df.index,
                            y=norm_df[col],
                            name=col,
                            line=dict(color=colors.get(col, 'black'), width=width, dash=dash)
                        ))
                    fig_macro.update_layout(height=400, yaxis_title="Performance (%)", hovermode="x unified")
                    st.plotly_chart(fig_macro, use_container_width=True)
                with c_stats:
                    st.markdown("#### 🔢 Correlation")
                    corr_matrix = macro_df.pct_change().corr()['Invicta'].drop('Invicta')
                    st.dataframe(corr_matrix, use_container_width=True)

        with tab_fin:
            st.subheader("📊 Financial Health Comparison")
            
            st.markdown("### 🗓️ Annual Results (Full Financial Year)")
            if not financials.empty and financials.shape[1] >= 2:
                latest_date = financials.columns[0]
                prev_date = financials.columns[1]
                
                metrics_list = ['Total Revenue', 'Gross Profit', 'EBITDA', 'Net Income', 'Basic EPS']
                fin_data = []
                for m in metrics_list:
                    val_curr = safe_get_financial_value(financials, m, 0)
                    val_prev = safe_get_financial_value(financials, m, 1)
                    if val_curr != 0:
                        change = calculate_change(val_curr, val_prev)
                        fin_data.append({
                            "Metric": m,
                            f"{latest_date.strftime('%Y')} (Latest)": format_large_number(val_curr),
                            f"{prev_date.strftime('%Y')} (Prior)": format_large_number(val_prev),
                            "Change (%)": f"{change:+.2f}%"
                        })
                st.dataframe(pd.DataFrame(fin_data), use_container_width=True, hide_index=True)
            else:
                st.warning("Insufficient Annual Data available.")

            st.divider()

            sept_col = None
            if not quarterly_fin.empty:
                for col in quarterly_fin.columns:
                    if isinstance(col, pd.Timestamp) and col.month == 9:
                        sept_col = col
                        break

            if sept_col:
                st.markdown("### ⏱️ Year to Date / Interim (September Trend)")
                st.caption(f"Showing Interim Data for: {sept_col.strftime('%B %Y')}")
                q_metrics = ['Total Revenue', 'Net Income', 'Operating Income']
                q_data = []
                for m in q_metrics:
                    q_curr = quarterly_fin.loc[m, sept_col] if m in quarterly_fin.index else 0
                    
                    prev_sept_col = None
                    for col in quarterly_fin.columns:
                        if isinstance(col, pd.Timestamp) and col.month == 9 and col.year == (sept_col.year - 1):
                            prev_sept_col = col
                            break
                    
                    if prev_sept_col:
                        q_last = quarterly_fin.loc[m, prev_sept_col] if m in quarterly_fin.index else 0
                        compare_label = f"Sept {prev_sept_col.year}"
                    else:
                        q_last = 0 
                        compare_label = "Prior Period"

                    if q_curr != 0:
                        q_chg = calculate_change(q_curr, q_last)
                        q_data.append({
                            "Metric": m,
                            "Period": sept_col.strftime('%b %Y'),
                            "Current": format_large_number(q_curr),
                            compare_label: format_large_number(q_last),
                            "Change (%)": f"{q_chg:+.2f}%"
                        })
                st.dataframe(pd.DataFrame(q_data), use_container_width=True, hide_index=True)
            
            st.divider()
            
            c_r1, c_r2 = st.columns(2)
            
            with c_r1:
                st.markdown("### 📉 Ratio Analysis: Year-over-Year")
                if not financials.empty and financials.shape[1] >= 2:
                    
                    def calc_ratio(df, num, den, col):
                        try:
                            n = df.loc[num].iloc[col]
                            d = df.loc[den].iloc[col]
                            return (n / d) * 100 if d != 0 else 0
                        except:
                            return 0

                    ratios_data = []
                    
                    gm_curr = calc_ratio(financials, 'Gross Profit', 'Total Revenue', 0)
                    gm_prev = calc_ratio(financials, 'Gross Profit', 'Total Revenue', 1)
                    ratios_data.append({"Ratio": "Gross Margin", "Current (%)": f"{gm_curr:.2f}%", "Prior (%)": f"{gm_prev:.2f}%"})
                    
                    om_curr = calc_ratio(financials, 'Operating Income', 'Total Revenue', 0)
                    om_prev = calc_ratio(financials, 'Operating Income', 'Total Revenue', 1)
                    ratios_data.append({"Ratio": "Operating Margin", "Current (%)": f"{om_curr:.2f}%", "Prior (%)": f"{om_prev:.2f}%"})

                    nm_curr = calc_ratio(financials, 'Net Income', 'Total Revenue', 0)
                    nm_prev = calc_ratio(financials, 'Net Income', 'Total Revenue', 1)
                    ratios_data.append({"Ratio": "Net Margin", "Current (%)": f"{nm_curr:.2f}%", "Prior (%)": f"{nm_prev:.2f}%"})
                    
                    st.dataframe(pd.DataFrame(ratios_data), use_container_width=True, hide_index=True)
                else:
                    st.caption("Insufficient data for ratio analysis.")

            with c_r2:
                st.markdown("### 🏆 Peer Benchmarking: Efficiency")
                with st.spinner("Compare with peers..."):
                    comp_metrics, _ = get_competitor_financials()
                
                if not comp_metrics.empty:
                    cols_to_show = ["Company", "Gross Margin (%)", "Op Margin (%)", "ROE (%)"]
                    valid_cols = [c for c in cols_to_show if c in comp_metrics.columns]
                    
                    comp_metrics_style = comp_metrics[valid_cols].copy()
                    for c in valid_cols[1:]:
                        comp_metrics_style[c] = comp_metrics_style[c].apply(lambda x: f"{x:.2f}%")
                        
                    st.dataframe(comp_metrics_style, use_container_width=True, hide_index=True)

            if not financials.empty:
                fin_T = financials.T.iloc[:4][::-1]
                try:
                    formatted_dates = [d.strftime('%b %Y') if isinstance(d, pd.Timestamp) else str(d) for d in fin_T.index]
                except:
                    formatted_dates = fin_T.index.astype(str)

                fig_fin = go.Figure()
                if 'Total Revenue' in fin_T.columns:
                    fig_fin.add_trace(go.Bar(x=formatted_dates, y=fin_T['Total Revenue'], name='Revenue', marker_color='#1f77b4'))
                if 'Net Income' in fin_T.columns:
                    fig_fin.add_trace(go.Bar(x=formatted_dates, y=fin_T['Net Income'], name='Net Income', marker_color='#2ca02c'))
                
                fig_fin.update_layout(height=350, title="Annual Trend (Year End March)")
                st.plotly_chart(fig_fin, use_container_width=True)

        with tab_comp:
            st.markdown("### ⚔️ Peer Comparison")
            with st.spinner("Analyzing Peers..."):
                comp_metrics, comp_history = get_competitor_financials()
            
            if not comp_metrics.empty:
                col_c1, col_c2 = st.columns([3, 2])
                with col_c1:
                    fig_rel = go.Figure()
                    colors = {
                        "Invicta (IVT)": "#1f77b4",
                        "Hudaco (HDC)": "#d62728",
                        "Barloworld (BAW)": "#2ca02c",
                        "Bell Equipment (BEL)": "#9467bd",
                        "Master Drilling (MDI)": "#8c564b",
                        "enX Group (ENX)": "#7f7f7f"
                    }
                    for col in comp_history.columns:
                        width = 4 if "Invicta" in col else 2
                        fig_rel.add_trace(go.Scatter(
                            x=comp_history.index,
                            y=comp_history[col],
                            mode='lines',
                            name=col,
                            line=dict(width=width, color=colors.get(col, "gray"))
                        ))
                    fig_rel.update_layout(height=350, margin=dict(l=0, r=0, t=30, b=0), yaxis_title="Growth %")
                    st.plotly_chart(fig_rel, use_container_width=True)
                with col_c2:
                    styled_df = comp_metrics.copy()
                    styled_df['Price'] = styled_df['Price'].apply(lambda x: f"R {x:.2f}")
                    styled_df['P/E Ratio'] = styled_df['P/E Ratio'].apply(lambda x: f"{x:.2f}")
                    styled_df['Div Yield (%)'] = styled_df['Div Yield (%)'].apply(lambda x: f"{x:.2f}%")
                    styled_df['Market Cap (B)'] = styled_df['Market Cap (B)'].apply(lambda x: f"R {x:.2f} B")
                    st.dataframe(
                        styled_df[['Company', 'Price', 'P/E Ratio', 'Div Yield (%)', 'Market Cap (B)']],
                        hide_index=True,
                        use_container_width=True
                    )

        with tab_sent:
            with st.spinner("AI is reading the news..."):
                ivt_score, ivt_news = fetch_news_score("Invicta Holdings", article_limit=6)
                hdc_score, _ = fetch_news_score("Hudaco Industries", article_limit=3)
                baw_score, _ = fetch_news_score("Barloworld", article_limit=3)
                bell_score, _ = fetch_news_score("Bell Equipment", article_limit=3)
                mdi_score, _ = fetch_news_score("Master Drilling", article_limit=3)
                enx_score, _ = fetch_news_score("enX Group", article_limit=3)

            if ivt_news:
                trend_df = pd.DataFrame(ivt_news)
                trend_df['DateParsed'] = pd.to_datetime(
                    trend_df['date'],
                    format="%d %b %Y",
                    errors='coerce'
                )
                trend_df = trend_df.dropna(subset=['DateParsed'])
                if not trend_df.empty:
                    daily = (
                        trend_df.groupby('DateParsed')['Score']
                        .mean()
                        .reset_index()
                        .sort_values('DateParsed')
                    )
                else:
                    daily = pd.DataFrame()
            else:
                daily = pd.DataFrame()

            s1, s2 = st.columns([1.2, 1.8])

            with s1:
                st.markdown("#### IVT Sentiment Trend")
                if not daily.empty:
                    fig_trend = go.Figure(
                        go.Scatter(
                            x=daily['DateParsed'],
                            y=daily['Score'],
                            mode='lines+markers',
                            name='IVT Sentiment'
                        )
                    )
                    fig_trend.update_layout(
                        height=220,
                        margin=dict(l=0, r=0, t=10, b=0),
                        yaxis=dict(title="Score (-1 to +1)", range=[-1, 1])
                    )
                    st.plotly_chart(fig_trend, use_container_width=True)
                else:
                    st.caption("Not enough dated news to plot a sentiment trend yet.")

                st.markdown("#### Sentiment Battle")
                comp_data = {
                    'Company': [
                        'Invicta',
                        'Hudaco',
                        'Barloworld',
                        'Bell Equipment',
                        'Master Drilling',
                        'enX Group'
                    ],
                    'Score': [
                        ivt_score,
                        hdc_score,
                        baw_score,
                        bell_score,
                        mdi_score,
                        enx_score
                    ]
                }
                df_comp = pd.DataFrame(comp_data)
                fig_comp = go.Figure(
                    go.Bar(
                        x=df_comp['Score'],
                        y=df_comp['Company'],
                        orientation='h'
                    )
                )
                fig_comp.update_layout(
                    height=260,
                    margin=dict(l=0, r=0, t=10, b=0),
                    xaxis=dict(range=[-1.0, 1.0], title="Score (-1 to +1)")
                )
                st.plotly_chart(fig_comp, use_container_width=True)

            with s2:
                st.markdown("#### Invicta News Feed")
                if ivt_news:
                    for item in ivt_news:
                        with st.expander(f"{item['Icon']} {item['title']}"):
                            c_t, c_i = st.columns([3, 1])
                            with c_t:
                                st.caption(f"{item['source']} | {item['date']} | {item['Method']}")
                                
                                # 👇 Show what the AI actually read
                                st.write(item['snippet'])
                                
                                st.markdown("---")
                                st.write(item['Explanation'])
                                st.markdown(f"[Read Article]({item['link']})")
                            with c_i:
                                st.progress((item['Score'] + 1) / 2)
                                st.caption(f"Impact: {item['Score']:.2f}")
                else:
                    st.caption("No recent Invicta news articles found.")

if __name__ == "__main__":
    main()
