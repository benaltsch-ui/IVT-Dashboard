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
from datetime import datetime, timezone
from typing import Dict, List, Tuple

# --- CONFIGURATION ---
st.set_page_config(
    page_title="Invicta Holdings Live Dashboard",
    layout="wide",
    page_icon="🏭",
)

# --- CACHING & RESOURCES ---

@st.cache_resource(show_spinner=False)
def get_vader_analyzer() -> SentimentIntensityAnalyzer:
    """Load VADER model and extend it with a financial lexicon."""
    nltk.download("vader_lexicon", quiet=True)
    analyzer = SentimentIntensityAnalyzer()

    # --- FINANCIAL LEXICON EXTENSION ---
    finance_lexicon = {
        "beat": 2.0,
        "beats": 2.0,
        "miss": -2.0,
        "missed": -2.0,
        "downgrade": -2.5,
        "downgraded": -2.5,
        "upgrade": 2.5,
        "upgraded": 2.5,
        "strong": 1.5,
        "weak": -1.5,
        "plunge": -2.5,
        "plunges": -2.5,
        "slump": -2.0,
        "slumps": -2.0,
        "soar": 3.0,
        "soars": 3.0,
        "surge": 2.5,
        "surges": 2.5,
        "profit": 2.0,
        "loss": -2.0,
        "liquidity": 1.5,
        "default": -3.0,
        "bankrupt": -3.5,
        "bankruptcy": -3.5,
    }
    analyzer.lexicon.update(finance_lexicon)
    return analyzer


@st.cache_data(show_spinner=False)
def get_market_data(ticker: str, period: str = "1y") -> Tuple[pd.DataFrame, Dict]:
    """Fetch historical data and basic info for a ticker."""
    ticker_obj = yf.Ticker(ticker)
    hist = ticker_obj.history(period=period)
    info = ticker_obj.info or {}
    return hist, info


def compute_price_change(hist: pd.DataFrame) -> Tuple[float, float]:
    """
    Compute absolute and percentage price change using last two closes.
    Returns (abs_change, pct_change). If insufficient data, returns (0.0, 0.0).
    """
    if hist is None or hist.empty or len(hist["Close"]) < 2:
        return 0.0, 0.0
    last = float(hist["Close"].iloc[-1])
    prev = float(hist["Close"].iloc[-2])
    abs_change = last - prev
    pct_change = (abs_change / prev) * 100 if prev != 0 else 0.0
    return abs_change, pct_change


@st.cache_data(show_spinner=False)
def get_competitor_financials(tickers: Dict[str, str]) -> pd.DataFrame:
    """
    Build a DataFrame with basic valuation metrics for a dict of {name: ticker}.
    """
    metrics: List[Dict] = []
    for name, symbol in tickers.items():
        try:
            stock = yf.Ticker(symbol)
            info = stock.info or {}
            pe = info.get("trailingPE")
            pb = info.get("priceToBook")
            roic = info.get("returnOnEquity")
            mkt_cap = info.get("marketCap")

            metrics.append(
                {
                    "Company": name,
                    "Ticker": symbol,
                    "P/E Ratio": pe,
                    "P/B Ratio": pb,
                    "ROE (proxy for ROIC)": roic,
                    "Market Cap": mkt_cap,
                }
            )
        except Exception:
            # Skip if we can't fetch fundamentals for this ticker
            continue

    if not metrics:
        return pd.DataFrame()
    return pd.DataFrame(metrics)


def get_final_url(url: str, timeout: int = 8) -> str:
    """Follow redirects to get final URL."""
    try:
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
        return resp.url
    except Exception:
        return url


def get_article_content(url: str, timeout: int = 8) -> str:
    """Extract main article text using trafilatura."""
    try:
        final_url = get_final_url(url, timeout=timeout)
        downloaded = trafilatura.fetch_url(final_url)
        if not downloaded:
            return ""
        extracted = trafilatura.extract(downloaded, include_comments=False, include_tables=False)
        return extracted or ""
    except Exception:
        return ""


def analyze_content(text: str) -> Tuple[float, Dict[str, float]]:
    """
    Analyze sentiment of a text.
    Returns overall compound score and a dict of key financial driver terms.
    """
    if not text:
        return 0.0, {}

    analyzer = get_vader_analyzer()
    scores = analyzer.polarity_scores(text)
    compound = scores["compound"]

    # Identify key drivers from the extended lexicon that actually appear in the text
    drivers: Dict[str, float] = {}
    lowered = text.lower()
    for word, val in analyzer.lexicon.items():
        if word in lowered and abs(val) >= 1.0:
            drivers[word] = val

    return compound, drivers


@st.cache_data(show_spinner=False)
def fetch_news_score(
    query: str,
    article_limit: int = 7,
) -> Tuple[float, List[Dict]]:
    """
    Fetch news for a query using Google News RSS, analyze sentiment,
    and return (weighted_score, articles_with_metadata).

    Each article dict contains:
    - title
    - link
    - published
    - sentiment
    """
    base_url = "https://news.google.com/rss/search"
    params = {"q": query, "hl": "en-ZA", "gl": "ZA", "ceid": "ZA:en"}
    try:
        resp = requests.get(base_url, params=params, timeout=8)
        resp.raise_for_status()
        feed = feedparser.parse(resp.text)
    except Exception:
        return 0.0, []

    articles: List[Dict] = []
    total_weight = 0.0
    weighted_sum = 0.0
    now = datetime.now(timezone.utc)

    for entry in feed.entries[:article_limit]:
        link = entry.get("link")
        if not link:
            continue

        content = get_article_content(link)
        if not content:
            continue

        score, _ = analyze_content(content)

        # Recency weighting: newer articles get higher weight
        published = entry.get("published")
        if published:
            try:
                pub_dt = parsedate_to_datetime(published)
            except Exception:
                pub_dt = now
        else:
            pub_dt = now

        days_old = max((now - pub_dt).days, 0)
        # Exponential decay: 1.0 today, decreases with age
        weight = 1.0 / (1.0 + 0.1 * days_old)

        weighted_sum += score * weight
        total_weight += weight

        articles.append(
            {
                "title": entry.get("title", "No title"),
                "link": link,
                "published": pub_dt,
                "sentiment": score,
            }
        )

    if not articles or total_weight == 0:
        return 0.0, []

    overall = weighted_sum / total_weight
    return overall, articles


def format_pct(p: float) -> str:
    return f"{p:+.2f}%"


def sentiment_label(score: float) -> str:
    if score >= 0.25:
        return "Positive"
    if score <= -0.25:
        return "Negative"
    return "Neutral"


# --- SIDEBAR CONTROLS ---

with st.sidebar:
    st.header("Dashboard Settings")

    main_ticker = st.text_input("Primary ticker", value="IVT.JO", help="Yahoo Finance ticker symbol")
    period = st.selectbox("Price history period", ["3mo", "6mo", "1y", "5y"], index=2)

    st.markdown("---")

    default_peers = {
        "Invicta (IVT)": "IVT.JO",
        "Hudaco (HDC)": "HDC.JO",
        "Barloworld (BAW)": "BAW.JO",
    }
    peer_symbols = list(default_peers.values())
    peer_labels = list(default_peers.keys())

    selected_peers = st.multiselect(
        "Peers for comparison",
        options=peer_symbols,
        default=peer_symbols,
        format_func=lambda x: peer_labels[peer_symbols.index(x)],
        help="Choose which tickers to include in competitor analysis",
    )

    news_article_limit = st.slider("News articles per company", 3, 15, 7, 1)

    sentiment_mode = st.radio(
        "Sentiment comparison universe",
        ["Invicta only", "All selected peers"],
        index=1,
    )


# --- MAIN LAYOUT ---

st.title("🏭 Invicta Holdings Live Dashboard")

# Fetch main ticker data
main_hist, main_info = get_market_data(main_ticker, period=period)

if main_hist.empty:
    st.error("Could not load market data for the selected ticker. Please check the symbol and try again.")
    st.stop()

last_price = float(main_hist["Close"].iloc[-1])
abs_chg, pct_chg = compute_price_change(main_hist)

col_kpi1, col_kpi2, col_kpi3 = st.columns(3)
with col_kpi1:
    st.metric("Last Price", f"{last_price:.2f}", format_pct(pct_chg))
with col_kpi2:
    mkt_cap = main_info.get("marketCap")
    st.metric("Market Cap", f"{mkt_cap:,.0f}" if mkt_cap else "N/A")
with col_kpi3:
    sector = main_info.get("sector", "N/A")
    st.metric("Sector", sector)

st.caption(
    "Price performance and market data sourced from Yahoo Finance. "
    "News sentiment is based on recent media coverage."
)

# --- SECTION 1: PRICE HISTORY & MARKET OVERVIEW ---

st.subheader("Market Performance")

fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=main_hist.index,
        y=main_hist["Close"],
        mode="lines",
        name=main_ticker,
    )
)
fig.update_layout(
    xaxis_title="Date",
    yaxis_title="Price",
    hovermode="x unified",
    height=400,
)
st.plotly_chart(fig, use_container_width=True)

# --- SECTION 2: COMPETITOR VALUATION SNAPSHOT ---

st.subheader("Financial Comparison vs. Competitors")

selected_peers_dict = {k: v for k, v in default_peers.items() if v in selected_peers}

if not selected_peers_dict:
    st.info("Select at least one peer in the sidebar to see competitor comparisons.")
else:
    comp_df = get_competitor_financials(selected_peers_dict)
    if comp_df.empty:
        st.warning("Could not load competitor fundamentals.")
    else:
        st.dataframe(comp_df.set_index("Company"))

        # Simple P/E vs. P/B scatter if data is available
        scatter_df = comp_df.dropna(subset=["P/E Ratio", "P/B Ratio"])
        if not scatter_df.empty:
            scat_fig = go.Figure()
            scat_fig.add_trace(
                go.Scatter(
                    x=scatter_df["P/B Ratio"],
                    y=scatter_df["P/E Ratio"],
                    mode="markers+text",
                    text=scatter_df["Company"],
                    textposition="top center",
                )
            )
            scat_fig.update_layout(
                xaxis_title="P/B Ratio",
                yaxis_title="P/E Ratio",
                title="Valuation Positioning",
                height=450,
            )
            st.plotly_chart(scat_fig, use_container_width=True)

# --- SECTION 3: NEWS & SENTIMENT FOR INVICTA ---

st.subheader("News & Sentiment – Invicta")

ivt_query = "Invicta Holdings South Africa"
ivt_score, ivt_articles = fetch_news_score(ivt_query, article_limit=news_article_limit)

col_sent1, col_sent2 = st.columns([1, 2])
with col_sent1:
    st.metric(
        "Media Sentiment (Invicta)",
        f"{ivt_score:+.3f}",
        sentiment_label(ivt_score),
    )
with col_sent2:
    if ivt_articles:
        pos = sum(1 for a in ivt_articles if a["sentiment"] >= 0.25)
        neg = sum(1 for a in ivt_articles if a["sentiment"] <= -0.25)
        neu = len(ivt_articles) - pos - neg
        st.write(f"Based on {len(ivt_articles)} recent articles: {pos} positive, {neu} neutral, {neg} negative.")
    else:
        st.write("No sufficiently long or parsable recent articles found for Invicta.")

if ivt_articles:
    with st.expander("View analysed Invicta articles"):
        for art in ivt_articles:
            st.markdown(f"**[{art['title']}]({art['link']})**")
            st.write(
                f"Published: {art['published'].strftime('%Y-%m-%d')} · "
                f"Sentiment: {art['sentiment']:+.3f} ({sentiment_label(art['sentiment'])})"
            )
            st.markdown("---")

# --- SECTION 4: SENTIMENT BATTLE – PEERS VS INVICTA ---

st.subheader("Sentiment Battle – Invicta vs Peers")

sentiment_universe: Dict[str, Tuple[float, List[Dict]]] = {}

# Always include Invicta by name
sentiment_universe["Invicta Holdings"] = (ivt_score, ivt_articles)

if sentiment_mode == "All selected peers":
    for name, symbol in selected_peers_dict.items():
        # Skip Invicta – already added
        if "Invicta" in name:
            continue
        peer_query = name.split("(")[0].strip() + " South Africa"
        score, arts = fetch_news_score(peer_query, article_limit=news_article_limit)
        sentiment_universe[name] = (score, arts)

if not sentiment_universe:
    st.info("No companies available to compare sentiment.")
else:
    comp_names = list(sentiment_universe.keys())
    scores = [sentiment_universe[n][0] for n in comp_names]

    bar_fig = go.Figure()
    bar_fig.add_trace(
        go.Bar(
            x=comp_names,
            y=scores,
            text=[f"{s:+.2f}" for s in scores],
            textposition="auto",
        )
    )
    bar_fig.update_layout(
        yaxis_title="Sentiment score (VADER compound, -1 to 1)",
        yaxis=dict(range=[-1, 1]),
        height=450,
    )
    st.plotly_chart(bar_fig, use_container_width=True)

    with st.expander("Articles by company"):
        for name in comp_names:
            score, arts = sentiment_universe[name]
            st.markdown(f"### {name} ({sentiment_label(score)}) – {score:+.3f}")
            if not arts:
                st.write("No sufficiently long or parsable recent articles.")
            else:
                for art in arts:
                    st.markdown(
                        f"- [{art['title']}]({art['link']}) · "
                        f"{art['published'].strftime('%Y-%m-%d')} · "
                        f"{art['sentiment']:+.3f} ({sentiment_label(art['sentiment'])})"
                    )
            st.markdown("---")


st.caption("This dashboard is for informational purposes only and does not constitute investment advice.")
