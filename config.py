"""Central config. Edit tickers, feeds, hosting details here."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DOCS = ROOT / "docs"
EPISODES_DIR = DOCS / "episodes"
FEED_PATH = DOCS / "feed.xml"
STATE_PATH = ROOT / ".state.json"

# Market data
INDICES = {
    "^GSPC": "S&P 500",
    "^DJI": "Dow Jones",
    "^IXIC": "Nasdaq",
    "^RUT": "Russell 2000",
    "^VIX": "VIX",
}
SECTOR_ETFS = {
    "XLK": "Tech",
    "XLF": "Financials",
    "XLE": "Energy",
    "XLV": "Healthcare",
    "XLY": "Consumer Disc.",
    "XLP": "Consumer Staples",
    "XLI": "Industrials",
    "XLU": "Utilities",
    "XLRE": "Real Estate",
    "XLB": "Materials",
    "XLC": "Communication",
}
MACRO = {
    "^TNX": "10Y Treasury",
    "DX-Y.NYB": "Dollar Index",
    "CL=F": "WTI Crude",
    "GC=F": "Gold",
    "BTC-USD": "Bitcoin",
}
MOVERS_UNIVERSE = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA", "AVGO",
    "BRK-B", "LLY", "JPM", "V", "UNH", "XOM", "WMT", "MA", "PG", "JNJ",
    "HD", "COST", "NFLX", "BAC", "CRM", "ORCL", "AMD", "ADBE", "PEP",
    "TMO", "CVX", "ABBV", "KO", "MRK", "CSCO", "ACN", "MCD", "DIS",
]

# News RSS (free, no key)
RSS_FEEDS = [
    "https://finance.yahoo.com/news/rssindex",
    "https://www.marketwatch.com/rss/topstories",
    "https://feeds.content.dowjones.io/public/rss/mw_topstories",
    "https://seekingalpha.com/market_currents.xml",
    "https://www.cnbc.com/id/100003114/device/rss/rss.html",  # top news
    "https://www.cnbc.com/id/10001147/device/rss/rss.html",   # markets
    "https://www.federalreserve.gov/feeds/press_all.xml",
]
HEADLINE_LIMIT = 40  # cap total headlines sent to LLM

# LLM (Ollama local)
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen2.5:14b"
OLLAMA_TIMEOUT = 600

# Script length: flexible. LLM picks based on news density.
MIN_WORDS = 400   # floor ~2.5 min
MAX_WORDS = 1800  # ceiling ~12 min

# TTS
TTS_VOICE = "Samantha"
TTS_RATE = 185  # words per minute

# Podcast metadata (edit these)
PODCAST_TITLE = "Market Today, Explained"
PODCAST_AUTHOR = "Saksham Gupta"
PODCAST_EMAIL = "gsaksham@gmail.com"
PODCAST_DESCRIPTION = "Daily AI-generated recap of US markets with the news that moved them."
PODCAST_LANGUAGE = "en-us"
# Set after GitHub Pages is live. Example: https://<user>.github.io/<repo>
PODCAST_BASE_URL = "https://sakzgupzzz.github.io/explain-market-today"
PODCAST_CATEGORY = "Business"
PODCAST_SUBCATEGORY = "Investing"
