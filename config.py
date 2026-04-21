"""Central config. Edit tickers, feeds, hosting details here."""
import os
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

# News RSS grouped by beat. Morning-Brew style: markets core + tech + business + world + culture.
RSS_FEEDS_BY_CATEGORY: dict[str, list[str]] = {
    "markets": [
        "https://finance.yahoo.com/news/rssindex",
        "https://www.marketwatch.com/rss/topstories",
        "https://feeds.content.dowjones.io/public/rss/mw_topstories",
        "https://seekingalpha.com/market_currents.xml",
        "https://www.cnbc.com/id/10001147/device/rss/rss.html",   # markets
        "https://feeds.reuters.com/reuters/businessNews",
        "https://www.federalreserve.gov/feeds/press_all.xml",
    ],
    "business": [
        "https://www.cnbc.com/id/100003114/device/rss/rss.html",
        "https://feeds.reuters.com/reuters/companyNews",
        "https://feeds.apnews.com/ApBusiness",
        "https://api.axios.com/feed/",
    ],
    "tech": [
        "https://techcrunch.com/feed/",
        "https://www.theverge.com/rss/index.xml",
        "https://feeds.arstechnica.com/arstechnica/index",
        "https://hnrss.org/frontpage?points=300",
    ],
    "world": [
        "https://feeds.reuters.com/reuters/topNews",
        "https://feeds.apnews.com/ApTopHeadlines",
        "https://feeds.bbci.co.uk/news/rss.xml",
    ],
    "culture": [
        "https://www.theatlantic.com/feed/all/",
        "https://feeds.npr.org/1001/rss.xml",
    ],
}
HEADLINES_PER_CATEGORY = {
    "markets": 20,
    "business": 10,
    "tech": 10,
    "world": 8,
    "culture": 5,
}

# LLM (Ollama local or Actions runner). Override via env.
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/generate")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:14b")
OLLAMA_TIMEOUT = int(os.environ.get("OLLAMA_TIMEOUT", "1800"))

# Script length: flexible. LLM picks based on news density.
MIN_WORDS = 400   # floor ~2.5 min
MAX_WORDS = 1800  # ceiling ~12 min

# TTS — macOS `say` on Darwin, Piper on Linux
TTS_VOICE = os.environ.get("TTS_VOICE", "Samantha")
TTS_RATE = int(os.environ.get("TTS_RATE", "185"))
PIPER_VOICE_PATH = os.environ.get(
    "PIPER_VOICE_PATH",
    str(Path.home() / ".local/share/piper-voices/en_US-libritts_r-medium.onnx"),
)
# Multi-speaker cast. Speaker IDs index into the libritts_r model (900+ speakers).
# Spread across the range so voices are distinct.
CHARACTERS = {
    "JAMIE": {"speaker": 79,  "description": "main host — upbeat, drives the show, asks sharp questions, reacts with personality"},
    "ALEX":  {"speaker": 13,  "description": "markets analyst — dry, precise, explains the WHY behind moves, deadpan humor"},
    "MAYA":  {"speaker": 411, "description": "tech correspondent — fast-talker, hype-aware but skeptical, loves a good product-launch story"},
    "RIO":   {"speaker": 218, "description": "world/culture correspondent — warm, storyteller voice, brings human texture to big stories"},
    "KAI":   {"speaker": 635, "description": "quick-hits floater — punchy one-liners, delivers the odd-thing closer, trivia energy"},
}
DEFAULT_CHARACTER = "JAMIE"
INTER_LINE_SILENCE_MS = 160  # natural breath between speaker swaps

# Podcast metadata (edit these)
PODCAST_TITLE = "Market Today, Explained"
PODCAST_AUTHOR = "Saksham Gupta"
PODCAST_EMAIL = "gsaksham@gmail.com"
PODCAST_DESCRIPTION = "Daily fast, funny roundtable on US markets, business, tech, and one weird thing — AI-generated each afternoon."
PODCAST_LANGUAGE = "en-us"
# Set after GitHub Pages is live. Example: https://<user>.github.io/<repo>
PODCAST_BASE_URL = "https://sakzgupzzz.github.io/explain-market-today"
PODCAST_CATEGORY = "Business"
PODCAST_SUBCATEGORY = "Investing"
