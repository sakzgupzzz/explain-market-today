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
    "markets": 30,
    "business": 18,
    "tech": 18,
    "world": 12,
    "culture": 8,
}

# LLM (Ollama local or Actions runner). Override via env.
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/generate")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen3:14b")
OLLAMA_CRITIC_MODEL = os.environ.get("OLLAMA_CRITIC_MODEL", OLLAMA_MODEL)

# Groq — fast hosted inference of open-weight models. Free tier covers
# daily-podcast workload comfortably (30k tokens/min limit, we burn ~20k/run).
# When GROQ_API_KEY is set, generate_script._llm_call uses Groq instead of
# Ollama. Keeps Ollama as the local default.
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_URL = os.environ.get("GROQ_URL", "https://api.groq.com/openai/v1/chat/completions")
GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
# Critique pass needs a 32k-context model: prompt = system + market data +
# all headlines + full draft script, which can exceed the 8k window on
# llama-3.1-8b-instant (caused HTTP 413 on 53-headline days). 70b-versatile
# has 32k context. We also cap headlines-per-beat in the critique prompt as
# belt-and-suspenders (see CRITIQUE_HEADLINES_PER_BEAT in generate_script).
GROQ_CRITIC_MODEL = os.environ.get("GROQ_CRITIC_MODEL", "llama-3.3-70b-versatile")
OLLAMA_TIMEOUT = int(os.environ.get("OLLAMA_TIMEOUT", "1800"))

# Script length: flexible. LLM picks based on news density.
# Bumped from 400/1800 → 800/2200 after early v3-pipeline episodes felt
# under-baked (16 turns ~60 words each). Higher floor + explicit minimum
# turn count in the prompt gets the conversational density back.
MIN_WORDS = 1000
MAX_WORDS = 2700
# Hard floor for retry trigger. Anything under this regenerates with a
# stronger 'more turns' prompt. 30 leaves slack — first-pass scripts at
# 32+ turns ship without retry; only the truly thin ones regenerate.
MIN_TURNS = 30

# ElevenLabs v3 default delivery is podcast-narration paced — about 15-20%
# slower than what old Piper/macOS-say episodes felt like. Post-process
# the mastered audio with ffmpeg atempo for a tighter feel without
# pitch-shifting. 1.08 = 8% faster. Tune in [1.0, 1.20]; >1.15 starts
# to sound rushed.
AUDIO_SPEEDUP = float(os.environ.get("AUDIO_SPEEDUP", "1.10"))

# TTS — macOS `say` on Darwin, Piper on Linux
TTS_VOICE = os.environ.get("TTS_VOICE", "Samantha")
TTS_RATE = int(os.environ.get("TTS_RATE", "185"))
PIPER_VOICE_PATH = os.environ.get(
    "PIPER_VOICE_PATH",
    str(Path.home() / ".local/share/piper-voices/en_US-libritts_r-medium.onnx"),
)
# Multi-speaker cast. Speaker IDs index into the libritts_r model (900+ speakers).
# Spread across the range so voices are distinct.
# Three hosts — listener feedback was that 8 voices is hard to track. Trimmed
# to a host + 2 specialists who together cover markets/business/macro and
# tech/culture/odd-thing. Listeners can tell who's talking from voice alone
# instead of having to remember an 8-person org chart.
CHARACTERS = {
    "JAMIE": {
        "speaker": 79,
        "description": "main host — drives the show, asks sharp questions, reacts with personality, bookends every episode",
        "tags": ["[curious]", "[excited]", "[laughs]"],
    },
    "ALEX": {
        "speaker": 13,
        "description": "markets + business + macro desk — dry, precise, explains the WHY behind moves; covers equities, gainers/losers, rates, the Fed, the dollar, big corporate stories; deadpan humor",
        "tags": ["[deadpan]", "[sarcastic]", "[sighs]"],
    },
    "MAYA": {
        "speaker": 411,
        "description": "tech + culture + odd-thing desk — fast-talker, hype-aware but skeptical; covers product launches, AI/crypto, world stories with human angles, and the absurdist closer",
        "tags": ["[rushed]", "[excited]", "[mischievously]", "[laughs]"],
    },
}
DEFAULT_CHARACTER = "JAMIE"
INTER_LINE_SILENCE_MS = 160  # natural breath between speaker swaps

# TTS backend selection. "eleven" | "mac" | "piper" | "auto" (default).
# auto = eleven if ELEVENLABS_API_KEY set, else mac on Darwin, else piper.
TTS_BACKEND = os.environ.get("TTS_BACKEND", "auto").lower()

# ElevenLabs config. Creator plan = 100k chars/mo (~10 hours of audio).
ELEVENLABS_API_KEY = os.environ.get("ELEVENLABS_API_KEY", "")
ELEVENLABS_MODEL = os.environ.get("ELEVENLABS_MODEL", "eleven_multilingual_v2")
ELEVENLABS_OUTPUT_FORMAT = os.environ.get("ELEVENLABS_OUTPUT_FORMAT", "mp3_44100_128")
# Per-character ElevenLabs voice IDs. Defaults are public premade voices on the
# ElevenLabs platform. Override per host via env: ELEVEN_VOICE_<NAME>.
# Browse voices at https://elevenlabs.io/app/voice-library to pick custom IDs.
ELEVEN_CHARACTER_VOICES = {
    # 3-host cast: warm female host, deep male analyst, bright female tech.
    # Three distinct timbres listeners can identify within one syllable.
    "JAMIE": os.environ.get("ELEVEN_VOICE_JAMIE", "EXAVITQu4vr4xnSDxMaL"),  # Sarah — confident, warm, professional host
    "ALEX":  os.environ.get("ELEVEN_VOICE_ALEX",  "nPczCjzI2devNBz1zQrb"),  # Brian — deep, resonant analyst
    "MAYA":  os.environ.get("ELEVEN_VOICE_MAYA",  "cgSgspJ2msm6clMCkdW9"),  # Jessica — playful, bright tech reporter
}

# Podcast metadata (edit these)
PODCAST_TITLE = "Market Today, Explained — Daily Markets & Tech News"
PODCAST_AUTHOR = "Saksham Gupta | Daily Markets Podcast"
PODCAST_EMAIL = "gsaksham@gmail.com"
PODCAST_DESCRIPTION = (
    "Daily fast, funny roundtable on US markets, business, tech, world, and culture. "
    "Eight hosts riff on the day's news in 5–10 minutes. AI-generated each afternoon. "
    "Not investment advice — see disclaimer."
)
PODCAST_LANGUAGE = "en-us"
# Set after GitHub Pages is live. Example: https://<user>.github.io/<repo>
PODCAST_BASE_URL = "https://sakzgupzzz.github.io/explain-market-today"
PODCAST_CATEGORY = "Business"
PODCAST_SUBCATEGORY = "Investing"
# Stable show identity for Podcasting 2.0. Random UUID generated once; never change.
PODCAST_GUID = os.environ.get("PODCAST_GUID", "0d3b1a8e-3e8d-4f7a-a4b2-9e6d1f4a2c5b")

# Legal disclaimer. Short version is read aloud in the outro by the LLM;
# full version goes in feed + episode descriptions.
DISCLAIMER_SHORT = (
    "This show is for entertainment and education only — nothing here is investment advice."
)
DISCLAIMER_FULL = (
    "Market Today, Explained is for entertainment and educational purposes only. "
    "Nothing in this podcast is investment, financial, legal, or tax advice. "
    "Hosts are AI-generated voices, not licensed financial advisors, and have no fiduciary "
    "relationship with listeners. Market data may be delayed or inaccurate. "
    "Some segments are dramatized for comedic effect; references to real companies and "
    "people are made for commentary and satire. Always consult a licensed professional "
    "before making investment decisions."
)

# Phrases the LLM tends to over-use. Banned in the prompt; sanitizer scrubs leftovers.
BANNED_PHRASES = [
    "buckle up",
    "let's dive in",
    "let's dive into",
    "in today's fast-paced world",
    "fascinating",
    "welcome to the show",
    "welcome back",
    "welcome to your daily",
    "good morning everyone",
    "good morning folks",
    "hey everyone",
    "hello listeners",
    "well folks",
    "as always",
    "stay tuned",
    "without further ado",
    "ready for some laughs",
    "ready for some insights",
    "without a doubt",
    "needless to say",
    "at the end of the day",
]

# Show structure beats — used for chapter generation + critique pass.
BEATS = [
    "cold_open",
    "markets",
    "big_story",
    "quick_hits",
    "odd_thing",
    "sign_off",
]
BEAT_TITLES = {
    "cold_open": "Cold open",
    "markets": "Markets",
    "big_story": "Big story",
    "quick_hits": "Quick hits",
    "odd_thing": "Odd thing of the day",
    "sign_off": "Sign-off",
}
