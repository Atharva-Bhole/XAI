"""
Social media URL scraper.

Fetches publicly visible text from a URL using requests + BeautifulSoup.
For Twitter/X use the official API if a bearer token is available.
"""
import logging
import re
from typing import Dict

logger = logging.getLogger(__name__)

TWITTER_URL_RE = re.compile(r"https?://(www\.)?(twitter\.com|x\.com)/\S+/status/(\d+)", re.IGNORECASE)


def _fetch_twitter(tweet_id: str, bearer_token: str) -> str:
    """Fetch tweet text via Twitter v2 API."""
    try:
        import requests
        url = f"https://api.twitter.com/2/tweets/{tweet_id}"
        headers = {"Authorization": f"Bearer {bearer_token}"}
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        return data.get("data", {}).get("text", "")
    except Exception as exc:
        logger.warning("Twitter API fetch failed: %s", exc)
        return ""


def _fetch_generic_url(url: str) -> str:
    """Scrape visible paragraph text from any public web page."""
    try:
        import requests
        from bs4 import BeautifulSoup

        headers = {"User-Agent": "Mozilla/5.0 (X-Sense Sentiment Analyser)"}
        resp = requests.get(url, headers=headers, timeout=15)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        # Remove script / style noise
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()

        text = " ".join(p.get_text(separator=" ") for p in soup.find_all(["p", "article", "section"]))
        return text[:5000].strip()
    except Exception as exc:
        logger.warning("Generic URL fetch failed for %s: %s", url, exc)
        return ""


def fetch_text_from_url(url: str, bearer_token: str = "") -> Dict:
    """Return {"text": str, "source": str}"""
    # Validate the URL is http/https to prevent SSRF via other schemes
    safe_url_re = re.compile(r"^https?://", re.IGNORECASE)
    if not safe_url_re.match(url):
        return {"text": "", "source": "invalid_url", "error": "Only HTTP/HTTPS URLs are supported."}

    # Block private/internal addresses
    internal_re = re.compile(
        r"https?://(localhost|127\.\d+\.\d+\.\d+|10\.\d+\.\d+\.\d+|192\.168\.\d+\.\d+|172\.(1[6-9]|2\d|3[01])\.\d+\.\d+)",
        re.IGNORECASE,
    )
    if internal_re.match(url):
        return {"text": "", "source": "blocked", "error": "Access to internal network addresses is not allowed."}

    m = TWITTER_URL_RE.match(url)
    if m and bearer_token:
        tweet_id = m.group(3)
        text = _fetch_twitter(tweet_id, bearer_token)
        if text:
            return {"text": text, "source": "twitter"}

    text = _fetch_generic_url(url)
    return {"text": text, "source": "web_scrape"}
