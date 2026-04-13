"""
Social media URL scraper.

Fetches publicly visible text from a URL using requests + BeautifulSoup.
For Twitter/X use the official API if a bearer token is available.
"""
import logging
import json
import re
from typing import Dict, List
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)

TWITTER_URL_RE = re.compile(r"https?://(www\.)?(twitter\.com|x\.com)/\S+/status/(\d+)", re.IGNORECASE)
INSTAGRAM_URL_RE = re.compile(r"https?://(www\.)?instagram\.com/(p|reel)/([A-Za-z0-9_-]+)", re.IGNORECASE)


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _http_get(url: str, timeout: int = 15, headers: Dict = None) -> str:
    """HTTP GET helper for remote public content retrieval."""
    req = Request(url, headers=headers or {})
    with urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", errors="replace")


def _http_get_json(url: str, timeout: int = 15, headers: Dict = None) -> Dict:
    payload = _http_get(url, timeout=timeout, headers=headers)
    return json.loads(payload)


def _fetch_twitter(tweet_id: str, bearer_token: str) -> Dict:
    """Fetch tweet text via Twitter v2 API."""
    try:
        base_url = f"https://api.twitter.com/2/tweets/{tweet_id}"
        params = {
            "tweet.fields": "created_at,text",
        }
        url = f"{base_url}?{urlencode(params)}"
        headers = {"Authorization": f"Bearer {bearer_token}"}
        data = _http_get_json(url, timeout=10, headers=headers)
        tweet = data.get("data", {})
        text = _normalize_text(tweet.get("text", ""))
        if not text:
            return {"posts": []}
        return {
            "posts": [{
                "id": tweet.get("id", tweet_id),
                "text": text,
                "created_at": tweet.get("created_at", ""),
            }]
        }
    except Exception as exc:
        logger.warning("Twitter API fetch failed: %s", exc)
        return {"posts": []}


def _extract_social_text_from_html(html: str) -> str:
    """Extract social post-like text from meta/script HTML blocks."""
    try:
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html, "html.parser")

        for selector in [
            "meta[property='og:description']",
            "meta[name='description']",
            "meta[property='twitter:description']",
        ]:
            tag = soup.select_one(selector)
            if tag and tag.get("content"):
                text = _normalize_text(tag.get("content", ""))
                if text:
                    return text

        for script in soup.find_all("script", attrs={"type": "application/ld+json"}):
            try:
                payload = json.loads(script.get_text(strip=True) or "{}")
            except Exception:
                continue

            candidates = []
            if isinstance(payload, dict):
                candidates.extend([
                    payload.get("articleBody", ""),
                    payload.get("caption", ""),
                    payload.get("description", ""),
                ])
            for candidate in candidates:
                text = _normalize_text(str(candidate))
                if text:
                    return text
    except Exception as exc:
        logger.debug("Social HTML parse failed: %s", exc)

    return ""


def _fetch_instagram_post(post_url: str, access_token: str = "") -> Dict:
    """Fetch Instagram post text (caption/description) via oEmbed or HTML fallback."""
    try:
        if access_token:
            # Meta Graph oEmbed endpoint (requires app/user token with oEmbed permissions).
            oembed_url = (
                "https://graph.facebook.com/v21.0/instagram_oembed"
                f"?url={quote(post_url, safe='')}&access_token={quote(access_token, safe='')}"
            )
            data = _http_get_json(oembed_url, timeout=12)
            text = _normalize_text(data.get("title", "") or data.get("author_name", ""))
            if text:
                return {
                    "posts": [{
                        "id": data.get("media_id", ""),
                        "text": text,
                        "created_at": "",
                    }]
                }

        headers = {"User-Agent": "Mozilla/5.0 (X-Sense Sentiment Analyser)"}
        html = _http_get(post_url, headers=headers, timeout=15)
        text = _extract_social_text_from_html(html)
        if text:
            return {"posts": [{"id": "", "text": text, "created_at": ""}]}
    except Exception as exc:
        logger.warning("Instagram fetch failed: %s", exc)

    return {"posts": []}


def _fetch_generic_url(url: str) -> Dict:
    """Scrape visible paragraph text from any public web page."""
    try:
        from bs4 import BeautifulSoup

        headers = {"User-Agent": "Mozilla/5.0 (X-Sense Sentiment Analyser)"}
        html = _http_get(url, headers=headers, timeout=15)
        soup = BeautifulSoup(html, "html.parser")

        # Remove script / style noise
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()

        text = " ".join(p.get_text(separator=" ") for p in soup.find_all(["p", "article", "section"]))
        text = _normalize_text(text[:5000])
        if not text:
            return {"posts": []}
        return {"posts": [{"id": "", "text": text, "created_at": ""}]}
    except Exception as exc:
        logger.warning("Generic URL fetch failed for %s: %s", url, exc)
        return {"posts": []}


def _coalesce_posts(posts: List[Dict]) -> List[Dict]:
    cleaned = []
    for idx, post in enumerate(posts, start=1):
        text = _normalize_text(str(post.get("text", "")))
        if not text:
            continue
        cleaned.append({
            "id": post.get("id") or f"post_{idx}",
            "text": text,
            "created_at": post.get("created_at", ""),
        })
    return cleaned


def fetch_text_from_url(url: str, bearer_token: str = "", instagram_token: str = "") -> Dict:
    """Return extracted URL content and a list of post-like text items."""
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
        out = _fetch_twitter(tweet_id, bearer_token)
        posts = _coalesce_posts(out.get("posts", []))
        if posts:
            return {
                "text": "\n\n".join(p["text"] for p in posts),
                "source": "twitter",
                "posts": posts,
            }

    if m:
        generic = _fetch_generic_url(url)
        posts = _coalesce_posts(generic.get("posts", []))
        if posts:
            return {
                "text": "\n\n".join(p["text"] for p in posts),
                "source": "twitter_scrape",
                "posts": posts,
            }

    ig = INSTAGRAM_URL_RE.match(url)
    if ig:
        out = _fetch_instagram_post(url, instagram_token)
        posts = _coalesce_posts(out.get("posts", []))
        if posts:
            return {
                "text": "\n\n".join(p["text"] for p in posts),
                "source": "instagram",
                "posts": posts,
            }

    generic = _fetch_generic_url(url)
    posts = _coalesce_posts(generic.get("posts", []))
    return {
        "text": "\n\n".join(p["text"] for p in posts),
        "source": "web_scrape",
        "posts": posts,
    }
