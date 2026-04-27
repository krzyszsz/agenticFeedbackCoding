from __future__ import annotations

from dataclasses import asdict
from html.parser import HTMLParser
import json
import re
from typing import Any
from urllib.parse import quote_plus, unquote, urlparse, parse_qs
import urllib.error
import urllib.request

from .config import WebResearchConfig

URL_RE = re.compile(r"https?://[^\s)\]}>\"']+")
EXPLICIT_RESEARCH_MARKERS = [
    "search the web",
    "research the web",
    "browse the web",
    "look up",
    "google",
    "find online",
    "current",
    "latest",
    "recent",
    "up to date",
    "documentation",
    "official docs",
    "docs for",
]


class TextExtractor(HTMLParser):
    """Tiny HTML-to-text extractor used by the research tool.

    BeautifulSoup is available in the runtime, but using the stdlib parser here
    keeps the core harness testable even before optional dependencies are
    installed. Script/style content is ignored to keep summaries compact.
    """

    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self.title_parts: list[str] = []
        self._skip_depth = 0
        self._in_title = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"script", "style", "noscript"}:
            self._skip_depth += 1
        if tag == "title":
            self._in_title = True

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "noscript"} and self._skip_depth:
            self._skip_depth -= 1
        if tag == "title":
            self._in_title = False

    def handle_data(self, data: str) -> None:
        text = " ".join(data.split())
        if not text or self._skip_depth:
            return
        if self._in_title:
            self.title_parts.append(text)
        self.parts.append(text)

    @property
    def title(self) -> str:
        return " ".join(self.title_parts).strip()

    @property
    def text(self) -> str:
        return " ".join(self.parts).strip()


def explicit_research_requested(prompt: str) -> bool:
    lower = prompt.lower()
    return bool(URL_RE.search(prompt)) or any(marker in lower for marker in EXPLICIT_RESEARCH_MARKERS)


def extract_urls(prompt: str) -> list[str]:
    urls: list[str] = []
    for match in URL_RE.findall(prompt):
        cleaned = match.rstrip(".,;:")
        if cleaned not in urls:
            urls.append(cleaned)
    return urls


def _request(url: str, cfg: WebResearchConfig) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": cfg.user_agent})
    with urllib.request.urlopen(req, timeout=cfg.timeout_seconds) as response:
        return response.read(cfg.max_page_bytes)


def fetch_page(url: str, cfg: WebResearchConfig) -> dict[str, Any]:
    try:
        raw = _request(url, cfg)
        parser = TextExtractor()
        parser.feed(raw.decode("utf-8", errors="replace"))
        text = parser.text
        return {
            "url": url,
            "status": "ok",
            "title": parser.title or url,
            "excerpt": text[: cfg.excerpt_chars],
            "bytes_read": len(raw),
        }
    except (urllib.error.URLError, TimeoutError, OSError, UnicodeError) as exc:
        return {"url": url, "status": "error", "error": str(exc)}


def _extract_duckduckgo_url(href: str) -> str:
    if href.startswith("//"):
        href = "https:" + href
    parsed = urlparse(href)
    if parsed.netloc.endswith("duckduckgo.com") and parsed.path.startswith("/l/"):
        target = parse_qs(parsed.query).get("uddg", [""])[0]
        return unquote(target) if target else href
    return href


def search_web(query: str, cfg: WebResearchConfig) -> list[str]:
    """Best-effort lightweight search used when a prompt asks for web research.

    This is deliberately small and optional. If a search engine blocks the
    request, the research result records the error and the agent can still ask
    for a better source URL instead of pretending research happened.
    """
    url = f"https://duckduckgo.com/html/?q={quote_plus(query)}"
    try:
        raw = _request(url, cfg)
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        return [f"ERROR:{exc}"]
    html = raw.decode("utf-8", errors="replace")
    links: list[str] = []
    for href in re.findall(r'<a[^>]+class=["\']result__a["\'][^>]+href=["\']([^"\']+)', html):
        target = _extract_duckduckgo_url(href)
        if target.startswith("http") and target not in links:
            links.append(target)
        if len(links) >= cfg.max_search_results:
            break
    return links


def run_web_research(prompt: str, cfg: WebResearchConfig) -> dict[str, Any]:
    if not cfg.enabled:
        return {
            "status": "skipped",
            "requested": False,
            "reason": "web_research.enabled is false.",
            "config": asdict(cfg),
            "targets": [],
        }
    if not explicit_research_requested(prompt):
        return {
            "status": "skipped",
            "requested": False,
            "reason": "No explicit web research request or source URL detected.",
            "config": asdict(cfg),
            "targets": [],
        }

    urls = extract_urls(prompt)
    search_errors: list[str] = []
    if not urls:
        for item in search_web(prompt[:500], cfg):
            if item.startswith("ERROR:"):
                search_errors.append(item[6:])
                continue
            urls.append(item)
            if len(urls) >= cfg.max_pages:
                break

    urls = urls[: cfg.max_pages]
    targets = [fetch_page(url, cfg) for url in urls]
    ok_count = sum(1 for item in targets if item.get("status") == "ok")
    status = "completed" if ok_count else "failed"
    if ok_count and ok_count < len(targets):
        status = "partial"

    return {
        "status": status,
        "requested": True,
        "config": asdict(cfg),
        "search_errors": search_errors,
        "targets": targets,
    }


def research_to_markdown(result: dict[str, Any]) -> str:
    lines = ["# Web Research Evidence", ""]
    lines.append(f"- Status: {result.get('status')}")
    lines.append(f"- Requested: {result.get('requested')}")
    if result.get("reason"):
        lines.append(f"- Reason: {result['reason']}")
    errors = result.get("search_errors") or []
    if errors:
        lines.append("- Search errors:")
        for error in errors:
            lines.append(f"  - {error}")
    lines.append("")
    lines.append("## Sources")
    targets = result.get("targets") or []
    if not targets:
        lines.append("")
        lines.append("- None.")
    for index, item in enumerate(targets, start=1):
        lines.append("")
        lines.append(f"### Source {index}")
        lines.append(f"- URL: {item.get('url')}")
        lines.append(f"- Status: {item.get('status')}")
        if item.get("title"):
            lines.append(f"- Title: {item['title']}")
        if item.get("error"):
            lines.append(f"- Error: {item['error']}")
        if item.get("excerpt"):
            lines.append("")
            lines.append("Excerpt:")
            lines.append("")
            lines.append(item["excerpt"])
    return "\n".join(lines) + "\n"


def compact_research_for_prompt(result: dict[str, Any], max_chars: int = 6000) -> str:
    payload = {
        "status": result.get("status"),
        "requested": result.get("requested"),
        "targets": [
            {
                "url": item.get("url"),
                "status": item.get("status"),
                "title": item.get("title"),
                "excerpt": item.get("excerpt", "")[:1200],
                "error": item.get("error"),
            }
            for item in result.get("targets", [])
        ],
    }
    text = json.dumps(payload, ensure_ascii=True)
    return text[:max_chars]
