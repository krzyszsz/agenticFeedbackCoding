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
WORD_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9'./+-]{1,}")
FILENAME_TOKEN_RE = re.compile(r"^[A-Za-z0-9_.-]+\.(?:md|txt|json|csv|tsv|py|js|html|css|xml|yaml|yml)$", re.I)
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
SEARCH_STOPWORDS = {
    "about",
    "after",
    "analysis",
    "available",
    "benchmark",
    "build",
    "capture",
    "checks",
    "cite",
    "clear",
    "code",
    "concise",
    "current",
    "deliverable",
    "deliverables",
    "documented",
    "documentation",
    "explaining",
    "facts",
    "fetched",
    "file",
    "files",
    "focus",
    "include",
    "includes",
    "instructions",
    "invent",
    "long",
    "maintain",
    "medium",
    "must",
    "output",
    "project",
    "report",
    "research",
    "review",
    "script",
    "separate",
    "sourced",
    "short",
    "source",
    "sources",
    "summary",
    "term",
    "validation",
    "with",
    "write",
    "all",
    "not",
    "use",
    "web",
}


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


def _request_with_content_type(url: str, cfg: WebResearchConfig) -> tuple[bytes, str]:
    req = urllib.request.Request(url, headers={"User-Agent": cfg.user_agent})
    with urllib.request.urlopen(req, timeout=cfg.timeout_seconds) as response:
        content_type = response.headers.get_content_type() if response.headers else ""
        return response.read(cfg.max_page_bytes), content_type


def _looks_like_text_content(raw: bytes, content_type: str) -> bool:
    """Return whether a fetched response is safe to decode into prompt text."""
    normalized = content_type.lower()
    if normalized.startswith("text/") or normalized in {
        "application/json",
        "application/ld+json",
        "application/xml",
        "application/xhtml+xml",
        "application/rss+xml",
        "application/atom+xml",
    }:
        return True
    if normalized and normalized not in {"application/octet-stream", "binary/octet-stream"}:
        return False
    if raw.startswith(b"%PDF"):
        return False
    sample = raw[:4096]
    if not sample:
        return True
    control_bytes = sum(1 for byte in sample if byte < 32 and byte not in (9, 10, 13))
    replacement_chars = raw[:4096].decode("utf-8", errors="replace").count("\ufffd")
    return control_bytes / max(len(sample), 1) < 0.08 and replacement_chars < 20


def fetch_page(url: str, cfg: WebResearchConfig) -> dict[str, Any]:
    try:
        raw, content_type = _request_with_content_type(url, cfg)
        if not _looks_like_text_content(raw, content_type):
            return {
                "url": url,
                "status": "error",
                "title": url,
                "excerpt": "",
                "error": (
                    f"Unsupported non-text content type {content_type or 'unknown'}; "
                    "URL was fetched but no prompt-safe text excerpt was extracted."
                ),
                "content_type": content_type,
                "bytes_read": len(raw),
            }
        parser = TextExtractor()
        parser.feed(raw.decode("utf-8", errors="replace"))
        text = parser.text
        return {
            "url": url,
            "status": "ok",
            "title": parser.title or url,
            "excerpt": text[: cfg.excerpt_chars],
            "content_type": content_type,
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


def search_queries_for_prompt(prompt: str) -> list[str]:
    """Create focused search queries from a project prompt.

    Project prompts often mix the topic with deliverable instructions. Sending
    the first few hundred characters verbatim to a search engine can produce no
    useful results, so this function keeps topic-bearing words and phrase-like
    fragments without adding any workload-specific recipes. Example benchmarks
    must not leak into this generic search helper.
    """
    compact_prompt = " ".join(prompt.split())
    queries: list[str] = []

    def filename_token_count(text: str) -> int:
        return sum(1 for token in WORD_RE.findall(text) if FILENAME_TOKEN_RE.match(token.strip(".,:;()[]{}\"'")))

    if len(compact_prompt) <= 240 and filename_token_count(compact_prompt) < 2:
        queries.append(compact_prompt)

    sentences = [
        sentence.strip(" -:;,.")
        for sentence in re.split(r"(?<=[.!?])\s+|\n+", compact_prompt)
        if sentence.strip()
    ]
    scored_sentences: list[tuple[int, str]] = []
    for sentence in sentences:
        # Sentences that mostly list desired artifacts (README.md,
        # SOURCES.json, validate.py, etc.) describe the output contract, not
        # the external topic to research. Skipping them keeps the helper
        # generic without baking in any benchmark-specific query text.
        if filename_token_count(sentence) >= 2:
            continue
        words = [
            word.strip(".,:;()[]{}\"'").lower()
            for word in WORD_RE.findall(sentence)
        ]
        topical = [
            word for word in words
            if len(word) >= 3 and word not in SEARCH_STOPWORDS
        ]
        if topical:
            scored_sentences.append((len(topical), sentence))
    for _, sentence in sorted(scored_sentences, key=lambda item: item[0], reverse=True)[:2]:
        if len(sentence) <= 240:
            queries.append(sentence)

    terms: list[str] = []
    for word in WORD_RE.findall(compact_prompt):
        stripped = word.strip(".,:;()[]{}\"'")
        normalized = stripped.lower()
        if FILENAME_TOKEN_RE.match(stripped):
            continue
        if len(normalized) < 3 or normalized in SEARCH_STOPWORDS:
            continue
        # Preserve recognisable acronyms/proper nouns in the query text while
        # deduplicating case-insensitively.
        if normalized not in {item.lower() for item in terms}:
            terms.append(stripped)
        if len(terms) >= 24:
            break
    if terms:
        queries.append(" ".join(terms[:12]))
        if len(terms) > 12:
            queries.append(" ".join(terms))

    deduped: list[str] = []
    for query in queries:
        query = query[:240].strip()
        if query and query not in deduped:
            deduped.append(query)
    return deduped or [compact_prompt[:240]]


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
        queries = search_queries_for_prompt(prompt)
        max_per_query = max(1, (cfg.max_pages + max(len(queries), 1) - 1) // max(len(queries), 1))
        for query in queries:
            added_for_query = 0
            for item in search_web(query, cfg):
                if item.startswith("ERROR:"):
                    search_errors.append(item[6:])
                    continue
                if item not in urls:
                    urls.append(item)
                    added_for_query += 1
                if len(urls) >= cfg.max_pages:
                    break
                if added_for_query >= max_per_query:
                    break
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
