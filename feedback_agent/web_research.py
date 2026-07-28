from __future__ import annotations

from dataclasses import asdict
from html.parser import HTMLParser
import ipaddress
import json
import re
import socket
from typing import Any
from urllib.parse import quote_plus, unquote, urlparse, parse_qs
import urllib.error
import urllib.request

from .config import WebResearchConfig

URL_RE = re.compile(r"https?://[^\s)\]}>\"']+")


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


class SearchResultLinkExtractor(HTMLParser):
    """Collect result links using HTML structure rather than page-text regex."""

    def __init__(self) -> None:
        super().__init__()
        self.hrefs: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag != "a":
            return
        attributes = dict(attrs)
        classes = set(str(attributes.get("class") or "").split())
        href = attributes.get("href")
        if "result__a" in classes and href:
            self.hrefs.append(href)


def extract_urls(prompt: str) -> list[str]:
    urls: list[str] = []
    for match in URL_RE.findall(prompt):
        cleaned = match.rstrip(".,;:")
        if cleaned not in urls:
            urls.append(cleaned)
    return urls


def _validate_network_target(url: str, cfg: WebResearchConfig) -> None:
    """Reject model-selected local/private fetch targets unless explicitly allowed."""
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("Research URL must use http or https and include a hostname.")
    if parsed.username or parsed.password:
        raise ValueError("Research URL must not contain embedded credentials.")
    if cfg.allow_private_network:
        return
    try:
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
    except ValueError as exc:
        raise ValueError(f"Research URL has an invalid port: {exc}") from exc
    try:
        literal = ipaddress.ip_address(parsed.hostname)
        addresses = {literal}
    except ValueError:
        try:
            resolved = socket.getaddrinfo(parsed.hostname, port, type=socket.SOCK_STREAM)
        except socket.gaierror as exc:
            raise ValueError(f"Research hostname could not be resolved: {parsed.hostname}") from exc
        addresses = {ipaddress.ip_address(item[4][0]) for item in resolved}
    if not addresses:
        raise ValueError(f"Research hostname resolved to no addresses: {parsed.hostname}")
    blocked = [str(address) for address in addresses if not address.is_global]
    if blocked:
        raise ValueError(
            "Research URL resolves to a private, loopback, link-local, reserved, or otherwise non-public "
            "address. Set web_research.allow_private_network=true only for a trusted local source."
        )


class _ValidatedRedirectHandler(urllib.request.HTTPRedirectHandler):
    def __init__(self, cfg: WebResearchConfig):
        self.cfg = cfg
        super().__init__()

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[no-untyped-def]
        _validate_network_target(newurl, self.cfg)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _open_url(url: str, cfg: WebResearchConfig):
    _validate_network_target(url, cfg)
    req = urllib.request.Request(url, headers={"User-Agent": cfg.user_agent})
    opener = urllib.request.build_opener(_ValidatedRedirectHandler(cfg))
    return opener.open(req, timeout=cfg.timeout_seconds)


def _request(url: str, cfg: WebResearchConfig) -> bytes:
    with _open_url(url, cfg) as response:
        return response.read(cfg.max_page_bytes)


def _request_with_content_type(url: str, cfg: WebResearchConfig) -> tuple[bytes, str, bool]:
    with _open_url(url, cfg) as response:
        content_type = response.headers.get_content_type() if response.headers else ""
        raw = response.read(cfg.max_page_bytes + 1)
        truncated = len(raw) > cfg.max_page_bytes
        return raw[: cfg.max_page_bytes], content_type, truncated


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
        raw, content_type, response_truncated = _request_with_content_type(url, cfg)
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
                "response_truncated": response_truncated,
            }
        parser = TextExtractor()
        parser.feed(raw.decode("utf-8", errors="replace"))
        text = parser.text
        excerpt_truncated = len(text) > cfg.excerpt_chars or response_truncated
        excerpt = text[: cfg.excerpt_chars]
        if excerpt_truncated:
            excerpt += (
                f"\n[research source truncated: retained at most {cfg.max_page_bytes} response bytes "
                f"and {cfg.excerpt_chars} text characters]"
            )
        return {
            "url": url,
            "status": "ok",
            "title": parser.title or url,
            "excerpt": excerpt,
            "content_type": content_type,
            "bytes_read": len(raw),
            "response_truncated": response_truncated,
        }
    except (urllib.error.URLError, TimeoutError, OSError, UnicodeError, ValueError) as exc:
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
    except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
        return [f"ERROR:{exc}"]
    parser = SearchResultLinkExtractor()
    parser.feed(raw.decode("utf-8", errors="replace"))
    links: list[str] = []
    for href in parser.hrefs:
        target = _extract_duckduckgo_url(href)
        if target.startswith("http") and target not in links:
            links.append(target)
        if len(links) >= cfg.max_search_results:
            break
    return links


def run_web_research(request: dict[str, Any], cfg: WebResearchConfig) -> dict[str, Any]:
    """Fetch bounded sources selected by the model-owned research protocol."""
    if not cfg.enabled:
        return {
            "status": "skipped",
            "requested": False,
            "reason": "web_research.enabled is false.",
            "config": asdict(cfg),
            "targets": [],
        }
    decision = str(request.get("decision") or "").strip()
    if decision not in {"research", "skip"}:
        return {
            "status": "failed",
            "requested": False,
            "reason": f"Unsupported research-decision protocol token: {decision!r}.",
            "protocol_error": True,
            "config": asdict(cfg),
            "targets": [],
        }
    queries_value = request.get("queries")
    urls_value = request.get("urls")
    if (
        not isinstance(queries_value, list)
        or not all(isinstance(value, str) for value in queries_value)
        or not isinstance(urls_value, list)
        or not all(isinstance(value, str) for value in urls_value)
    ):
        return {
            "status": "failed",
            "requested": False,
            "reason": "Research-decision queries and urls must be lists of strings.",
            "protocol_error": True,
            "config": asdict(cfg),
            "targets": [],
        }
    if decision == "skip":
        return {
            "status": "skipped",
            "requested": False,
            "reason": str(request.get("rationale") or "The research decision skipped external fetching."),
            "config": asdict(cfg),
            "targets": [],
        }

    urls: list[str] = []
    for value in urls_value:
        if len(urls) >= cfg.max_pages:
            break
        url = str(value).strip().rstrip(".,;:")
        parsed = urlparse(url)
        if parsed.scheme in {"http", "https"} and parsed.netloc and url not in urls:
            urls.append(url)
    queries: list[str] = []
    for value in queries_value:
        if len(queries) >= cfg.max_search_results:
            break
        query = " ".join(str(value).split())[:240]
        if query and query not in queries:
            queries.append(query)
    search_errors: list[str] = []
    if not urls:
        if not queries:
            return {
                "status": "failed",
                "requested": True,
                "reason": "Research was selected but no valid URL or search query was supplied.",
                "config": asdict(cfg),
                "targets": [],
            }
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
                "response_truncated": item.get("response_truncated", False),
            }
            for item in result.get("targets", [])
        ],
    }
    text = json.dumps(payload, ensure_ascii=True)
    if max_chars <= 0:
        return ""
    if len(text) <= max_chars:
        return text
    marker = f"\n[research evidence truncated: kept head and tail from {len(text)} chars]\n"
    if len(marker) >= max_chars:
        return marker[:max_chars]
    available = max(0, max_chars - len(marker))
    head = available // 2
    tail = available - head
    return text[:head] + marker + (text[-tail:] if tail else "")
