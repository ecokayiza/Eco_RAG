from __future__ import annotations

import json
from html import unescape
from html.parser import HTMLParser
from typing import Any
from urllib.parse import parse_qs, quote_plus, urlparse, urlunparse
from urllib.request import Request, urlopen

from echo.settings import load_app_settings

SEARCH_BACKENDS = {
    "auto": (
        ("duckduckgo_html", "https://html.duckduckgo.com/html/?q={query}&kl=us-en", "duckduckgo"),
        ("duckduckgo_lite", "https://lite.duckduckgo.com/lite/?q={query}&kl=us-en", "duckduckgo"),
        ("baidu", "http://www.baidu.com/s?wd={query}&ie=utf-8", "baidu"),
    ),
    "duckduckgo": (
        ("duckduckgo_html", "https://html.duckduckgo.com/html/?q={query}&kl=us-en", "duckduckgo"),
        ("duckduckgo_lite", "https://lite.duckduckgo.com/lite/?q={query}&kl=us-en", "duckduckgo"),
    ),
    "baidu": (("baidu", "http://www.baidu.com/s?wd={query}&ie=utf-8", "baidu"),),
}
REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}


def web_search(
    query: str | None = None,
    queries: list[str] | None = None,
    max_results: int = 5,
) -> dict[str, Any]:
    """Search the public web for one or more fresh or external queries."""
    prepared_queries = [
        " ".join(str(item or "").strip().split())
        for item in ([query] if query is not None else []) + list(queries or [])
    ]
    prepared_queries = [item for item in prepared_queries if item]
    limit = _clamp_max_results(max_results)
    if not prepared_queries:
        return {
            "type": "context",
            "skill_name": "web_search",
            "items": [],
            "error": "Query cannot be empty.",
        }

    selected_backend = load_app_settings().web_search_backend
    items = []
    errors = []
    for cleaned in prepared_queries:
        try:
            items.extend(_search_one_query(cleaned, selected_backend))
        except Exception as exc:
            errors.append(f"{cleaned}: {exc}")

    items = _dedupe_items(items)[:limit]
    if not items and errors:
        return {
            "type": "context",
            "skill_name": "web_search",
            "items": [],
            "error": "; ".join(errors),
        }

    return {
        "type": "context",
        "skill_name": "web_search",
        "backend": selected_backend,
        "items": items,
    }


def _clamp_max_results(max_results: Any, default: int = 5) -> int:
    try:
        requested = int(max_results or default)
    except (TypeError, ValueError):
        requested = default
    configured_limit = load_app_settings().max_web_search_results
    return max(1, min(requested, max(1, configured_limit)))


class _DuckDuckGoParser(HTMLParser):
    """Parse DuckDuckGo HTML and Lite result pages."""

    def __init__(self):
        super().__init__()
        self.items: list[dict[str, str | None]] = []
        self._current: dict[str, Any] | None = None
        self._title_depth = 0
        self._snippet_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]):
        attributes = dict(attrs)
        classes = set((attributes.get("class") or "").split())

        if "result__a" in classes or "result-link" in classes:
            self._flush()
            self._current = {
                "title_parts": [],
                "snippet_parts": [],
                "url": _resolve_result_url(attributes.get("href")),
            }
            self._title_depth = 1
            return

        if self._current is None:
            return

        if "result__snippet" in classes or "result-snippet" in classes:
            self._snippet_depth = 1
            return

        if self._title_depth:
            self._title_depth += 1
        if self._snippet_depth:
            self._snippet_depth += 1

    def handle_endtag(self, _tag: str):
        if self._title_depth:
            self._title_depth -= 1
        if self._snippet_depth:
            self._snippet_depth -= 1

    def handle_data(self, data: str):
        if self._current is None:
            return
        if self._title_depth:
            self._current["title_parts"].append(data)
        elif self._snippet_depth:
            self._current["snippet_parts"].append(data)

    def close(self):
        super().close()
        self._flush()

    def _flush(self):
        if self._current is None:
            return
        title = " ".join(" ".join(self._current["title_parts"]).split())
        snippet = " ".join(" ".join(self._current["snippet_parts"]).split())
        if title:
            self.items.append(
                {
                    "title": unescape(title),
                    "content": unescape(snippet) or unescape(title),
                    "url": self._current["url"],
                }
            )
        self._current = None
        self._title_depth = 0
        self._snippet_depth = 0


class _BaiduParser(HTMLParser):
    """Parse Baidu result pages."""

    def __init__(self):
        super().__init__()
        self.items: list[dict[str, str | None]] = []
        self._current: dict[str, Any] | None = None
        self._container_depth = 0
        self._title_depth = 0
        self._snippet_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]):
        attributes = dict(attrs)
        classes = set((attributes.get("class") or "").split())

        if tag == "div" and "result" in classes and "c-container" in classes:
            self._flush()
            self._current = {
                "title_parts": [],
                "snippet_parts": [],
                "url": attributes.get("mu"),
                "fallback_url": None,
            }
            self._container_depth = 1
            return

        if self._current is None:
            return

        if self._container_depth and tag == "div":
            self._container_depth += 1

        if tag == "a" and self._current.get("fallback_url") is None:
            self._current["fallback_url"] = attributes.get("href")
            if self._title_depth == 0:
                self._title_depth = 1
                return

        if tag in {"h3", "a"} and self._title_depth:
            self._title_depth += 1
            return

        if "c-abstract" in classes or "content-right" in classes or "summary-content" in classes:
            self._snippet_depth = 1
            return

        if self._snippet_depth:
            self._snippet_depth += 1

    def handle_endtag(self, tag: str):
        if self._title_depth:
            self._title_depth -= 1
        if self._snippet_depth:
            self._snippet_depth -= 1
        if self._current is not None and self._container_depth and tag == "div":
            self._container_depth -= 1
            if self._container_depth <= 0:
                self._flush()

    def handle_data(self, data: str):
        if self._current is None:
            return
        if self._title_depth:
            self._current["title_parts"].append(data)
        elif self._snippet_depth:
            self._current["snippet_parts"].append(data)

    def handle_comment(self, data: str):
        if self._current is None or not data.startswith("s-data:"):
            return
        try:
            payload = json.loads(data.removeprefix("s-data:"))
        except json.JSONDecodeError:
            return
        text = " ".join(_json_text_values(payload))
        if text:
            self._current["snippet_parts"].append(text)

    def close(self):
        super().close()
        self._flush()

    def _flush(self):
        if self._current is None:
            return
        title = _plain_text(" ".join(self._current["title_parts"]))
        snippet = _plain_text(" ".join(self._current["snippet_parts"]))
        url = self._current["url"] or self._current["fallback_url"]
        if title:
            self.items.append(
                {
                    "title": title,
                    "content": snippet or title,
                    "url": url,
                }
            )
        self._current = None
        self._container_depth = 0
        self._title_depth = 0
        self._snippet_depth = 0


def _search_one_query(query: str, backend: str) -> list[dict[str, str | None]]:
    """Search one query through free no-key endpoints with fallback."""
    errors = []
    for endpoint_name, url_template, parser_name in SEARCH_BACKENDS[_normalize_backend(backend)]:
        try:
            payload = _fetch(url_template.format(query=quote_plus(query)))
            items = _parse_payload(payload, parser_name)
            if items:
                return items
            errors.append(f"{endpoint_name} returned no results")
        except Exception as exc:
            errors.append(f"{endpoint_name} failed: {exc}")
    raise RuntimeError("; ".join(errors))


def _fetch(url: str) -> str:
    request = Request(url, headers=REQUEST_HEADERS)
    with urlopen(request, timeout=12) as response:
        return response.read().decode("utf-8", errors="replace")


def _parse_payload(payload: str, parser_name: str) -> list[dict[str, str | None]]:
    if parser_name == "duckduckgo":
        return _parse_duckduckgo_results(payload)
    if parser_name == "baidu":
        return _parse_baidu_results(payload)
    raise ValueError(f"Unknown search parser '{parser_name}'.")


def _parse_duckduckgo_results(html: str) -> list[dict[str, str | None]]:
    """Parse and deduplicate DuckDuckGo HTML search results."""
    parser = _DuckDuckGoParser()
    parser.feed(html)
    parser.close()

    return _dedupe_items(parser.items)


def _parse_baidu_results(html: str) -> list[dict[str, str | None]]:
    """Parse and deduplicate Baidu HTML search results."""
    parser = _BaiduParser()
    parser.feed(html)
    parser.close()
    return _dedupe_items(parser.items)


def _dedupe_items(items: list[dict[str, str | None]]) -> list[dict[str, str | None]]:
    """Deduplicate search results while preserving order."""
    deduped: list[dict[str, str | None]] = []
    seen: set[tuple[str, ...]] = set()
    for item in items:
        url = _normalize_url(item.get("url"))
        key = ("url", url) if url else ("text", str(item.get("title", "")), str(item.get("content", "")))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def _resolve_result_url(value: str | None) -> str | None:
    """Resolve DuckDuckGo redirect URLs into destination URLs."""
    if not value:
        return None
    parsed = urlparse(value)
    query = parse_qs(parsed.query)
    target = query.get("uddg", [None])[0]
    if target:
        return target
    if value.startswith("//"):
        return f"https:{value}"
    return value


def _normalize_backend(value: str | None) -> str:
    cleaned = (value or "auto").strip().lower()
    aliases = {"ddg": "duckduckgo", "duck": "duckduckgo", "baidu_search": "baidu"}
    backend = aliases.get(cleaned, cleaned)
    return backend if backend in SEARCH_BACKENDS else "auto"


def _plain_text(value: str | None) -> str:
    text = unescape(value or "")
    if "<" not in text:
        return " ".join(text.split())
    parser = _TextParser()
    parser.feed(text)
    parser.close()
    return " ".join(" ".join(parser.parts).split())


def _json_text_values(value: Any) -> list[str]:
    if isinstance(value, dict):
        values = []
        for key, item in value.items():
            if key == "text" and isinstance(item, str):
                values.append(_plain_text(item))
            else:
                values.extend(_json_text_values(item))
        return [item for item in values if item]
    if isinstance(value, list):
        values = []
        for item in value:
            values.extend(_json_text_values(item))
        return values
    return []


def _normalize_url(value: str | None) -> str:
    if not value:
        return ""
    parsed = urlparse(value)
    scheme = parsed.scheme.lower()
    netloc = parsed.netloc.lower()
    path = parsed.path.rstrip("/")
    return urlunparse((scheme, netloc, path, "", parsed.query, ""))


class _TextParser(HTMLParser):
    """Extract visible text from small HTML fragments."""

    def __init__(self):
        super().__init__()
        self.parts: list[str] = []

    def handle_data(self, data: str):
        if data:
            self.parts.append(data)
