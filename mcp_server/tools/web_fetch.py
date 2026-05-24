from __future__ import annotations

import base64
import asyncio
import threading
from html import unescape
from html.parser import HTMLParser
from typing import Any
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from echo.settings import load_app_settings
from .web_search import REQUEST_HEADERS

MAX_FETCH_CHARS = 50_000
SCREENSHOT_VIEWPORT = {"width": 1280, "height": 1600}


def web_fetch(url: str, max_chars: int = 8000) -> dict[str, Any]:
    """Fetch one public web page, or capture only a screenshot when screenshot mode is enabled."""
    try:
        cleaned_url = _validated_url(url)
        screenshot_mode = load_app_settings().web_fetch_screenshot_mode
        limit = max(1, min(int(max_chars or 8000), MAX_FETCH_CHARS))

        if screenshot_mode:
            try:
                title, content, image_url = _fetch_with_browser(cleaned_url)
            except Exception as exc:
                return _context([], error=f"Screenshot capture failed: {exc}")
            return _context(
                [
                    {
                        "title": title or urlparse(cleaned_url).netloc,
                        "url": cleaned_url,
                        "content": content[:limit],
                        "image_url": image_url,
                    }
                ]
            )

        html = _fetch(cleaned_url)
        title, content = _extract_text(html)
        if not content.strip():
            return _context([], error="No readable text was extracted from the page.")

        item = {
            "title": title or urlparse(cleaned_url).netloc,
            "url": cleaned_url,
            "content": content[:limit],
        }
        return _context(
            [item]
        )
    except Exception as exc:
        return _context([], error=str(exc))


def _validated_url(value: str) -> str:
    cleaned = str(value or "").strip()
    parsed = urlparse(cleaned)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("web_fetch requires an http or https URL.")
    return cleaned


def _fetch(url: str) -> str:
    request = Request(url, headers=REQUEST_HEADERS)
    with urlopen(request, timeout=15) as response:
        body = response.read()
        headers = getattr(response, "headers", None)
        charset = headers.get_content_charset() if hasattr(headers, "get_content_charset") else None
        return body.decode(charset or "utf-8", errors="replace")


def _fetch_with_browser(url: str) -> tuple[str, str, str]:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return _fetch_with_browser_sync(url)

    result: dict[str, tuple[str, str, str]] = {}
    error: dict[str, BaseException] = {}

    def run():
        try:
            result["value"] = _fetch_with_browser_sync(url)
        except BaseException as exc:
            error["value"] = exc

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    thread.join()
    if error:
        raise error["value"]
    return result["value"]


def _capture_screenshot_with_browser(url: str) -> str:
    return _fetch_with_browser(url)[2]


def _fetch_with_browser_sync(url: str) -> tuple[str, str, str]:
    try:
        from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
        from playwright.sync_api import sync_playwright
    except ImportError as exc:
        raise RuntimeError("Screenshot mode requires Playwright. Install it and run `python -m playwright install chromium`.") from exc

    errors = []
    with sync_playwright() as playwright:
        browser = None
        for channel in ("msedge", "chrome", None):
            try:
                launch_kwargs = {"headless": True}
                if channel:
                    launch_kwargs["channel"] = channel
                browser = playwright.chromium.launch(**launch_kwargs)
                break
            except Exception as exc:
                errors.append(str(exc))
        if browser is None:
            raise RuntimeError("Could not launch a Chromium browser for screenshot mode. " + " ".join(errors[-2:]))

        try:
            page = browser.new_page(
                viewport=SCREENSHOT_VIEWPORT,
                device_scale_factor=1,
                extra_http_headers=REQUEST_HEADERS,
            )
            try:
                page.goto(url, wait_until="domcontentloaded", timeout=20_000)
            except PlaywrightTimeoutError:
                pass
            try:
                page.wait_for_load_state("networkidle", timeout=2_000)
            except PlaywrightTimeoutError:
                pass
            title = _plain_text(page.title())
            try:
                content = _plain_text(page.locator("body").inner_text(timeout=2_000))
            except Exception:
                content = ""
            screenshot = page.screenshot(type="jpeg", quality=70, full_page=False)
        finally:
            browser.close()

    image_url = f"data:image/jpeg;base64,{base64.b64encode(screenshot).decode('ascii')}"
    return title, content, image_url


def _capture_screenshot_with_browser_sync(url: str) -> str:
    return _fetch_with_browser_sync(url)[2]


def _extract_text(html: str) -> tuple[str, str]:
    parser = _ReadableTextParser()
    parser.feed(html)
    parser.close()
    return _plain_text(" ".join(parser.title_parts)), _plain_text(" ".join(parser.content_parts))


def _plain_text(value: str) -> str:
    return " ".join(unescape(value or "").split())


def _context(items: list[dict[str, Any]], *, error: str | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "type": "context",
        "skill_name": "web_fetch",
        "items": items,
    }
    if error:
        payload["error"] = error
    return payload


class _ReadableTextParser(HTMLParser):
    """Extract title and visible text from simple HTML."""

    ignored_tags = {"script", "style", "noscript"}

    def __init__(self):
        super().__init__()
        self.title_parts: list[str] = []
        self.content_parts: list[str] = []
        self._ignored_depth = 0
        self._in_title = False

    def handle_starttag(self, tag: str, _attrs: list[tuple[str, str | None]]):
        tag = tag.lower()
        if tag in self.ignored_tags:
            self._ignored_depth += 1
        elif tag == "title":
            self._in_title = True

    def handle_endtag(self, tag: str):
        tag = tag.lower()
        if tag == "title":
            self._in_title = False
        elif tag in self.ignored_tags and self._ignored_depth:
            self._ignored_depth -= 1

    def handle_data(self, data: str):
        if not data or self._ignored_depth:
            return
        if self._in_title:
            self.title_parts.append(data)
        else:
            self.content_parts.append(data)
