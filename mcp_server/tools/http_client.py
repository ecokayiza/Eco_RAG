from __future__ import annotations

import gzip
import ipaddress
import socket
import ssl
import zlib
from http.client import IncompleteRead
from typing import Any, Mapping
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import HTTPSHandler, ProxyHandler, Request, build_opener, urlopen

REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate",
}

_HTTPS_CONTEXT: ssl.SSLContext | None = None
_DIRECT_OPENER: Any = None


class WebRequestError(RuntimeError):
    """Readable single-request network failure."""


def fetch_text(
    url: str,
    *,
    timeout: float,
    max_bytes: int,
    headers: Mapping[str, str] | None = None,
) -> str:
    """Fetch text with one bounded request and browser-like headers."""
    request_headers = dict(REQUEST_HEADERS)
    request_headers.update(headers or {})
    request = Request(url, headers=request_headers)
    try:
        with _open(request, timeout=timeout, context=_context_for_url(url)) as response:
            body = _read_limited(response, max_bytes=max_bytes)
            body = _decode_transfer_encoding(body, response.headers)
            charset = response.headers.get_content_charset() if hasattr(response.headers, "get_content_charset") else None
            return body.decode(charset or "utf-8", errors="replace")
    except HTTPError as exc:
        raise WebRequestError(_http_error_message(exc, max_bytes=2048)) from exc
    except (URLError, ssl.SSLError, socket.timeout, TimeoutError, IncompleteRead, OSError, zlib.error) as exc:
        raise WebRequestError(_network_error_message(exc, timeout=timeout)) from exc


def _open(request: Request, *, timeout: float, context: ssl.SSLContext | None):
    url = request.full_url
    if _should_bypass_proxy(url):
        return _direct_opener().open(request, timeout=timeout)
    return urlopen(request, timeout=timeout, context=context)


def _direct_opener():
    global _DIRECT_OPENER
    if _DIRECT_OPENER is None:
        _DIRECT_OPENER = build_opener(ProxyHandler({}), HTTPSHandler(context=_https_context()))
    return _DIRECT_OPENER


def _should_bypass_proxy(url: str) -> bool:
    parsed = urlparse(url)
    host = parsed.hostname
    if not host:
        return False
    if host in {"localhost", "127.0.0.1", "::1"}:
        return True
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return False
    return address.is_private or address.is_loopback or address.is_link_local


def _context_for_url(url: str) -> ssl.SSLContext | None:
    if urlparse(url).scheme.lower() != "https":
        return None
    return _https_context()


def _https_context() -> ssl.SSLContext:
    global _HTTPS_CONTEXT
    if _HTTPS_CONTEXT is not None:
        return _HTTPS_CONTEXT
    context = ssl.create_default_context()
    ignore_unexpected_eof = getattr(ssl, "OP_IGNORE_UNEXPECTED_EOF", 0)
    if ignore_unexpected_eof:
        context.options |= ignore_unexpected_eof
    _HTTPS_CONTEXT = context
    return context


def _read_limited(response: Any, *, max_bytes: int) -> bytes:
    limit = max(1, int(max_bytes))
    body = response.read(limit + 1)
    if len(body) > limit:
        raise WebRequestError(f"Response body exceeded {limit} bytes.")
    return body


def _decode_transfer_encoding(body: bytes, headers: Any) -> bytes:
    encoding = str(headers.get("Content-Encoding", "") if headers is not None else "").strip().lower()
    if encoding == "gzip":
        return gzip.decompress(body)
    if encoding == "deflate":
        try:
            return zlib.decompress(body)
        except zlib.error:
            return zlib.decompress(body, -zlib.MAX_WBITS)
    return body


def _http_error_message(exc: HTTPError, *, max_bytes: int) -> str:
    detail = ""
    try:
        body = exc.read(max(0, max_bytes))
    except Exception:
        body = b""
    if body:
        charset = exc.headers.get_content_charset() if hasattr(exc.headers, "get_content_charset") else None
        detail = " ".join(body.decode(charset or "utf-8", errors="replace").split())
        if detail:
            detail = f": {detail[:500]}"
    reason = str(getattr(exc, "reason", "") or "").strip()
    suffix = f" {reason}" if reason else ""
    return f"HTTP {exc.code}{suffix}{detail}".strip()


def _network_error_message(exc: BaseException, *, timeout: float) -> str:
    reason = getattr(exc, "reason", exc)
    text = str(reason or exc)
    if _is_unexpected_tls_eof(reason) or _is_unexpected_tls_eof(text):
        return "TLS connection closed unexpectedly by the server or proxy before HTTPS completed."
    if isinstance(reason, socket.timeout) or isinstance(exc, (socket.timeout, TimeoutError)):
        return f"Timed out after {timeout:g} seconds."
    return text or exc.__class__.__name__


def _is_unexpected_tls_eof(value: Any) -> bool:
    text = str(value or "")
    return "UNEXPECTED_EOF_WHILE_READING" in text or "EOF occurred in violation of protocol" in text
