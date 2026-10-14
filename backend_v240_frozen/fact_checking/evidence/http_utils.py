from __future__ import annotations

import ipaddress
import socket
from urllib.parse import urljoin, urlparse

import requests


REDIRECT_CODES = {301, 302, 303, 307, 308}


def _is_public_ip(ip_text: str) -> bool:
    ip = ipaddress.ip_address(ip_text)
    return bool(ip.is_global)


def is_safe_public_http_url(url: str) -> bool:
    """
    Basic SSRF protection for URLs discovered from external search results.
    """
    try:
        parsed = urlparse(url)

        if parsed.scheme not in {"http", "https"}:
            return False

        host = parsed.hostname

        if not host or host.lower() == "localhost":
            return False

        # Literal IP.
        try:
            return _is_public_ip(host)
        except ValueError:
            pass

        # DNS hostname.
        infos = socket.getaddrinfo(
            host,
            parsed.port or (443 if parsed.scheme == "https" else 80),
            type=socket.SOCK_STREAM,
        )

        addresses = {info[4][0] for info in infos}

        if not addresses:
            return False

        return all(_is_public_ip(address) for address in addresses)

    except Exception:
        return False


def safe_get(
    url: str,
    *,
    timeout: float,
    headers: dict | None = None,
    stream: bool = False,
    max_redirects: int = 4,
) -> requests.Response:
    """
    GET with redirect-by-redirect public-host validation.
    """
    current = url

    for _ in range(max_redirects + 1):
        if not is_safe_public_http_url(current):
            raise ValueError("Blocked non-public or invalid URL")

        response = requests.get(
            current,
            headers=headers,
            timeout=timeout,
            stream=stream,
            allow_redirects=False,
        )

        if response.status_code not in REDIRECT_CODES:
            return response

        location = response.headers.get("location")
        response.close()

        if not location:
            raise requests.RequestException("Redirect response without Location")

        current = urljoin(current, location)

    raise requests.TooManyRedirects("Too many redirects")


def read_limited_bytes(
    response: requests.Response,
    max_bytes: int,
) -> bytes:
    content_length = response.headers.get("content-length")
    parsed_length = None

    if content_length:
        try:
            parsed_length = int(content_length)
        except (TypeError, ValueError):
            parsed_length = None

    if parsed_length is not None and parsed_length > max_bytes:
        raise ValueError("Remote object is larger than the configured limit")

    chunks = []
    total = 0

    for chunk in response.iter_content(chunk_size=65536):
        if not chunk:
            continue

        total += len(chunk)

        if total > max_bytes:
            raise ValueError("Remote object exceeded the configured byte limit")

        chunks.append(chunk)

    return b"".join(chunks)