import pytest

from fact_checking.url_utils import InvalidTikTokUrl, validate_tiktok_url


def test_accepts_tiktok_host():
    url = "https://www.tiktok.com/@demo/video/123"
    assert validate_tiktok_url(url) == url


def test_rejects_non_tiktok_host():
    with pytest.raises(InvalidTikTokUrl):
        validate_tiktok_url("https://example.com/video/123")
