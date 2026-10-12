import core.gemini_utils as gu


class ReadTimeout(Exception):
    pass


def test_read_timeout_is_retryable():
    assert gu._is_retryable(ReadTimeout("The read operation timed out")) is True


def test_hard_quota_is_not_retryable():
    exc = RuntimeError("429 RESOURCE_EXHAUSTED exceeded your current quota; check your plan and billing details")
    assert gu._is_hard_quota_error(exc) is True
    assert gu._is_retryable(exc) is False
