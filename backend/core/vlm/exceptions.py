class VLMError(RuntimeError):
    """Base exception for VLM provider failures."""


class VLMQuotaError(VLMError):
    """Provider quota/rate-limit exhausted."""

    def __init__(self, message: str, retry_after_seconds: float | None = None):
        super().__init__(message)
        self.retry_after_seconds = retry_after_seconds


class VLMUnavailableError(VLMError):
    """Provider cannot be used (missing key, outage, unsupported model, etc.)."""


class VLMResponseError(VLMError):
    """Provider returned a response that could not be parsed/validated."""
