import bisect
import re
import threading
import unicodedata
import warnings
from pathlib import Path


class VnCoreNLPSegmenter:
    """
    Unified VnCoreNLP segmenter for ATE V2 + ASC V2.

    Fixes:
    1. One shared JVM/model instance per Python process.
    2. Supplementary Unicode/emoji corruption through Java/VnCoreNLP.
    3. Raw-offset alignment differences such as tùy/tuỳ, smart punctuation,
       Unicode normalization and unusual whitespace.
    4. Provides both segment_with_offsets() and segment_text().
    5. Falls back to raw tokenization instead of crashing on a rare
       unalignable comment.

    Returned offsets ALWAYS refer to the original raw text.
    """

    _shared_model = None
    _shared_model_dir = None
    _init_lock = threading.Lock()
    _segment_lock = threading.Lock()

    _ZERO_WIDTH = {
        "\u200b",
        "\u200c",
        "\u200d",
        "\ufeff",
    }

    _PUNCT_CANON = {
        "’": "'",
        "‘": "'",
        "‚": "'",
        "`": "'",
        "´": "'",
        "“": '"',
        "”": '"',
        "„": '"',
        "–": "-",
        "—": "-",
        "−": "-",
        "‐": "-",
        "‑": "-",
        "…": "...",
    }

    def __init__(self, model_dir, strict_alignment=False):
        try:
            import py_vncorenlp
        except ImportError as exc:
            raise RuntimeError(
                "Thiếu py_vncorenlp. Chạy: pip install py_vncorenlp"
            ) from exc

        model_dir = Path(model_dir).resolve()
        jar_file = model_dir / "VnCoreNLP-1.2.jar"

        if not jar_file.exists():
            raise RuntimeError(
                "Chưa có VnCoreNLP model tại: "
                f"{model_dir}\n"
                "Hãy chạy setup_vncorenlp.py trước."
            )

        self.strict_alignment = bool(strict_alignment)
        cls = self.__class__

        # PyJNIus/JVM should only be initialized once per Python process.
        with cls._init_lock:
            if cls._shared_model is None:
                cls._shared_model = py_vncorenlp.VnCoreNLP(
                    annotators=["wseg"],
                    save_dir=str(model_dir),
                )
                cls._shared_model_dir = model_dir
            elif cls._shared_model_dir != model_dir:
                raise RuntimeError(
                    "VnCoreNLP đã được khởi tạo từ thư mục khác trong "
                    "cùng Python process.\n"
                    f"Đang dùng: {cls._shared_model_dir}\n"
                    f"Được yêu cầu: {model_dir}"
                )

        self.model = cls._shared_model

    # ------------------------------------------------------------------
    # Java-safe input
    # ------------------------------------------------------------------

    @staticmethod
    def _safe_text_for_vncorenlp(text):
        """
        Replace Unicode code points above U+FFFF with exactly one space
        ONLY in the copy sent to Java/VnCoreNLP.

        Example:
            raw : Cíu taoo🥺
            safe: Cíu taoo

        The original raw string is retained for returned offsets.
        """
        return "".join(
            ch if ord(ch) <= 0xFFFF else " "
            for ch in text
        )

    # ------------------------------------------------------------------
    # Robust normalization used ONLY for alignment
    # ------------------------------------------------------------------

    @classmethod
    def _fold_char(cls, ch):
        if ch in cls._ZERO_WIDTH:
            return ""

        ch = cls._PUNCT_CANON.get(ch, ch)
        output = []

        for unit in ch:
            if unit in ("đ", "Đ"):
                unit = "d"

            decomposed = unicodedata.normalize("NFD", unit)

            for c in decomposed:
                if unicodedata.combining(c):
                    continue
                output.append(c.lower())

        return "".join(output)

    @classmethod
    def _normalize_with_map(cls, text, *, compact=False):
        chars = []
        index_map = []
        previous_space = False

        for raw_index, ch in enumerate(text):
            if ch in cls._ZERO_WIDTH:
                continue

            if ch.isspace() or ch == "\u00a0":
                if compact:
                    continue

                if not previous_space:
                    chars.append(" ")
                    index_map.append(raw_index)

                previous_space = True
                continue

            previous_space = False
            folded = cls._fold_char(ch)

            for c in folded:
                if compact and not c.isalnum():
                    continue

                chars.append(c)
                index_map.append(raw_index)

        return "".join(chars), index_map

    @staticmethod
    def _trim_raw_span(text, start, end):
        while start < end and text[start].isspace():
            start += 1

        while end > start and text[end - 1].isspace():
            end -= 1

        return start, end

    @classmethod
    def _find_exact(cls, text, surface, cursor):
        parts = [
            part
            for part in re.split(r"\s+", surface.strip())
            if part
        ]

        if not parts:
            return None

        pattern = r"\s+".join(re.escape(part) for part in parts)

        match = re.search(
            pattern,
            text[cursor:],
            flags=re.IGNORECASE,
        )

        if match is None:
            return None

        return cls._trim_raw_span(
            text,
            cursor + match.start(),
            cursor + match.end(),
        )

    @classmethod
    def _find_normalized(cls, text, surface, cursor, *, compact=False):
        normalized_text, raw_map = cls._normalize_with_map(
            text,
            compact=compact,
        )

        normalized_surface, _ = cls._normalize_with_map(
            surface,
            compact=compact,
        )

        if not compact:
            normalized_surface = normalized_surface.strip()

        if not normalized_surface or not raw_map:
            return None

        normalized_cursor = bisect.bisect_left(raw_map, cursor)

        position = normalized_text.find(
            normalized_surface,
            normalized_cursor,
        )

        if position < 0:
            return None

        last_position = position + len(normalized_surface) - 1

        if last_position >= len(raw_map):
            return None

        raw_start = raw_map[position]
        raw_end = raw_map[last_position] + 1

        return cls._trim_raw_span(
            text,
            raw_start,
            raw_end,
        )

    @classmethod
    def _find_surface(cls, text, surface, cursor):
        surface = str(surface or "").strip()

        if not surface:
            raise ValueError("Empty segmented surface.")

        # Tier 1: normal exact alignment.
        result = cls._find_exact(text, surface, cursor)
        if result is not None:
            return result

        # Tier 2: tolerate accent/Unicode/punctuation differences.
        result = cls._find_normalized(
            text,
            surface,
            cursor,
            compact=False,
        )
        if result is not None:
            return result

        # Tier 3: ignore punctuation/whitespace while aligning.
        result = cls._find_normalized(
            text,
            surface,
            cursor,
            compact=True,
        )
        if result is not None:
            return result

        raise ValueError(
            "Không align được VnCoreNLP token với raw text "
            "sau exact + normalized + compact fallback.\n"
            f"token={surface!r}\n"
            f"cursor={cursor}\n"
            f"text={text!r}"
        )

    # ------------------------------------------------------------------
    # Raw fallback / skipped symbols
    # ------------------------------------------------------------------

    @staticmethod
    def _raw_gap_segments(text, start, end, source="raw"):
        if end <= start:
            return []

        chunk = text[start:end]
        segments = []

        for match in re.finditer(
            r"\w+|[^\w\s]+",
            chunk,
            flags=re.UNICODE,
        ):
            raw_start = start + match.start()
            raw_end = start + match.end()
            surface = text[raw_start:raw_end]

            if not surface.strip():
                continue

            segments.append({
                "word_form": surface.replace(" ", "_"),
                "surface": surface,
                "start": raw_start,
                "end": raw_end,
                "source": source,
            })

        return segments

    def _raw_fallback_segments(self, text):
        return self._raw_gap_segments(
            text,
            0,
            len(text),
            source="raw_fallback",
        )

    # ------------------------------------------------------------------
    # VnCoreNLP
    # ------------------------------------------------------------------

    def word_forms(self, text):
        text = str(text or "")

        if not text:
            return []

        # Critical fix for emoji/supplementary Unicode.
        safe_text = self._safe_text_for_vncorenlp(text)

        cls = self.__class__

        with cls._segment_lock:
            sentences = self.model.word_segment(safe_text)

        return [
            word_form
            for sentence in sentences
            for word_form in sentence.split()
            if word_form
        ]

    def segment_with_offsets(self, text):
        text = str(text or "")

        if not text:
            return []

        word_forms = self.word_forms(text)

        tokens = []
        cursor = 0

        try:
            for word_form in word_forms:
                surface = word_form.replace("_", " ")

                start, end = self._find_surface(
                    text,
                    surface,
                    cursor,
                )

                # Keep emoji/symbol/punctuation skipped by VnCoreNLP.
                tokens.extend(
                    self._raw_gap_segments(
                        text,
                        cursor,
                        start,
                        source="raw",
                    )
                )

                tokens.append({
                    "word_form": word_form,
                    "surface": text[start:end],
                    "start": start,
                    "end": end,
                    "source": "vncorenlp",
                })

                cursor = end

            tokens.extend(
                self._raw_gap_segments(
                    text,
                    cursor,
                    len(text),
                    source="raw",
                )
            )

            return tokens

        except ValueError as exc:
            if self.strict_alignment:
                raise

            warnings.warn(
                "VnCoreNLP raw-offset alignment fallback activated. "
                "Pipeline tiếp tục bằng raw tokenization cho comment này. "
                f"Chi tiết: {exc}",
                RuntimeWarning,
            )

            return self._raw_fallback_segments(text)

    def segment_text(self, text):
        """
        Used by ASC V2. Uses exactly the same path as ATE V2.
        """
        segments = self.segment_with_offsets(text)

        return " ".join(
            segment["word_form"]
            for segment in segments
        )
