from __future__ import annotations

from difflib import SequenceMatcher
from pathlib import Path
from typing import Iterable

from .hash_utils import sha256_file


def exact_video_duplicate(candidate: Path, known_hashes: Iterable[str]) -> tuple[bool, str]:
    digest = sha256_file(candidate)
    return digest in set(known_hashes), digest


def text_near_duplicate(a: str, b: str, threshold: float = 0.92) -> tuple[bool, float]:
    score = SequenceMatcher(None, a.casefold().strip(), b.casefold().strip()).ratio()
    return score >= threshold, score


def average_hash(image_path: Path, hash_size: int = 8) -> str:
    """Simple perceptual hash for near-duplicate keyframes."""
    from PIL import Image

    with Image.open(image_path) as img:
        img = img.convert("L").resize((hash_size, hash_size))
        pixels = list(img.getdata())
    avg = sum(pixels) / len(pixels)
    bits = "".join("1" if p >= avg else "0" for p in pixels)
    return f"{int(bits, 2):0{hash_size * hash_size // 4}x}"


def hamming_hex(a: str, b: str) -> int:
    if len(a) != len(b):
        raise ValueError("Hashes must have equal length")
    return (int(a, 16) ^ int(b, 16)).bit_count()
