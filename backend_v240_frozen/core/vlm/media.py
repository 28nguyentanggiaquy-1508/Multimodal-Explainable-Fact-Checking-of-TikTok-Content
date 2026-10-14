from __future__ import annotations

import math
import tempfile
from pathlib import Path


def sample_video_frames(
    video_path: str | Path,
    *,
    count: int = 10,
    jpeg_quality: int = 85,
) -> list[tuple[Path, float]]:
    """
    Uniformly sample representative frames.

    Returns [(jpeg_path, timestamp_seconds), ...]. The temp directory is kept
    for the lifetime of the process; Phase 2 removes the containing work dir.
    """
    try:
        import cv2  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "opencv-python-headless is required for OpenAI frame sampling"
        ) from exc

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    duration = (total_frames / fps) if total_frames > 0 and fps > 0 else 0.0

    count = max(1, int(count))
    if duration > 0:
        timestamps = [duration * (i + 0.5) / count for i in range(count)]
    elif total_frames > 0:
        timestamps = []
    else:
        timestamps = [0.0]

    out_dir = Path(tempfile.mkdtemp(prefix="multivlm_frames_"))
    results: list[tuple[Path, float]] = []

    if timestamps:
        for idx, ts in enumerate(timestamps):
            cap.set(cv2.CAP_PROP_POS_MSEC, ts * 1000.0)
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            path = out_dir / f"frame_{idx:02d}_{ts:.2f}s.jpg"
            cv2.imwrite(str(path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)])
            results.append((path, ts))
    else:
        step = max(1, math.floor(total_frames / count))
        for idx, frame_no in enumerate(range(0, total_frames, step)):
            if len(results) >= count:
                break
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            ts = frame_no / fps if fps > 0 else float(idx)
            path = out_dir / f"frame_{idx:02d}_{ts:.2f}s.jpg"
            cv2.imwrite(str(path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)])
            results.append((path, ts))

    cap.release()
    if not results:
        raise RuntimeError("No video frames could be sampled")
    return results
