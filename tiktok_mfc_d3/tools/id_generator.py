from __future__ import annotations

from pathlib import Path
from .common import read_json, write_json


class IdGenerator:
    def __init__(self, state_file: Path):
        self.state_file = state_file
        self.state = read_json(state_file)

    def _next(self, key: str) -> int:
        if key not in self.state:
            raise KeyError(f"Unknown ID namespace: {key}")
        self.state[key] += 1
        write_json(self.state_file, self.state)
        return self.state[key]

    def video_id(self) -> str:
        return f"TK_{self._next('TK'):06d}"

    def frame_id(self, video_id: str, index: int) -> str:
        return f"FR_{video_id.removeprefix('TK_')}_{index:03d}"

    def transcript_span_id(self, video_id: str, index: int) -> str:
        return f"TS_{video_id.removeprefix('TK_')}_{index:03d}"

    def claim_id(self, video_id: str, index: int) -> str:
        return f"CL_{video_id.removeprefix('TK_')}_{index:02d}"

    def document_id(self) -> str:
        return f"DOC_{self._next('DOC'):06d}"

    def evidence_id(self, evidence_type: str) -> str:
        evidence_type = evidence_type.upper()
        if evidence_type == "TEXT":
            return f"EV_TXT_{self._next('EV_TXT'):06d}"
        if evidence_type == "IMAGE":
            return f"EV_IMG_{self._next('EV_IMG'):06d}"
        raise ValueError("Only TEXT/IMAGE evidence is supported in v1 tooling")

    def verification_id(self, video_id: str, claim_index: int) -> str:
        return f"VR_{video_id.removeprefix('TK_')}_{claim_index:02d}"

    def explanation_id(self, video_id: str, claim_index: int) -> str:
        return f"EX_{video_id.removeprefix('TK_')}_{claim_index:02d}"

    def annotation_id(self) -> str:
        return f"AN_{self._next('AN'):06d}"
