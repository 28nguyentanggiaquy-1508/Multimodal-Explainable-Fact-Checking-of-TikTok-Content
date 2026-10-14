"""
Schema notes cho History.

Database tables:

analyses
    1 record / 1 lần bấm Analyze

comments
    N comments thuộc một analysis

aspect_results
    N aspect results thuộc một comment

The defaults below are only fallback values.
services/history_service.py prefers the model versions reported by
the live ABSA runtime, so rollback V2 -> V1 is recorded correctly.
"""


ANALYSIS_MODEL_VERSIONS = {
    "ate": "ate_phobert_v2",
    "asc": "asc_phobert_v2",
}
