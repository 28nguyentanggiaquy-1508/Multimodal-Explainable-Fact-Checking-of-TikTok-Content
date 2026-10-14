# TikTok-MFC Annotation Guideline v1.0 (skeleton)

Core order: source validation -> multimodal review -> atomic claim -> checkable/check-worthy -> evidence review -> verdict -> grounded explanation -> QC.

Important invariant: technical/system failure is never converted to `NEI`. In those cases `gold_verdict` remains `null` and a non-semantic status such as `FAILED`, `INSUFFICIENT_COLLECTION`, or `NEEDS_ADJUDICATION` must be used.
