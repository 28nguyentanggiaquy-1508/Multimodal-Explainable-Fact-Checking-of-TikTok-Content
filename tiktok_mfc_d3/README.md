# TikTok-MFC D3 Dataset Tooling

Bộ tooling này triển khai D3 cho TikTok-MFC Dataset Specification v1.0.

## Thành phần

- `dataset/`: cấu trúc dataset chuẩn v1.0.
- `dataset/schemas/`: JSON Schema cho video, claim, evidence, verification, explanation.
- `tools/id_generator.py`: sinh ID ổn định.
- `tools/hash_utils.py`: SHA-256 cho integrity.
- `tools/duplicate_checker.py`: exact hash, text near-duplicate, average hash cho keyframe.
- `tools/contract_guard.py`: chặn anti-pattern `system failure -> NEI`.
- `tools/phase14_adapter.py`: chuyển JSON output Phase 1-4 hiện tại thành candidate dataset records.
- `tools/validate.py`: validate JSONL theo JSON Schema.
- `tools/qc.py`: semantic QC.
- `tools/manifest.py`: cập nhật thống kê manifest.
- `tools/ingest_phase14_response.py`: CLI ingest.
- `tools/run_qc.py`: CLI schema + semantic QC.

## Cài đặt

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Ingest output backend hiện tại

Từ thư mục `tiktok_mfc_d3`:

```powershell
python -m tools.ingest_phase14_response .\response.json --dataset-root .\dataset
```

Tool sẽ:

1. tạo `TK_xxxxxx` và `CL_xxxxxx_xx`;
2. ghi system-generated claim dưới dạng candidate, không coi là Gold;
3. ghi Phase 4 prediction vào `system`;
4. nếu backend trả `NEI` trong khi evidence retrieval thất bại/không có evidence, tự đổi system prediction thành `null`, giữ `raw_verdict=NEI`, gắn `system_status=INSUFFICIENT_COLLECTION` và QC warning;
5. tuyệt đối không tạo Gold verdict tự động.

## Chạy QC

```powershell
python -m tools.run_qc --dataset-root .\dataset
```

## Chạy test

```powershell
python -m unittest discover -s tests -v
```

## Quan trọng

Output của backend Phase 1-4 hiện tại là **SYSTEM OUTPUT**, không phải ground truth. `gold_verdict`, `gold_evidence_ids` và Gold Explanation chỉ được điền sau human annotation/review.

---

## Tích hợp trực tiếp với backend Phase 1-4

Khi thư mục này đặt cùng cấp với `backend/`, dùng bridge trong gói integration:

```text
backend/dataset_integration/bridge.py
```

Bridge thực hiện:

1. lưu nguyên response Phase 1-4 vào `inbox/phase14/`;
2. import claim/evidence/verdict/reasoning dưới dạng **SYSTEM/CANDIDATE**;
3. không tạo Gold annotation;
4. chạy JSON Schema validation + QC;
5. tránh tạo source record trùng khi cùng TikTok được phân tích lại.

Current backend evidence trong `claim[].evidence` được chuyển sang `dataset/records/evidence.jsonl`, nhưng mặc định:

```json
{"is_gold": false}
```

cho đến khi con người review.
