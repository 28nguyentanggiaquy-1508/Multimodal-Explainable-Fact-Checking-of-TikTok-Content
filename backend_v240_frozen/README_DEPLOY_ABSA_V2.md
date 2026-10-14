# Deploy ABSA V2 Runtime

Patch này chuyển runtime ABSA mặc định từ:

```text
ATE V1 + ASC V1
```

sang:

```text
ATE V2 + ASC V2
```

nhưng vẫn giữ khả năng rollback về V1 bằng biến môi trường.

## File trong patch

```text
backend/services/absa_service.py
backend/services/ai_service.py
backend/services/history_service.py
backend/database/models.py

backup_v1/
├── absa_service_v1.py
└── ai_service_before_v2.py
```

## V2 runtime

ATE:

```text
backend/model/absa/ate_phobert_v2/
```

ASC:

```text
backend/model/absa/asc_phobert_v2/
```

VnCoreNLP:

```text
backend/assets/vncorenlp/VnCoreNLP-1.2.jar
```

Dependency:

```powershell
pip install py_vncorenlp
```

ATE V2 đọc threshold từ:

```text
backend/model/absa/ate_phobert_v2/inference_config.json
```

Nếu file không có / không đọc được, fallback là:

```text
0.45
```

## Vì sao ai_service + history_service cũng được sửa?

Runtime response thêm:

```json
{
  "ate_model_version": "ate_phobert_v2",
  "asc_model_version": "asc_phobert_v2"
}
```

`history_service.py` ưu tiên chính các version này khi lưu SQLite.

Nhờ vậy:

```text
V2 run -> history ghi V2
V1 rollback run -> history ghi V1
```

Compare / Trend tiếp tục phân biệt đúng model version.

## Rollback

V1 model KHÔNG bị xóa.

PowerShell:

```powershell
$env:ABSA_RUNTIME_VERSION="v1"
uvicorn main:app --reload
```

Đưa lại V2:

```powershell
$env:ABSA_RUNTIME_VERSION="v2"
uvicorn main:app --reload
```

Nếu không đặt biến môi trường, mặc định là V2.

## Smoke test sau deploy

Khởi động backend, kiểm tra log:

```text
[ABSA] Runtime: v2
[ABSA] Loading ATE model: ...ate_phobert_v2
[ABSA] Loading ASC model: ...asc_phobert_v2
[ABSA] Ready: {...}
```

Sau đó thử các câu:

```text
camera đẹp
app gì vậy
pin hơi yếu
camera đẹp nhưng pin yếu
máy đẹp nhưng pin không tốt
```

Và chạy một video qua `/analyze`.

Response mới nên có:

```json
{
  "ate_model_version": "ate_phobert_v2",
  "asc_model_version": "asc_phobert_v2",
  "absa_runtime": {
    "runtime_version": "v2",
    "ate_threshold": 0.45,
    "segmentation": "vncorenlp"
  }
}
```

## Lưu ý reproducibility

Frozen evaluation dùng segmentation đã preprocess sẵn để số liệu test lặp lại tuyệt đối.

Runtime thực tế phải xử lý comment mới nên bắt buộc chạy VnCoreNLP trực tiếp.
Trong một comment, patch segment comment đúng một lần rồi tái sử dụng kết quả
cho ATE và ASC để giảm biến động và giảm chi phí xử lý.

## Không cần sửa frontend

Contract cũ vẫn giữ:

```text
comments[].absa
absa_summary
```

Các field version/runtime chỉ là field bổ sung.
