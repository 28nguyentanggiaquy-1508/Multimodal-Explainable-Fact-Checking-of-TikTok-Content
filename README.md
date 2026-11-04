# Multimodal Explainable Fact-Checking of TikTok Content

> **Đồ án tốt nghiệp:** Kiểm chứng thông tin đa phương thức trên TikTok bằng Vision-Language Models (VLM) và suy luận dựa trên bằng chứng.

## Giới thiệu

Dự án được phát triển từ hệ thống phân tích bình luận/cảm xúc TikTok sang hướng **Multimodal Explainable Fact-Checking**. Hiện backend vẫn giữ hai luồng:

| Luồng | Mô tả | Ghi chú |
| --- | --- | --- |
| **Fact-checking (trọng tâm)** | Phân tích video, trích xuất claims, tìm bằng chứng, kiểm chứng và giải thích kết quả | Backend triển khai các Phase 1–4; tiếp tục nghiên cứu và đánh giá |
| **TikTok Analyzer (legacy)** | Phân tích bình luận, cảm xúc và ngữ cảnh video | Được giữ để tương thích; trang chủ frontend hiện dùng `/analyze` |

Đây là **hệ thống nghiên cứu**, không bảo đảm mọi kết luận của AI đều đúng. Luôn đối chiếu kết quả với nguồn bằng chứng và đánh giá của con người.

## Kiến trúc hệ thống

```text
TikTok URL
   |
   +--> Phase 1: Kiểm tra URL, tạo phiên xử lý
   |
   +--> Phase 2: Phân tích đa phương thức (hình ảnh, video, âm thanh)
   |             và trích xuất phát biểu cần kiểm chứng (claims)
   |
   +--> Phase 3: Tìm kiếm, lọc, xếp hạng bằng chứng
   |
   +--> Phase 4: Kiểm chứng dựa trên bằng chứng
   |             SUPPORTED / REFUTED / NEI + lời giải thích
   |
   +--> Kết quả API và đồng bộ dataset (tùy chọn)
```

**Thành phần:** FastAPI; Multi-VLM (Gemini, OpenAI, local/Ollama); tìm kiếm bằng chứng; cơ chế cache, quota/circuit breaker, checkpoint; các công cụ quản lý và đánh giá dataset. Việc chạy đầy đủ phụ thuộc provider, API key và tài nguyên máy.

**Phân biệt:** Kết luận tự sinh và giải thích của hệ thống **không mặc nhiên là dữ liệu Gold** đã được thẩm định.

## Cấu trúc thư mục chính

```text
.
├── backend/
│   ├── main.py                 # FastAPI entrypoint
│   ├── api/                    # Fact-check API, dataset sync
│   ├── core/vlm/               # Provider, cache, quota, circuit, checkpoint
│   ├── fact_checking/          # Multimedia, claims, evidence, verification
│   ├── dataset_integration/    # Đồng bộ đầu ra vào dataset
│   ├── model/                  # Checkpoint legacy nếu có
│   ├── .env.example            # File mẫu cấu hình
│   └── requirements.txt
├── frontend/
│   ├── app/                    # Next.js UI
│   ├── services/               # API client
│   └── package.json
├── tiktok_mfc_d3/
│   └── dataset/
│       ├── records/            # JSONL
│       ├── schemas/            # JSON Schema
│       ├── guidelines/         # Hướng dẫn gán nhãn
│       └── manifest.json
└── README.md
```

Một số model, dữ liệu runtime, cache và kết quả thực nghiệm có thể không nằm trên GitHub.

## Yêu cầu

- **Python 3.10+**, pip
- **Node.js 20.9+**, npm
- **Windows PowerShell** (ví dụ lệnh bên dưới), Git
- Internet để cài thư viện, tải video và tìm bằng chứng
- **Tùy chọn theo provider:** Gemini / OpenAI / Brave Search API key
- **Nếu chạy local VLM:** Ollama, model đã tải và cấu hình phần cứng phù hợp
- **Nếu xử lý âm thanh yêu cầu:** FFmpeg

Không bắt buộc sử dụng đồng thời mọi provider. Nếu thiếu dịch vụ phụ thuộc, một số phase có thể trả về `DEGRADED` hoặc `UNAVAILABLE`.

## 1. Cài đặt backend

Mở PowerShell tại thư mục gốc dự án, sau đó:

```powershell
cd D:\tiktok_chatbot\tiktok_chatbot
cd backend

python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

Copy-Item .env.example .env
```

Nếu kích hoạt môi trường ảo bị PowerShell chặn, chỉ cho phép script trong phiên terminal đang mở:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\venv\Scripts\Activate.ps1
```

### Cấu hình `backend/.env`

**Không đưa API key thật lên GitHub.** Kiểm tra và thay tất cả giá trị mẫu bằng cấu hình an toàn của riêng bạn. Ví dụ:

```dotenv
# Điền key CHỈ cho provider được sử dụng
GEMINI_API_KEY=
OPENAI_API_KEY=
BRAVE_SEARCH_API_KEY=

# Có thể chọn brave | gemini | brave_then_gemini
EVIDENCE_SEARCH_PROVIDER=brave

# Local VLM: chỉ bật khi Ollama và các model đã sẵn sàng
LOCAL_VLM_ENABLED=true
LOCAL_VLM_API_STYLE=ollama
LOCAL_VLM_BASE_URL=http://127.0.0.1:11434
LOCAL_PHASE2_MODEL=qwen3-vl:4b-instruct-q4_K_M
LOCAL_PHASE4_MODEL=qwen3-vl:8b-instruct-q4_K_M

# Thứ tự fallback ví dụ; điều chỉnh theo cấu hình thực tế
VLM_PROVIDER_ORDER=local,gemini,openai
VLM_VERIFICATION_PROVIDER_ORDER=gemini,local,openai

CORS_ALLOW_ORIGINS=http://127.0.0.1:3000,http://localhost:3000
```

Nếu chạy Ollama, cần cài Ollama, khởi động dịch vụ ở `127.0.0.1:11434` và tải các model đã khai báo. Model Ollama **không được tải bằng pip**. Nếu chạy Gemini/OpenAI/Brave, phải có API key, tên model và quota hợp lệ.

### Chạy backend (khuyến nghị)

Ở **thư mục `backend`** với venv đã kích hoạt:

```powershell
python -m uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```

Chạy từ `backend` giúp tránh lỗi đường dẫn tương đối của tài nguyên legacy. Nếu bắt buộc chạy từ thư mục gốc:

```powershell
cd D:\tiktok_chatbot\tiktok_chatbot
.\backend\venv\Scripts\Activate.ps1
$env:PYTHONPATH = (Resolve-Path .\backend).Path
python -m uvicorn main:app --app-dir backend --host 127.0.0.1 --port 8000 --reload
```

Kiểm tra:
- http://127.0.0.1:8000/ — API root
- http://127.0.0.1:8000/docs — Swagger UI
- http://127.0.0.1:8000/models — Model **legacy**
- http://127.0.0.1:8000/fact-check/health — Fact-check health

## 2. Cài đặt và chạy frontend

Mở **terminal thứ hai**:

```powershell
cd D:\tiktok_chatbot\tiktok_chatbot\frontend
npm install
```

Tạo file `frontend/.env.local`:

```dotenv
NEXT_PUBLIC_API_BASE_URL=http://127.0.0.1:8000
```

Khởi động:

```powershell
npm run dev
```

Mở http://localhost:3000.

**Hiện trạng:** Trang chủ Next.js đang gọi `/analyze` cho hệ thống phân tích bình luận/cảm xúc. Pipeline fact-checking mới đã có ở backend; không nên mặc định rằng giao diện kiểm chứng mới đã hoàn thiện.

## 3. API chính

| Method | Endpoint | Mục đích |
| --- | --- | --- |
| GET | `/` | Kiểm tra API |
| GET | `/docs` | Tài liệu API |
| GET | `/models` | Danh sách model cảm xúc cũ |
| POST | `/analyze` | Phân tích bình luận TikTok (legacy) |
| GET | `/fact-check/health` | Kiểm tra pipeline kiểm chứng |
| POST | `/fact-check/fact-check` | Kiểm chứng đa phương thức |
| POST | `/dataset/sync` | Đồng bộ thủ công đầu ra đã có vào dataset |

**Lưu ý:** `/fact-check/fact-check` là đường dẫn thực tế từ mã nguồn hiện tại (router prefix + route), không phải lỗi ghi nhầm.

Thử endpoint fact-check từ PowerShell (đổi thành URL TikTok thật):

```powershell
$body = @{ url = "https://www.tiktok.com/@example/video/VIDEO_ID"; sync_dataset = $false } | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8000/fact-check/fact-check" -ContentType "application/json" -Body $body
```

`sync_dataset = $false` tránh ghi dữ liệu khi chỉ thử API. Trong mã nguồn hiện tại, mặc định `sync_dataset` là `true`.

## 4. Model và dataset

Checkpoint của luồng phân tích cảm xúc cũ:

- `backend/model/inference_gr/*.pt`
- `backend/model/sequence_gr/*.pt`
- `backend/assets/dic_final_cleaned.json` (từ điển ngữ cảnh legacy)

Nếu `/models` rỗng, kiểm tra các checkpoint legacy này. Chúng **không phải** danh sách các provider VLM của pipeline mới.

Dataset nghiên cứu tại `tiktok_mfc_d3/dataset/` chứa records JSONL, schema, guideline và manifest. Khi đánh giá cần phân biệt **SYSTEM output**, dữ liệu đang thử nghiệm/QA và **Gold do con người thẩm định**. Không suy diễn dữ liệu Gold chỉ từ kết quả AI hoặc QA kỹ thuật.

## 5. Lỗi thường gặp

- **Frontend không gọi được API:** kiểm tra port 8000, `frontend/.env.local`, CORS và khởi động lại `npm run dev`.
- **`/models` trống:** kiểm tra checkpoint `.pt` dành cho pipeline cảm xúc legacy.
- **429 / quota / circuit breaker:** xem hạn mức API, model và thứ tự provider; tránh retry liên tục khi hết quota.
- **Ollama không hoạt động:** kiểm tra máy chủ local, model đã tải và khả năng đáp ứng của RAM/VRAM.
- **Thiếu FFmpeg:** kiểm tra `ffmpeg -version` nếu tác vụ âm thanh báo lỗi.
- **Thiếu Java:** chỉ cấu hình `JAVA_HOME` nếu module cụ thể đang sử dụng yêu cầu Java; đây không phải điều kiện chung để khởi động FastAPI.

Ví dụ kiểm tra JDK trên một máy Windows (đổi theo đường dẫn cài đặt thực tế):

```powershell
$env:JAVA_HOME = "C:\Program Files\Java\jdk-21.0.10"
$env:Path = "$env:JAVA_HOME\bin;$env:Path"
Test-Path "$env:JAVA_HOME\bin\java.exe"
java -version
```

## 6. Bảo mật và tái lập nghiên cứu

- Không commit `.env`, key/token, mật khẩu, thông tin cá nhân hay dữ liệu không có quyền chia sẻ.
- **Kiểm tra lịch sử commit:** xóa file hoặc thêm `.gitignore` không xóa được bí mật đã từng commit.
- Giữ phiên bản dataset, mô hình, cấu hình, điều kiện thực nghiệm và provenance của bằng chứng.
- Không trình bày kết quả tự động thành kết quả đã thẩm định hoặc Gold khi chưa được xác minh.

---

**GitHub:** https://github.com/28nguyentanggiaquy-1508/Multimodal-Explainable-Fact-Checking-of-TikTok-Content
