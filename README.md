# TikTok Chatbot / TikTok Analyzer

Ứng dụng gồm 2 phần:

- `backend`: FastAPI, crawl/phân tích bình luận TikTok và gọi Gemini để lấy ngữ cảnh video.
- `frontend`: Next.js, giao diện nhập link TikTok, chọn model và xem kết quả phân tích.

## Yêu cầu

- Python 3.10 hoặc mới hơn
- Node.js 20 hoặc mới hơn
- NPM
- Gemini API key

## 1. Cấu hình backend

Vào thư mục backend:

```powershell
cd backend
```

Tạo môi trường Python ảo nếu chưa có:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

Cài thư viện:

```powershell
pip install -r requirements.txt
```

Tạo file `.env` từ file mẫu:

```powershell
Copy-Item .env.example .env
```

Mở `backend/.env` và điền API key:

```env
GEMINI_API_KEY=your_google_gemini_api_key_here
GEMINI_VISION_MODEL=gemini-3-flash-preview
GEMINI_ANALYST_MODEL=gemini-3-flash-preview
```

Nếu máy không cho chạy script PowerShell, chạy lệnh này rồi activate lại venv:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

## 2. Chạy backend

Từ thư mục gốc dự án:

```powershell
python -m uvicorn main:app --host 127.0.0.1 --port 8000 --app-dir backend --reload
```

Kiểm tra backend:

- API root: <http://127.0.0.1:8000>
- Danh sách model: <http://127.0.0.1:8000/models>
- API docs: <http://127.0.0.1:8000/docs>

## 3. Cấu hình frontend

Mở terminal mới, vào thư mục frontend:

```powershell
cd frontend
```

Cài thư viện:

```powershell
npm install
```

Tạo file `frontend/.env.local`:

```env
NEXT_PUBLIC_API_BASE_URL=http://127.0.0.1:8000
```

Biến này giúp frontend gọi đúng backend đang chạy ở port `8000`.

## 4. Chạy frontend

Trong thư mục `frontend`:

```powershell
npm run dev
```

Mở trình duyệt tại:

<http://localhost:3000>

## Cách sử dụng

1. Chạy backend ở terminal thứ nhất.
2. Chạy frontend ở terminal thứ hai.
3. Mở `http://localhost:3000`.
4. Dán link video TikTok.
5. Chọn model nếu cần.
6. Bấm `Analyze` để phân tích bình luận, cảm xúc và ngữ cảnh video.

## Model và dữ liệu

Các checkpoint model đang được đọc từ:

- `backend/model/inference_gr/*.pt`
- `backend/model/sequence_gr/*.pt`

Từ điển ngữ cảnh được đọc từ:

- `backend/assets/dic_final_cleaned.json`

Nếu endpoint `/models` trả về danh sách rỗng, hãy kiểm tra lại các file `.pt` trong hai thư mục model trên.

## Lệnh nhanh

Terminal 1:

```powershell
python -m uvicorn main:app --host 127.0.0.1 --port 8000 --app-dir backend --reload
```

Terminal 2:

```powershell
cd frontend
npm run dev
```

## Lỗi thường gặp

### Frontend không gọi được API

Kiểm tra backend có chạy ở `http://127.0.0.1:8000` chưa, rồi kiểm tra file `frontend/.env.local`:

```env
NEXT_PUBLIC_API_BASE_URL=http://127.0.0.1:8000
```

Sau khi sửa `.env.local`, dừng frontend và chạy lại `npm run dev`.

### Thiếu Gemini API key

Kiểm tra `backend/.env` có biến sau:

```env
GEMINI_API_KEY=your_google_gemini_api_key_here
```

Backend vẫn có thể khởi động khi thiếu key, nhưng phần phân tích ngữ cảnh bằng Gemini sẽ không hoạt động đúng.

### Cài `torch` lâu hoặc lỗi

`torch` và `transformers` khá nặng. Nếu cài lỗi, hãy kiểm tra phiên bản Python và thử cập nhật pip:

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```
$env:JAVA_HOME="C:\Program Files\Java\jdk-21.0.10"
$env:Path="$env:JAVA_HOME\bin;$env:Path"
Test-Path "$env:JAVA_HOME\bin\java.exe"
python -c "import os; print(os.environ.get('JAVA_HOME'))"
# khai báo đường dẫn backend cho python hệ thống
$env:PYTHONPATH = "backend"