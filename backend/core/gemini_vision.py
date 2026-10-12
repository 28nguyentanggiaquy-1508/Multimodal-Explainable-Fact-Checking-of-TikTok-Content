import json
import mimetypes
import shutil
import tempfile
import time
from pathlib import Path

import requests
from google import genai
from google.genai import types
import yt_dlp

from config import (
    GEMINI_ANALYST_MODEL,
    GEMINI_API_KEY,
    GEMINI_VISION_MODEL,
)
from core.gemini_utils import (
    generate_content_with_retry,
)

gemini_client = (
    genai.Client(
        api_key=GEMINI_API_KEY,
        http_options=types.HttpOptions(
            timeout=90_000,
            retry_options=types.HttpRetryOptions(
                attempts=1,
            ),
        ),
    )
    if GEMINI_API_KEY
    else None
)


ALLOWED_TONES = ("sad", "funny", "angry", "love", "happy", "neutral")
ALLOWED_TONE_SET = set(ALLOWED_TONES)

VIDEO_CACHE_DIR = Path(__file__).resolve().parent.parent / ".video_cache"
VIDEO_CACHE_DIR.mkdir(exist_ok=True)


def _video_id_from_url(url: str):
    import re as _re
    match = _re.search(r"/video/(\d+)", url or "")
    return match.group(1) if match else None


def get_or_download_video(video_url: str, video_id: str = None) -> Path:
    """
    Return a cached mp4 file for the given video URL, downloading if needed.
    Used by the /video proxy endpoint to bypass TikTok CDN 403 in iframe embeds.
    """

    if not video_id:
        video_id = _video_id_from_url(video_url)

    if video_id:
        cached = VIDEO_CACHE_DIR / f"{video_id}.mp4"
        if cached.exists() and cached.stat().st_size > 0:
            return cached

    path, temp_dir = _download_tiktok_video(video_url)

    try:
        if video_id:
            cached = VIDEO_CACHE_DIR / f"{video_id}.mp4"
            shutil.copy2(path, cached)
            return cached
        return path
    finally:
        if video_id and temp_dir:
            shutil.rmtree(temp_dir, ignore_errors=True)

DESKTOP_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)
MOBILE_USER_AGENT = (
    "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) "
    "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 "
    "Mobile/15E148 Safari/604.1"
)


def _gemini_ready() -> bool:
    return gemini_client is not None


def _extract_json(text: str) -> dict:
    if not text:
        return {}

    try:
        return json.loads(text)
    except Exception:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except Exception:
            return {}

    return {}


def _fallback_context(comments=None) -> dict:
    from core.context_engine import extract_video_context

    return extract_video_context(comments or [])


def _build_video_context_prompt(sample_comments: str) -> str:
    return f"""
# VAI TRO
Ban la chuyen gia phan tich video TikTok da phuong thuc cho thi truong Viet Nam.
Ban xem video duoc upload kem theo va tao ra mot doi tuong NGU CANH (context object)
duoc dung lam dau vao cho mo hinh phan loai cam xuc tren binh luan.

# YEU CAU PHAN TICH (theo thu tu uu tien)
1. HINH ANH: nhan vat, boi canh, hanh dong, do hoa, chu chay tren man hinh, do vat.
2. AM THANH: loi noi, am nhac, hieu ung am thanh, tone giong, ngon ngu noi.
3. CAU TRUC NOI DUNG: mo dau (hook), dien bien, cao trao, ket thuc, twist.
4. BIEU CAM va NGON NGU CO THE: net mat, anh mat, cu chi, nang luong tong the.
5. NGU CANH VAN HOA VN: trend dang hot, meme, slang Gen Z, su kien thoi su lien quan.
6. Suy luan LY DO khan gia VN co the phan ung cam xuc (vui, buon, gian, yeu, bat ngo).

# QUY TAC RANG BUOC
- Comments ben duoi CHI la bang chung yeu, KHONG duoc de comments lat nguoc nhung
  gi ban thay ro trong video. Neu comments mau thuan voi video, uu tien video.
- KHONG bia danh tinh, KHONG suy doan chi tiet rieng tu khong duoc hien thi.
- Phan biet ro: FACT (thay/nghe truc tiep) vs INTERPRETATION (suy luan).
- Chon DUNG MOT primary tone trong: sad, funny, angry, love, happy, neutral.
- "scores" la phan phoi xac suat ky vong (0-100, integer) thuoc cam xuc nguoi xem
  bi tac dong boi NGU CANH VIDEO - KHONG phai ket qua phan loai comment.
  Tong cac score nen xap xi 100, va tone primary phai co diem cao nhat.

# DINH DANG OUTPUT
- Tra ve DUY NHAT mot JSON object hop le. KHONG markdown, KHONG giai thich ngoai JSON.
- Tat ca gia tri text bang TIENG VIET, ngan gon, model-friendly.
- "keywords" la 3-6 tu khoa loi mo ta chu de + sac thai (vd: "nhay-mua", "trend-hot",
  "tinh-cam-gia-dinh", "drama", "review-do-an").

# JSON SHAPE
{{
  "topic": "chu de ngan gon cua video, 3-8 tu",
  "tone": "sad | funny | angry | love | happy | neutral",
  "visual_summary": "1-2 cau mo ta nhung gi that su xay ra trong video, neu ro nhan vat va hanh dong chinh",
  "audience_context": "1-2 cau giai thich vi sao nguoi xem VN co the phan ung cam xuc theo cach do",
  "keywords": ["tu_khoa_1", "tu_khoa_2", "tu_khoa_3"],
  "scores": {{
    "sad": 0,
    "funny": 0,
    "angry": 0,
    "love": 0,
    "happy": 0,
    "neutral": 0
  }}
}}

# SAMPLE COMMENTS (bang chung yeu, dung tham khao)
{sample_comments or "(khong co)"}
""".strip()


def _build_emotion_analyst_prompt(
    summary: dict,
    emotions: dict,
    video_context,
    sample_comments: str,
) -> str:
    return f"""
# VAI TRO
Ban la chuyen gia phan tich insight mang xa hoi cho thi truong Viet Nam.
Ban viet bao cao cam xuc cua nguoi xem TikTok cho chu so huu noi dung,
ket hop 3 nguon du lieu:
1. Ngu canh video tu Gemini Vision (hieu nguoi/canh/hanh dong/am thanh).
2. Phan phoi cam xuc tu mo hinh phan loai binh luan.
3. Cac binh luan tieu bieu.

# YEU CAU CHAT LUONG
- BAM SAT ngu canh video lam diem neo. KHONG noi chung chung kieu "hay tang tuong tac".
- Khi neu nhan dinh, NEU BANG CHUNG (vd: "vi visual_summary cho thay X, comment Y noi Z").
- Khi du lieu mau thuan (vd: video happy nhung comment angry cao), HAY GOI TEN
  mau thuan do va dua ra gia thuyet (vd: drama, hieu nham, troll, fanbase doi lap).
- Goi y hanh dong phai CU THE va lien quan toi noi dung cu the cua video nay.

# KIEM TRA LUAN LY (lam tu duy noi bo truoc khi viet)
- Trigger cam xuc chinh trong video la gi?
- Phan phoi positive/neutral/negative noi len dieu gi ve do don nhan?
- Cam xuc fine-grained noi bat (ENJOYMENT, ANGER, SADNESS, ...) lien he the nao
  voi tone cua video?
- Co dau hieu drama, tranh cai, hay mismatch giua y do va phan ung?

# INPUTS
Video context JSON:
{json.dumps(video_context or {}, ensure_ascii=False)}

Sentiment summary:
{json.dumps(summary, ensure_ascii=False)}

Emotion counts:
{json.dumps(emotions, ensure_ascii=False)}

Representative comments:
{sample_comments or "(khong co)"}

# OUTPUT FORMAT (tieng Viet, markdown nhe, ngan gon, tong 200-350 tu)
**Tong quan cam xuc**
- 2-3 cau ve xu huong chung va con so noi bat.

**Ngu canh video**
- 2-3 cau tom tat video co ban (dua tren visual_summary va keywords).

**Hanh vi nguoi xem**
- 2-3 cau giai thich tai sao nguoi xem phan ung nhu vay, co dan chung tu comment.

**Rui ro / tranh cai**
- 1-2 cau ve mau thuan, drama, hoac rui ro thuong hieu (neu co); neu khong co thi noi ro.

**De xuat toi uu**
- 2-3 gach dau dong, moi gach dau dong la 1 hanh dong CU THE cho video tiep theo
  (chu de, hook, cta, audio, ...). Tranh loi khuyen generic.
""".strip()


def _resolve_short_url(video_url: str) -> str:
    """Follow vm.tiktok.com / vt.tiktok.com shortlinks to the canonical URL."""

    if "tiktok.com" not in video_url:
        return video_url

    if "/video/" in video_url:
        return video_url

    try:
        resp = requests.head(
            video_url,
            headers={"User-Agent": DESKTOP_USER_AGENT},
            allow_redirects=True,
            timeout=10,
        )
        return resp.url or video_url
    except Exception:
        return video_url


def _download_via_ytdlp(video_url: str, temp_dir: str) -> Path:
    output_template = str(Path(temp_dir) / "video.%(ext)s")

    base_headers = {
        "User-Agent": DESKTOP_USER_AGENT,
        "Referer": "https://www.tiktok.com/",
        "Accept-Language": "en-US,en;q=0.9,vi;q=0.8",
    }

    attempts = [
        {
            "format": "mp4/best[ext=mp4]/best",
            "http_headers": base_headers,
        },
        {
            "format": "download_addr-0/play_addr-0/best",
            "http_headers": {**base_headers, "User-Agent": MOBILE_USER_AGENT},
        },
        {
            "format": "best",
            "http_headers": base_headers,
            "extractor_args": {"tiktok": {"app_version": ["34.1.2"]}},
        },
    ]

    last_error = None

    for attempt in attempts:
        options = {
            "outtmpl": output_template,
            "merge_output_format": "mp4",
            "noplaylist": True,
            "quiet": True,
            "no_warnings": True,
            "retries": 3,
            "fragment_retries": 3,
            "socket_timeout": 30,
            **attempt,
        }

        try:
            with yt_dlp.YoutubeDL(options) as downloader:
                info = downloader.extract_info(video_url, download=True)
                downloaded = downloader.prepare_filename(info)

            path = Path(downloaded)
            if not path.exists():
                mp4_path = Path(temp_dir) / "video.mp4"
                if mp4_path.exists():
                    path = mp4_path

            if path.exists() and path.stat().st_size > 0:
                return path

        except Exception as exc:
            last_error = exc
            continue

    raise RuntimeError(f"yt-dlp failed to download TikTok video: {last_error}")


def _download_via_tikwm(video_url: str, temp_dir: str) -> Path:
    """Public TikTok mirror API used as fallback when yt-dlp is blocked."""

    api_resp = requests.post(
        "https://www.tikwm.com/api/",
        data={"url": video_url, "hd": "1"},
        headers={"User-Agent": DESKTOP_USER_AGENT},
        timeout=20,
    )
    api_resp.raise_for_status()
    payload = api_resp.json()

    if payload.get("code") != 0:
        raise RuntimeError(f"tikwm error: {payload.get('msg')}")

    data = payload.get("data") or {}
    play_path = data.get("hdplay") or data.get("play") or data.get("wmplay")
    if not play_path:
        raise RuntimeError("tikwm response missing play URL")

    if play_path.startswith("/"):
        play_url = f"https://www.tikwm.com{play_path}"
    else:
        play_url = play_path

    video_resp = requests.get(
        play_url,
        headers={
            "User-Agent": DESKTOP_USER_AGENT,
            "Referer": "https://www.tikwm.com/",
        },
        timeout=60,
        stream=True,
    )
    video_resp.raise_for_status()

    out_path = Path(temp_dir) / "video.mp4"
    with open(out_path, "wb") as fh:
        for chunk in video_resp.iter_content(chunk_size=1 << 16):
            if chunk:
                fh.write(chunk)

    if not out_path.exists() or out_path.stat().st_size == 0:
        raise RuntimeError("tikwm fallback produced empty file")

    return out_path


def _download_tiktok_video(video_url: str):
    temp_dir = tempfile.mkdtemp(prefix="tiktok_video_")
    canonical_url = _resolve_short_url(video_url)

    errors = []

    try:
        path = _download_via_ytdlp(canonical_url, temp_dir)
        return path, temp_dir
    except Exception as exc:
        errors.append(f"yt-dlp: {exc}")
        print(f"[Gemini Vision] yt-dlp failed, trying fallback. {exc}")

    try:
        path = _download_via_tikwm(canonical_url, temp_dir)
        return path, temp_dir
    except Exception as exc:
        errors.append(f"tikwm: {exc}")
        print(f"[Gemini Vision] tikwm fallback failed. {exc}")

    shutil.rmtree(temp_dir, ignore_errors=True)
    raise RuntimeError(
        "Could not download TikTok video. Tried: " + " | ".join(errors)
    )


def _response_text(response) -> str:
    """Extract text from a Gemini response, handling thought_signature parts."""

    text = getattr(response, "text", None)
    if text:
        return text

    chunks = []
    for candidate in getattr(response, "candidates", None) or []:
        content = getattr(candidate, "content", None)
        for part in getattr(content, "parts", None) or []:
            value = getattr(part, "text", None)
            if value:
                chunks.append(value)

    return "".join(chunks)


def _upload_video_to_gemini(video_path: Path):
    mime_type = mimetypes.guess_type(video_path.name)[0] or "video/mp4"

    uploaded_file = gemini_client.files.upload(
        file=str(video_path),
        config=types.UploadFileConfig(mime_type=mime_type),
    )

    for _ in range(60):
        current = gemini_client.files.get(name=uploaded_file.name)
        state = getattr(current, "state", None)
        state_name = getattr(state, "name", str(state))

        if state_name == "ACTIVE":
            return current

        if state_name == "FAILED":
            raise RuntimeError("Gemini video processing failed")

        time.sleep(1)

    raise TimeoutError("Gemini video processing timed out")


def analyze_tiktok_video_context(video_url: str, comments=None) -> dict:
    """
    Use Gemini Flash vision to understand the TikTok video and return context
    for emotion prediction.
    """

    if not _gemini_ready():
        return _fallback_context(comments)

    temp_dir = None
    uploaded_file = None

    try:
        video_path, temp_dir = _download_tiktok_video(video_url)

        # Cache the file so the frontend /video proxy can stream it back
        # without TikTok CDN's 403 referer block.
        vid = _video_id_from_url(video_url)
        if vid:
            try:
                shutil.copy2(video_path, VIDEO_CACHE_DIR / f"{vid}.mp4")
            except Exception as cache_exc:
                print(f"[Gemini Vision] cache copy failed: {cache_exc}")

        uploaded_file = _upload_video_to_gemini(video_path)

        sample_comments = "\n".join((comments or [])[:20])
        prompt = _build_video_context_prompt(sample_comments)

        response = gemini_client.models.generate_content(
            model=GEMINI_VISION_MODEL,
            contents=[prompt, uploaded_file],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
            ),
        )

        data = _extract_json(_response_text(response))
        if not data:
            print("[Gemini Vision] empty/invalid JSON, falling back")
            return _fallback_context(comments)

        tone = str(data.get("tone", "neutral")).lower().strip()
        if tone not in ALLOWED_TONE_SET:
            tone = "neutral"

        raw_scores = data.get("scores")
        normalized_scores = {t: 0 for t in ALLOWED_TONES}
        if isinstance(raw_scores, dict):
            for key, value in raw_scores.items():
                k = str(key).lower().strip()
                if k in ALLOWED_TONE_SET:
                    try:
                        normalized_scores[k] = max(0, min(100, int(round(float(value)))))
                    except (TypeError, ValueError):
                        continue
        if sum(normalized_scores.values()) == 0:
            normalized_scores[tone] = 100
        else:
            top_tone = max(ALLOWED_TONES, key=lambda key: normalized_scores[key])
            if normalized_scores[top_tone] > normalized_scores[tone]:
                tone = top_tone

        raw_keywords = data.get("keywords")
        keywords = []
        if isinstance(raw_keywords, list):
            keywords = [str(k).strip() for k in raw_keywords if str(k).strip()][:8]

        return {
            "topic": str(data.get("topic") or "Khong xac dinh").strip(),
            "tone": tone,
            "scores": normalized_scores,
            "visual_summary": str(data.get("visual_summary") or "").strip(),
            "audience_context": str(data.get("audience_context") or "").strip(),
            "keywords": keywords,
            "source": "gemini_flash_vision",
        }

    except Exception as exc:
        import traceback
        print(f"[Gemini Vision Error] {type(exc).__name__}: {exc}")
        traceback.print_exc()
        context = _fallback_context(comments)
        context["source"] = "local_fallback"
        context["error"] = f"{type(exc).__name__}: {exc}"
        return context

    finally:
        if uploaded_file is not None:
            try:
                gemini_client.files.delete(name=uploaded_file.name)
            except Exception:
                pass

        if temp_dir:
            shutil.rmtree(temp_dir, ignore_errors=True)

# ============================================================
# LEGACY PUBLIC COMPATIBILITY API
# ============================================================

def build_local_emotion_report(
    summary: dict,
    emotions: dict,
    video_context=None,
    comments=None,
    **kwargs,
) -> str:
    """
    Compatibility wrapper cho các service legacy.

    Không thuộc pipeline Fact-Checking V2.
    """

    return _local_emotion_report(
        summary=summary or {},
        emotions=emotions or {},
        video_context=video_context,
    )


def generate_emotion_analyst_report(
    summary: dict,
    emotions: dict,
    video_context=None,
    comments=None,
    **kwargs,
) -> str:
    """
    Compatibility wrapper cho:
        services/analyst_cache_service.py

    Tái sử dụng Gemini analyst cũ thông qua analyze_video_context().
    """

    return analyze_video_context(
        summary=summary or {},
        emotions=emotions or {},
        video_context=video_context,
        comments=comments or [],
    )

def _local_emotion_report(summary: dict, emotions: dict, video_context=None) -> str:
    pos = summary.get("positive", 0)
    neu = summary.get("neutral", 0)
    neg = summary.get("negative", 0)
    total = max(pos + neu + neg, 1)

    pos_ratio = pos / total
    neg_ratio = neg / total

    if pos_ratio >= 0.65:
        trend = "rat tich cuc"
    elif pos_ratio >= 0.5:
        trend = "tich cuc"
    elif neg_ratio >= 0.55:
        trend = "tieu cuc manh"
    elif neg_ratio >= 0.4:
        trend = "kha tieu cuc"
    else:
        trend = "phan hoa / trung lap"

    dominant_emotion = max(emotions, key=emotions.get) if emotions else "OTHER"
    context_text = ""

    if isinstance(video_context, dict):
        context_text = (
            video_context.get("visual_summary")
            or video_context.get("audience_context")
            or video_context.get("topic")
            or ""
        )

    return f"""
TONG QUAN CAM XUC

Video dang co xu huong {trend}.

Positive: {pos} comments
Neutral: {neu} comments
Negative: {neg} comments

Cam xuc noi bat: {dominant_emotion}
Ngu canh video: {context_text}

Ty le phan hoi tieu cuc: {round(neg_ratio * 100, 1)}%

De xuat: tiep tuc theo doi cac nhom binh luan noi bat va dieu chinh noi dung theo phan ung cua nguoi xem.
""".strip()

def analyze_video_context(
    summary: dict,
    emotions: dict,
    video_context=None,
    comments=None,
) -> str:
    """
    Use Gemini to write an analyst report about viewer emotions.
    """

    if not _gemini_ready():
        return _local_emotion_report(summary, emotions, video_context)

    try:
        sample_comments = "\n".join((comments or [])[:30])
        prompt = _build_emotion_analyst_prompt(
            summary=summary,
            emotions=emotions,
            video_context=video_context,
            sample_comments=sample_comments,
        )

        response = gemini_client.models.generate_content(
            model=GEMINI_ANALYST_MODEL,
            contents=prompt,
        )

        text = _response_text(response).strip()
        if text:
            return text

    except Exception as exc:
        print(f"[Gemini Analyst Error] {exc}")

    return _local_emotion_report(summary, emotions, video_context)
# ============================================================================
# FACT-CHECKING V2: MULTIMEDIA ANALYSIS + CLAIM EXTRACTION
# ============================================================================
# This section is intentionally isolated from the legacy emotion pipeline.
# The legacy functions above stay unchanged and continue to serve /analyze.


def _extract_tiktok_post_metadata(video_url: str) -> dict:
    """
    Best-effort extraction of TikTok post metadata without downloading media.
    The post caption is treated as one possible textual modality.
    """
    canonical_url = _resolve_short_url(video_url)

    options = {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
        "noplaylist": True,
        "http_headers": {
            "User-Agent": DESKTOP_USER_AGENT,
            "Referer": "https://www.tiktok.com/",
            "Accept-Language": "vi-VN,vi;q=0.9,en-US;q=0.8,en;q=0.7",
        },
    }

    try:
        with yt_dlp.YoutubeDL(options) as downloader:
            info = downloader.extract_info(canonical_url, download=False) or {}

        caption = (
            info.get("description")
            or info.get("title")
            or ""
        )

        return {
            "caption": str(caption).strip(),
            "uploader": str(info.get("uploader") or "").strip(),
            "uploader_id": str(info.get("uploader_id") or "").strip(),
            "upload_date": str(info.get("upload_date") or "").strip(),
            "duration": info.get("duration"),
            "webpage_url": str(info.get("webpage_url") or canonical_url).strip(),
        }
    except Exception as exc:
        print(f"[FactCheck Metadata] could not read TikTok metadata: {exc}")
        return {
            "caption": "",
            "uploader": "",
            "uploader_id": "",
            "upload_date": "",
            "duration": None,
            "webpage_url": canonical_url,
        }


def _build_fact_check_multimedia_prompt(post_metadata: dict | None = None) -> str:
    """
    Phase 2 prompt for multimedia observation + claim extraction.

    Important separation of responsibilities:
    - always analyze observable video modalities first;
    - only then decide whether a checkable/check-worthy claim exists;
    - never perform evidence retrieval or verdict prediction here.
    """
    post_metadata = post_metadata or {}
    caption_text = str(post_metadata.get("caption") or "").strip() or "(khong co caption metadata)"
    upload_date = str(post_metadata.get("upload_date") or "").strip() or "(khong co)"
    uploader = str(post_metadata.get("uploader") or "").strip() or "(khong co)"
    duration = post_metadata.get("duration")

    return f"""
# VAI TRO

Ban la module Multimedia Claim Extraction trong he thong
Multimodal Automated Fact-Checking cho video TikTok.

BUOC NAY CO HAI NHIEM VU TACH BIET:
A. BAT BUOC phan tich multimedia trong video.
B. Sau do moi xac dinh co factual claim can kiem chung hay khong.

KHONG DUOC fact-check va KHONG DUOC dua ra verdict.


# A. MULTIMEDIA EXTRACTION - BAT BUOC

Ngay ca khi video KHONG co factual claim can kiem chung,
ban VAN PHAI phan tich noi dung nhin/nghe/doc duoc.

1. VISUAL
- nguoi, nhan vat, vat the, boi canh, hanh dong;
- do hoa, bieu do, anh chen, canh phim/anime/game neu co;
- chi mo ta nhung gi quan sat duoc.

2. SPEECH
- trich loi noi co the nghe duoc;
- ghi dung y nghia;
- neu khong nghe ro thi khong tu doan.

3. OCR
- chu tren man hinh;
- phu de, tieu de, ten rieng, dia diem, ngay thang, so lieu;
- neu khong doc duoc thi khong tu tao.

4. CAPTION / POST TEXT
- caption metadata duoc cung cap ben duoi;
- day la mot modality text rieng;
- khong coi caption la su that chi vi no xuat hien.

5. CROSS-MODAL
- neu loi noi, OCR, caption va hinh anh cung bo tro hoac mau thuan nhau,
  ghi ngan gon vao cross_modal_notes.
- KHONG suy doan mot ngay/thang la loi, la tuong lai hay sai boi canh neu metadata
  khong cung cap can cu truc tiep. Khi so sanh thoi gian, chi dung metadata ben duoi.

QUY TAC BAT BUOC:
- candidate_claims CO THE la [].
- NHUNG khong duoc de dong thoi visual_observations,
  speech_segments va ocr_segments deu rong neu video thuc su
  co noi dung nhin/nghe/doc duoc.
- Neu mot modality thuc su khong co du lieu, modality do co the rong.


# CONTENT TYPE

Phan loai noi dung vao DUNG MOT gia tri:
- real_world: noi dung chu yeu noi ve su kien/thuc te doi thuc;
- fictional: phim, anime, truyen hu cau, fan-edit, parody, gameplay
  hoac noi dung hu cau ro rang;
- mixed: ket hop noi dung hu cau va factual claim doi thuc;
- uncertain: khong du thong tin de phan loai chac chan.

Neu video la fictional:
- van phai phan tich visual/speech/OCR day du;
- KHONG bien tinh tiet hu cau thanh factual claim ngoai doi thuc.


# FACT VA INTERPRETATION

FACT_OBSERVED:
nhung gi co the truc tiep nhin thay, nghe thay hoac doc duoc.

INTERPRETATION:
suy luan cua mo hinh.

KHONG bien INTERPRETATION thanh factual claim.


# B. CLAIM DETECTION / EXTRACTION

Chi dua vao candidate_claims neu noi dung:
- mang tinh factual;
- ve nguyen tac co the xac minh bang bang chung;
- co y nghia kiem chung trong boi canh doi thuc;
- khong chi la cam xuc, y kien ca nhan, cau dua, loi keu goi;
- neu claim ngan ngam tu nhieu modality, ghi source_modality = "cross_modal".

Cac nhom claim:
- statistical:
  so lieu, ty le, so luong;
- event_place_person:
  su kien, dia diem, nhan vat, to chuc;
- other_factual:
  factual claim khac;
- authenticity_context:
  claim ve nguon goc, thoi gian, dia diem, tinh xac thuc
  hoac boi canh cua anh/video.

Neu khong co claim du kha nang kiem chung:
candidate_claims = [].


# CAM TUYET DOI

KHONG DUOC:
- ket luan SUPPORTED;
- ket luan REFUTED;
- ket luan NEI;
- tao external evidence;
- dung kien thuc nen de kiem chung claim;
- bia danh tinh;
- suy doan thong tin rieng tu;
- coi caption/OCR/loi noi la su that chi vi no xuat hien.


# POST METADATA

Caption: {caption_text}
Uploader: {uploader}
Upload date (YYYYMMDD neu co): {upload_date}
Duration seconds: {duration}

Metadata chi la thong tin nguon/boi canh; khong tu dong chung minh claim dung.

# OUTPUT

Tra ve DUY NHAT JSON hop le:

{{
  "content_type": "real_world | fictional | mixed | uncertain",

  "visual_observations": [
    {{
      "timestamp": "00:00-00:05",
      "fact": "..."
    }}
  ],

  "speech_segments": [
    {{
      "timestamp": "00:00-00:05",
      "text": "..."
    }}
  ],

  "speech_transcript": "...",

  "ocr_segments": [
    {{
      "timestamp": "00:00-00:05",
      "text": "..."
    }}
  ],

  "ocr_text": "...",

  "caption": "...",

  "candidate_claims": [
    {{
      "id": "claim_01",
      "text": "...",
      "source_modality": "speech | ocr | caption | visual | cross_modal",
      "timestamp": "00:00-00:05",
      "checkable": true,
      "check_worthy": true,
      "claim_type": "statistical | event_place_person | other_factual | authenticity_context",
      "rationale": "..."
    }}
  ],

  "cross_modal_notes": [
    "..."
  ]
}}
""".strip()


def _normalize_fact_check_payload(data: dict, metadata: dict) -> dict:
    """Defensive normalization before Pydantic validation in Phase 2."""
    if not isinstance(data, dict):
        data = {}

    def _list_of_dicts(value):
        if not isinstance(value, list):
            return []
        return [item for item in value if isinstance(item, dict)]

    def _list_of_strings(value):
        if not isinstance(value, list):
            return []
        return [str(item).strip() for item in value if str(item).strip()]

    content_type = str(
        data.get("content_type") or "uncertain"
    ).strip().lower()

    if content_type not in {
        "real_world",
        "fictional",
        "mixed",
        "uncertain",
    }:
        content_type = "uncertain"

    caption = str(
        data.get("caption")
        or metadata.get("caption")
        or ""
    ).strip()

    return {
        "content_type": content_type,
        "visual_observations": _list_of_dicts(data.get("visual_observations")),
        "speech_segments": _list_of_dicts(data.get("speech_segments")),
        "speech_transcript": str(data.get("speech_transcript") or "").strip(),
        "ocr_segments": _list_of_dicts(data.get("ocr_segments")),
        "ocr_text": str(data.get("ocr_text") or "").strip(),
        "caption": caption or None,
        "candidate_claims": _list_of_dicts(data.get("candidate_claims")),
        "cross_modal_notes": _list_of_strings(data.get("cross_modal_notes")),
        "metadata": metadata,
    }

def _build_caption_fallback_claims(
    caption: str,
) -> list[dict]:
    """
    Fallback nhẹ khi VLM/Gemini tạm thời không khả dụng.

    Chỉ tạo candidate claim khi caption có dấu hiệu factual rõ ràng.
    Hàm này không thay thế claim extractor chính và không đưa ra verdict.
    """

    import re

    caption = str(caption or "").strip()
    if not caption:
        return []

    # Bỏ hashtag để claim sạch hơn.
    cleaned = re.sub(r"#\S+", " ", caption)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    if len(cleaned) < 10:
        return []

    # Heuristic tối thiểu cho fallback.
    # Đây chỉ là cơ chế dự phòng khi VLM lỗi, không phải bộ phân loại claim chính.
    factual_patterns = [
        r"\bhiện\s+có\s+\d+",
        r"\bcó\s+\d+\b",
        r"\bnăm\s+\d{4}\b",
        r"\bngày\s+\d{1,2}[/-]\d{1,2}",
        r"\bđược\s+công\s+bố\b",
        r"\bđược\s+ban\s+hành\b",
        r"\bcông\s+bố\s+rằng\b",
    ]

    is_factual = any(
        re.search(pattern, cleaned, flags=re.IGNORECASE)
        for pattern in factual_patterns
    )

    if not is_factual:
        return []

    claim_type = (
        "statistical"
        if re.search(r"\d", cleaned)
        else "other_factual"
    )

    return [
        {
            "id": "claim_caption_01",
            "text": cleaned,
            "source_modality": "caption",
            "timestamp": None,
            "checkable": True,
            "check_worthy": True,
            "claim_type": claim_type,
            "rationale": (
                "Fallback claim được trích trực tiếp từ caption TikTok "
                "khi VLM video tạm thời không khả dụng."
            ),
        }
    ]


def _build_multimedia_fallback_result(
    metadata: dict,
    *,
    source: str,
    error: str,
) -> dict:
    """
    Tạo response fallback nhất quán cho Phase 2.

    Nếu caption có factual claim rõ ràng, pipeline vẫn có thể tiếp tục sang
    Evidence Retrieval dù visual/speech/OCR chưa có vì Gemini bị lỗi.
    """

    result = _normalize_fact_check_payload({}, metadata)

    caption = str(metadata.get("caption") or "").strip()
    fallback_claims = _build_caption_fallback_claims(caption)

    if fallback_claims:
        result["candidate_claims"] = fallback_claims
        result["source"] = "caption_fallback"
        result["cross_modal_notes"] = [
            "VLM video không khả dụng; candidate claim được lấy từ caption metadata."
        ]
        result.setdefault("metadata", {})[
            "caption_claim_fallback_used"
        ] = True
        result.setdefault("metadata", {})[
            "multimedia_failure_source"
        ] = source
    else:
        result["source"] = source

    result["error"] = error
    return result


def analyze_tiktok_multimedia(
    video_url: str,
) -> dict:
    """
    Phase 2 - Multimedia Analysis + Claim Extraction.

    TikTok URL
        -> Visual
        -> Speech
        -> OCR
        -> Caption
        -> Candidate Claims

    Không thực hiện:
    - evidence retrieval
    - verdict prediction
    - explanation generation
    """

    metadata = _extract_tiktok_post_metadata(video_url)

    # ========================================================
    # GEMINI NOT AVAILABLE
    # ========================================================
    if not _gemini_ready():
        return _build_multimedia_fallback_result(
            metadata,
            source="gemini_unavailable",
            error="GEMINI_API_KEY is missing",
        )

    temp_dir = None
    uploaded_file = None

    try:
        # ====================================================
        # 1. DOWNLOAD VIDEO
        # ====================================================
        video_path, temp_dir = _download_tiktok_video(video_url)

        # ====================================================
        # 2. CACHE VIDEO
        # ====================================================
        video_id = _video_id_from_url(video_url)

        if video_id:
            try:
                shutil.copy2(
                    video_path,
                    VIDEO_CACHE_DIR / f"{video_id}.mp4",
                )
            except Exception as cache_exc:
                print(
                    "[FactCheck Multimedia] cache copy failed: "
                    f"{cache_exc}"
                )

        # ====================================================
        # 3. UPLOAD VIDEO TO GEMINI
        # ====================================================
        uploaded_file = _upload_video_to_gemini(video_path)

        # ====================================================
        # 4. BUILD PROMPT
        # ====================================================
        prompt = _build_fact_check_multimedia_prompt(metadata)

        # ====================================================
        # 5. GEMINI + RETRY
        # ====================================================
        retry_result = generate_content_with_retry(
            gemini_client,
            model=GEMINI_VISION_MODEL,
            contents=[prompt, uploaded_file],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
            ),
        )

        # Defensive check để tránh lỗi biến chưa được khởi tạo nếu helper
        # bị sửa sai ở nơi khác.
        if (
            not isinstance(retry_result, tuple)
            or len(retry_result) != 2
        ):
            raise RuntimeError(
                "generate_content_with_retry() must return "
                "(response, model_used)."
            )

        response, model_used = retry_result

        if response is None:
            raise RuntimeError("Gemini returned None response.")

        # ====================================================
        # 6. PARSE GEMINI RESPONSE
        # ====================================================
        response_text = _response_text(response)
        data = _extract_json(response_text)

        if not data:
            raise RuntimeError(
                "Gemini returned empty or invalid multimedia JSON."
            )

        # ====================================================
        # 7. NORMALIZE
        # ====================================================
        result = _normalize_fact_check_payload(data, metadata)

        # Nếu VLM thành công nhưng bỏ sót một factual claim rất rõ trong
        # caption, dùng caption heuristic làm fallback bổ sung.
        if not result.get("candidate_claims"):
            caption_claims = _build_caption_fallback_claims(
                result.get("caption") or ""
            )
            if caption_claims:
                result["candidate_claims"] = caption_claims
                result.setdefault("cross_modal_notes", []).append(
                    "VLM không tạo candidate claim; hệ thống bổ sung một "
                    "candidate claim factual rõ ràng từ caption metadata."
                )
                result.setdefault("metadata", {})[
                    "caption_claim_fallback_used"
                ] = True

        result["source"] = "gemini_multimodal"
        result["model"] = str(model_used or GEMINI_VISION_MODEL)
        result["error"] = None

        print(
            "[FactCheck Multimedia] "
            f"OK model={result['model']} "
            f"claims={len(result.get('candidate_claims', []))}"
        )

        return result

    # ========================================================
    # ERROR FALLBACK
    # ========================================================
    except Exception as exc:
        import traceback

        print(
            "[FactCheck Multimedia Error] "
            f"{type(exc).__name__}: {exc}"
        )
        traceback.print_exc()

        return _build_multimedia_fallback_result(
            metadata,
            source="multimedia_error",
            error=f"{type(exc).__name__}: {exc}",
        )

    # ========================================================
    # CLEANUP
    # ========================================================
    finally:
        if uploaded_file is not None:
            try:
                gemini_client.files.delete(name=uploaded_file.name)
            except Exception:
                pass

        if temp_dir:
            shutil.rmtree(temp_dir, ignore_errors=True)
