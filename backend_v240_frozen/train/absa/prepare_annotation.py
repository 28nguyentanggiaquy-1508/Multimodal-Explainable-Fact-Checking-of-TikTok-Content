import json
import random
import re
from pathlib import Path


BACKEND_DIR = Path(__file__).resolve().parents[2]

RAW_FILE = (
    BACKEND_DIR
    / "data"
    / "absa"
    / "raw"
    / "raw_comments.jsonl"
)

OUTPUT_DIR = (
    BACKEND_DIR
    / "data"
    / "absa"
    / "annotated"
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "annotation_candidates.jsonl"
)


# ============================================================
# CONFIG
# ============================================================

RANDOM_SEED = 42

MAX_SAMPLES = 1000

MIN_TEXT_LENGTH = 5
MAX_TEXT_LENGTH = 300


# Một số từ thường cho thấy comment có nhận xét / đánh giá
OPINION_WORDS = [
    # positive
    "đẹp",
    "hay",
    "tốt",
    "ổn",
    "xịn",
    "ngon",
    "thích",
    "yêu",
    "tiện",
    "hiệu quả",
    "mượt",
    "nhanh",
    "rẻ",
    "đáng tiền",
    "dễ dùng",

    # negative
    "xấu",
    "tệ",
    "chán",
    "dở",
    "đắt",
    "mắc",
    "chậm",
    "yếu",
    "lỗi",
    "khó dùng",
    "không tốt",
    "không ổn",
    "phí",
    "ồn",
    "nóng",
]


# Từ thường xuất hiện trong các yêu cầu / chủ thể có thể là aspect.
# Chỉ dùng để ưu tiên candidate, KHÔNG phải danh sách aspect cố định.
ASPECT_HINTS = [
    "giá",
    "camera",
    "pin",
    "màn hình",
    "thiết kế",
    "âm thanh",
    "chất lượng",
    "giao hàng",
    "shop",
    "dịch vụ",
    "sản phẩm",
    "video",
    "máy",
    "app",
    "ứng dụng",
    "cách dùng",
    "hiệu năng",
    "tốc độ",
    "đồ ăn",
    "món",
    "nhân viên",
]


def load_jsonl(path):
    rows = []

    if not path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy: {path}"
        )

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            try:
                rows.append(
                    json.loads(line)
                )

            except json.JSONDecodeError:
                continue

    return rows


def normalize_text(text):
    text = str(text or "")

    text = text.replace(
        "\n",
        " "
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


def vietnamese_score(text):
    """
    Heuristic đơn giản để loại bớt comment hoàn toàn bằng
    tiếng Anh / ngôn ngữ khác.

    Không dùng để phân loại ngôn ngữ chính thức.
    """

    vietnamese_chars = (
        "ăâđêôơư"
        "áàảãạ"
        "ắằẳẵặ"
        "ấầẩẫậ"
        "éèẻẽẹ"
        "ếềểễệ"
        "íìỉĩị"
        "óòỏõọ"
        "ốồổỗộ"
        "ớờởỡợ"
        "úùủũụ"
        "ứừửữự"
        "ýỳỷỹỵ"
    )

    lower = text.lower()

    return sum(
        lower.count(char)
        for char in vietnamese_chars
    )


def is_valid_text(text):

    if len(text) < MIN_TEXT_LENGTH:
        return False

    if len(text) > MAX_TEXT_LENGTH:
        return False

    # URL / spam thuần link
    if text.startswith("http"):
        return False

    # Chỉ ký tự / emoji
    letters = re.findall(
        r"[A-Za-zÀ-ỹ]",
        text
    )

    if len(letters) < 2:
        return False

    return True


def candidate_score(text):

    lower = text.lower()

    score = 0

    for word in OPINION_WORDS:

        if word in lower:
            score += 2

    for word in ASPECT_HINTS:

        if word in lower:
            score += 1

    # Câu có từ nối đối lập thường rất hữu ích cho ABSA
    contrast_words = [
        "nhưng",
        "mà",
        "tuy",
        "còn",
        "trừ",
    ]

    if any(
        word in lower
        for word in contrast_words
    ):
        score += 2

    return score


def build_annotation_row(row):

    return {
        "comment_id": row.get(
            "comment_id"
        ),

        "video_id": row.get(
            "video_id"
        ),

        "text": normalize_text(
            row.get("text")
        ),

        # Người gán nhãn sẽ điền
        "has_aspect": None,

        "aspects": [],

        # trạng thái annotation
        "annotation_status": "pending",
    }


def main():

    random.seed(
        RANDOM_SEED
    )

    rows = load_jsonl(
        RAW_FILE
    )

    print(
        f"Raw comments: {len(rows)}"
    )

    valid = []

    seen_text = set()

    for row in rows:

        text = normalize_text(
            row.get("text")
        )

        if not is_valid_text(text):
            continue

        # Loại duplicate theo text
        key = text.lower()

        if key in seen_text:
            continue

        seen_text.add(key)

        # Comment không có dấu tiếng Việt vẫn có thể là tiếng Việt:
        # "dep qua", "hay qua", ...
        # vì vậy không loại quá mạnh.
        lang_score = vietnamese_score(
            text
        )

        score = candidate_score(
            text
        )

        valid.append({
            "row": row,
            "score": score,
            "lang_score": lang_score,
        })

    print(
        f"Valid comments: {len(valid)}"
    )

    # ========================================================
    # Ưu tiên comment có khả năng chứa aspect + opinion
    # ========================================================

    strong_candidates = [
        item
        for item in valid
        if item["score"] >= 2
    ]

    weak_candidates = [
        item
        for item in valid
        if item["score"] < 2
    ]

    random.shuffle(
        strong_candidates
    )

    random.shuffle(
        weak_candidates
    )

    # 70% candidate mạnh
    strong_limit = int(
        MAX_SAMPLES * 0.7
    )

    selected = (
        strong_candidates[
            :strong_limit
        ]
    )

    remaining = (
        MAX_SAMPLES
        - len(selected)
    )

    # 30% random comment để model học trường hợp không có aspect
    selected.extend(
        weak_candidates[
            :remaining
        ]
    )

    random.shuffle(
        selected
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8",
    ) as f:

        for item in selected:

            annotation = (
                build_annotation_row(
                    item["row"]
                )
            )

            f.write(
                json.dumps(
                    annotation,
                    ensure_ascii=False,
                )
                + "\n"
            )

    print(
        f"Candidates: {len(selected)}"
    )

    print(
        f"Saved to: {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()