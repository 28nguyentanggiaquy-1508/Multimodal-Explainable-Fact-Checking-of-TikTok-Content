import json
import sys
import time
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

BACKEND_DIR = Path(__file__).resolve().parents[2]

if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))


from core.crawler import get_all_comments


CURRENT_DIR = Path(__file__).resolve().parent

URL_FILE = CURRENT_DIR / "video_urls.txt"

OUTPUT_DIR = BACKEND_DIR / "data" / "absa" / "raw"
OUTPUT_FILE = OUTPUT_DIR / "raw_comments.jsonl"


# ============================================================
# CONFIG
# ============================================================

COMMENTS_PER_VIDEO = 200

SLEEP_BETWEEN_VIDEOS = 2


# ============================================================
# LOAD URLS
# ============================================================

def load_urls():
    if not URL_FILE.exists():
        raise FileNotFoundError(
            f"Không tìm thấy file: {URL_FILE}"
        )

    urls = []

    with open(URL_FILE, "r", encoding="utf-8") as f:
        for line in f:
            url = line.strip()

            if not url:
                continue

            if url.startswith("#"):
                continue

            urls.append(url)

    return urls


# ============================================================
# LOAD EXISTING IDS
# ============================================================

def load_existing_comment_ids():
    ids = set()

    if not OUTPUT_FILE.exists():
        return ids

    with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            try:
                row = json.loads(line)

                comment_id = str(
                    row.get("comment_id") or ""
                )

                if comment_id:
                    ids.add(comment_id)

            except json.JSONDecodeError:
                continue

    return ids


# ============================================================
# SAVE COMMENT
# ============================================================

def build_dataset_row(
    comment,
    video_id,
    video_url,
):
    """
    Chỉ lưu dữ liệu cần thiết cho nghiên cứu ABSA.

    Không đưa thông tin cá nhân của tác giả comment
    vào dataset train.
    """

    return {
        "comment_id": comment.get("comment_id"),
        "video_id": video_id,
        "video_url": video_url,

        "text": comment.get("text", ""),

        "created_at": comment.get("created_at"),
        "created_at_iso": comment.get("created_at_iso"),

        "like_count": comment.get(
            "like_count",
            0
        ),

        "reply_count": comment.get(
            "reply_count",
            0
        ),
    }


# ============================================================
# MAIN
# ============================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    urls = load_urls()

    existing_ids = load_existing_comment_ids()

    print("=" * 60)
    print("ABSA RAW DATA COLLECTOR")
    print("=" * 60)

    print(f"Videos: {len(urls)}")
    print(
        f"Existing comments: "
        f"{len(existing_ids)}"
    )

    total_new = 0

    with open(
        OUTPUT_FILE,
        "a",
        encoding="utf-8"
    ) as output:

        for index, url in enumerate(
            urls,
            start=1
        ):

            print()
            print(
                f"[{index}/{len(urls)}]"
            )
            print(url)

            comments, video_id = (
                get_all_comments(
                    url,
                    max_comments=COMMENTS_PER_VIDEO
                )
            )

            if not comments:
                print(
                    "[!] Không lấy được comment"
                )
                continue

            added = 0

            for comment in comments:

                comment_id = str(
                    comment.get(
                        "comment_id"
                    )
                    or ""
                )

                # --------------------------------------------
                # DUPLICATE CHECK
                # --------------------------------------------

                if (
                    comment_id
                    and comment_id
                    in existing_ids
                ):
                    continue

                text = str(
                    comment.get("text")
                    or ""
                ).strip()

                if len(text) < 2:
                    continue

                row = build_dataset_row(
                    comment=comment,
                    video_id=video_id,
                    video_url=url,
                )

                output.write(
                    json.dumps(
                        row,
                        ensure_ascii=False
                    )
                    + "\n"
                )

                if comment_id:
                    existing_ids.add(
                        comment_id
                    )

                added += 1
                total_new += 1

            output.flush()

            print(
                f"[+] Added: {added}"
            )

            print(
                f"[+] Dataset total: "
                f"{len(existing_ids)}"
            )

            time.sleep(
                SLEEP_BETWEEN_VIDEOS
            )

    print()
    print("=" * 60)

    print(
        f"New comments: {total_new}"
    )

    print(
        f"Saved to: {OUTPUT_FILE}"
    )

    print("=" * 60)


if __name__ == "__main__":
    main()