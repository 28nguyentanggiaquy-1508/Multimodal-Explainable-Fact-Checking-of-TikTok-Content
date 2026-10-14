import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path


# ============================================================
# CONFIG
# ============================================================

RANDOM_SEED = 42

TRAIN_RATIO = 0.75
VAL_RATIO = 0.125
TEST_RATIO = 0.125


# ============================================================
# PATHS
# ============================================================

BACKEND_DIR = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    BACKEND_DIR
    / "data"
    / "absa"
    / "annotated"
    / "annotated.jsonl"
)

OUTPUT_DIR = (
    BACKEND_DIR
    / "data"
    / "absa"
    / "processed"
)

TRAIN_FILE = OUTPUT_DIR / "train.json"
VAL_FILE = OUTPUT_DIR / "validation.json"
TEST_FILE = OUTPUT_DIR / "test.json"

STATS_FILE = (
    OUTPUT_DIR
    / "dataset_stats.json"
)


# ============================================================
# LOAD JSONL
# ============================================================

def load_jsonl(path: Path):

    if not path.exists():

        raise FileNotFoundError(
            f"Không tìm thấy dataset: {path}"
        )

    rows = []

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:

        for line_number, line in enumerate(
            f,
            start=1,
        ):

            line = line.strip()

            if not line:
                continue

            try:
                row = json.loads(line)

            except json.JSONDecodeError as exc:

                raise ValueError(
                    f"JSON lỗi tại dòng "
                    f"{line_number}: {exc}"
                )

            rows.append(row)

    return rows


# ============================================================
# VALIDATION
# ============================================================

VALID_POLARITIES = {
    "POSITIVE",
    "NEGATIVE",
    "NEUTRAL",
}


def validate_dataset(rows):

    errors = []

    seen_ids = set()

    for i, row in enumerate(rows):

        comment_id = str(
            row.get("comment_id") or ""
        )

        video_id = str(
            row.get("video_id") or ""
        )

        text = str(
            row.get("text") or ""
        )

        status = row.get(
            "annotation_status"
        )

        has_aspect = row.get(
            "has_aspect"
        )

        aspects = row.get(
            "aspects"
        ) or []

        # --------------------------------------------
        # Required fields
        # --------------------------------------------

        if not comment_id:

            errors.append(
                f"Row {i}: thiếu comment_id"
            )

        if not video_id:

            errors.append(
                f"Row {i}: thiếu video_id"
            )

        if not text:

            errors.append(
                f"Row {i}: text rỗng"
            )

        # --------------------------------------------
        # Duplicate comment ID
        # --------------------------------------------

        if comment_id in seen_ids:

            errors.append(
                f"Duplicate comment_id: "
                f"{comment_id}"
            )

        seen_ids.add(comment_id)

        # --------------------------------------------
        # Annotation status
        # --------------------------------------------

        if status != "done":

            errors.append(
                f"Comment {comment_id}: "
                f"status != done ({status})"
            )

        # --------------------------------------------
        # has_aspect
        # --------------------------------------------

        if has_aspect not in {
            True,
            False,
        }:

            errors.append(
                f"Comment {comment_id}: "
                f"has_aspect không hợp lệ"
            )

        if (
            has_aspect is False
            and len(aspects) > 0
        ):

            errors.append(
                f"Comment {comment_id}: "
                f"No Aspect nhưng vẫn có aspects"
            )

        if (
            has_aspect is True
            and len(aspects) == 0
        ):

            errors.append(
                f"Comment {comment_id}: "
                f"has_aspect=True "
                f"nhưng aspects rỗng"
            )

        # --------------------------------------------
        # Validate aspects
        # --------------------------------------------

        previous_spans = []

        for aspect_index, aspect in enumerate(
            aspects
        ):

            term = str(
                aspect.get("term") or ""
            )

            start = aspect.get(
                "start"
            )

            end = aspect.get(
                "end"
            )

            polarity = str(
                aspect.get("polarity")
                or ""
            ).upper()

            if polarity not in VALID_POLARITIES:

                errors.append(
                    f"Comment {comment_id}, "
                    f"aspect {aspect_index}: "
                    f"polarity={polarity}"
                )

            if not isinstance(
                start,
                int,
            ):

                errors.append(
                    f"Comment {comment_id}: "
                    f"start không phải int"
                )

                continue

            if not isinstance(
                end,
                int,
            ):

                errors.append(
                    f"Comment {comment_id}: "
                    f"end không phải int"
                )

                continue

            if (
                start < 0
                or end > len(text)
                or start >= end
            ):

                errors.append(
                    f"Comment {comment_id}: "
                    f"span [{start}, {end}] "
                    f"không hợp lệ"
                )

                continue

            actual_term = text[
                start:end
            ]

            if actual_term != term:

                errors.append(
                    f"Comment {comment_id}: "
                    f"term='{term}' "
                    f"không khớp span="
                    f"'{actual_term}'"
                )

            # ----------------------------------------
            # overlapping spans
            # ----------------------------------------

            for old_start, old_end in previous_spans:

                overlap = not (
                    end <= old_start
                    or start >= old_end
                )

                if overlap:

                    errors.append(
                        f"Comment {comment_id}: "
                        f"aspect span overlap"
                    )

            previous_spans.append(
                (
                    start,
                    end,
                )
            )

    if errors:

        print()
        print("=" * 70)
        print("DATASET VALIDATION FAILED")
        print("=" * 70)

        for error in errors[:50]:
            print("[ERROR]", error)

        if len(errors) > 50:

            print(
                f"... và "
                f"{len(errors) - 50} lỗi khác"
            )

        raise ValueError(
            f"Dataset có "
            f"{len(errors)} lỗi."
        )

    print(
        "[OK] Dataset validation passed"
    )


# ============================================================
# TOKENIZATION FOR BIO LABEL
# ============================================================

TOKEN_PATTERN = re.compile(
    r"\w+|[^\w\s]",
    flags=re.UNICODE,
)


def tokenize_with_offsets(text):

    tokens = []

    offsets = []

    for match in TOKEN_PATTERN.finditer(
        text
    ):

        tokens.append(
            match.group()
        )

        offsets.append(
            [
                match.start(),
                match.end(),
            ]
        )

    return tokens, offsets


# ============================================================
# CHAR SPAN -> BIO
# ============================================================

def build_bio_labels(
    text,
    aspects,
):

    tokens, offsets = (
        tokenize_with_offsets(text)
    )

    labels = [
        "O"
        for _ in tokens
    ]

    for aspect in aspects:

        aspect_start = aspect[
            "start"
        ]

        aspect_end = aspect[
            "end"
        ]

        matched_indices = []

        for i, (
            token_start,
            token_end,
        ) in enumerate(offsets):

            overlap = not (
                token_end <= aspect_start
                or token_start >= aspect_end
            )

            if overlap:
                matched_indices.append(i)

        if not matched_indices:
            continue

        labels[
            matched_indices[0]
        ] = "B-ASP"

        for index in matched_indices[1:]:

            labels[index] = "I-ASP"

    return (
        tokens,
        offsets,
        labels,
    )


# ============================================================
# CONVERT SAMPLE
# ============================================================

def convert_sample(row):

    text = row["text"]

    aspects = row.get(
        "aspects"
    ) or []

    tokens, offsets, bio_labels = (
        build_bio_labels(
            text,
            aspects,
        )
    )

    return {
        "comment_id":
            str(row["comment_id"]),

        "video_id":
            str(row["video_id"]),

        "text":
            text,

        "has_aspect":
            bool(
                row["has_aspect"]
            ),

        "aspects":
            aspects,

        "tokens":
            tokens,

        "token_offsets":
            offsets,

        "bio_labels":
            bio_labels,
    }


# ============================================================
# VIDEO STATISTICS
# ============================================================

def get_video_stats(rows):

    stats = defaultdict(
        lambda: {
            "comments": 0,
            "has_aspect": 0,
            "no_aspect": 0,

            "POSITIVE": 0,
            "NEGATIVE": 0,
            "NEUTRAL": 0,
        }
    )

    for row in rows:

        video_id = str(
            row["video_id"]
        )

        stats[
            video_id
        ]["comments"] += 1

        if row.get(
            "has_aspect"
        ):

            stats[
                video_id
            ]["has_aspect"] += 1

        else:

            stats[
                video_id
            ]["no_aspect"] += 1

        for aspect in (
            row.get("aspects")
            or []
        ):

            polarity = (
                aspect.get(
                    "polarity"
                )
                or ""
            ).upper()

            if polarity in VALID_POLARITIES:

                stats[
                    video_id
                ][polarity] += 1

    return dict(stats)


# ============================================================
# SPLIT BY VIDEO
# ============================================================

def split_by_video(rows):

    # ========================================================
    # FIXED GROUP SPLIT - ABSA DATASET V1
    #
    # Không được thay đổi các ID này sau khi bắt đầu train,
    # để các lần thực nghiệm có thể so sánh công bằng.
    # ========================================================

    VAL_VIDEO_IDS = {
        "7682441737914092821",
        "7682769069434539285",
    }

    TEST_VIDEO_IDS = {
        "7200064689764715818",
        "7666732540748762386",
    }

    all_video_ids = {
        str(row["video_id"])
        for row in rows
    }

    # ========================================================
    # VALIDATION
    # ========================================================

    missing_val = (
        VAL_VIDEO_IDS
        - all_video_ids
    )

    missing_test = (
        TEST_VIDEO_IDS
        - all_video_ids
    )

    if missing_val:

        raise ValueError(
            f"Không tìm thấy validation video: "
            f"{missing_val}"
        )

    if missing_test:

        raise ValueError(
            f"Không tìm thấy test video: "
            f"{missing_test}"
        )

    if (
        VAL_VIDEO_IDS
        & TEST_VIDEO_IDS
    ):

        raise ValueError(
            "Validation và test bị trùng video."
        )

    # ========================================================
    # TRAIN = TẤT CẢ VIDEO CÒN LẠI
    # ========================================================

    TRAIN_VIDEO_IDS = (
        all_video_ids
        - VAL_VIDEO_IDS
        - TEST_VIDEO_IDS
    )

    train_rows = []

    val_rows = []

    test_rows = []

    for row in rows:

        video_id = str(
            row["video_id"]
        )

        if video_id in VAL_VIDEO_IDS:

            val_rows.append(
                row
            )

        elif video_id in TEST_VIDEO_IDS:

            test_rows.append(
                row
            )

        elif video_id in TRAIN_VIDEO_IDS:

            train_rows.append(
                row
            )

    # ========================================================
    # SAFETY CHECK: NO VIDEO LEAKAGE
    # ========================================================

    train_videos = {
        str(r["video_id"])
        for r in train_rows
    }

    val_videos = {
        str(r["video_id"])
        for r in val_rows
    }

    test_videos = {
        str(r["video_id"])
        for r in test_rows
    }

    if train_videos & val_videos:

        raise ValueError(
            "Video leakage: train ↔ validation"
        )

    if train_videos & test_videos:

        raise ValueError(
            "Video leakage: train ↔ test"
        )

    if val_videos & test_videos:

        raise ValueError(
            "Video leakage: validation ↔ test"
        )

    return (
        train_rows,
        val_rows,
        test_rows,
        sorted(train_videos),
        sorted(val_videos),
        sorted(test_videos),
    )

# ============================================================
# DATASET STATS
# ============================================================

def calculate_split_stats(rows):

    polarity_counter = Counter()

    aspect_count = 0

    multi_aspect_comments = 0

    has_aspect = 0

    no_aspect = 0

    bio_counter = Counter()

    videos = set()

    for row in rows:

        videos.add(
            str(row["video_id"])
        )

        aspects = (
            row.get("aspects")
            or []
        )

        if aspects:

            has_aspect += 1

        else:

            no_aspect += 1

        aspect_count += len(
            aspects
        )

        if len(aspects) > 1:

            multi_aspect_comments += 1

        for aspect in aspects:

            polarity_counter[
                aspect["polarity"]
            ] += 1

        for label in (
            row.get(
                "bio_labels"
            )
            or []
        ):

            bio_counter[
                label
            ] += 1

    return {
        "comments":
            len(rows),

        "videos":
            len(videos),

        "has_aspect":
            has_aspect,

        "no_aspect":
            no_aspect,

        "aspect_count":
            aspect_count,

        "multi_aspect_comments":
            multi_aspect_comments,

        "polarity_distribution":
            dict(
                polarity_counter
            ),

        "bio_distribution":
            dict(
                bio_counter
            ),
    }


# ============================================================
# SAVE JSON
# ============================================================

def save_json(
    path: Path,
    data,
):

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            data,
            f,
            ensure_ascii=False,
            indent=2,
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("ABSA TRAINING DATA PREPARATION")
    print("=" * 70)

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    rows = load_jsonl(
        INPUT_FILE
    )

    print(
        f"Loaded: {len(rows)} comments"
    )

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    validate_dataset(
        rows
    )

    # --------------------------------------------------------
    # Split by video
    # --------------------------------------------------------

    (
        train_rows,
        val_rows,
        test_rows,
        train_videos,
        val_videos,
        test_videos,
    ) = split_by_video(
        rows
    )

    # --------------------------------------------------------
    # Convert to training format
    # --------------------------------------------------------

    train_data = [
        convert_sample(row)
        for row in train_rows
    ]

    val_data = [
        convert_sample(row)
        for row in val_rows
    ]

    test_data = [
        convert_sample(row)
        for row in test_rows
    ]

    # --------------------------------------------------------
    # Create output directory
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    save_json(
        TRAIN_FILE,
        train_data,
    )

    save_json(
        VAL_FILE,
        val_data,
    )

    save_json(
        TEST_FILE,
        test_data,
    )

    # --------------------------------------------------------
    # Stats
    # --------------------------------------------------------

    stats = {

        "seed":
            RANDOM_SEED,

        "split_strategy":
            "grouped_by_video_id",

        "ratios": {
            "train":
                TRAIN_RATIO,

            "validation":
                VAL_RATIO,

            "test":
                TEST_RATIO,
        },

        "video_split": {

            "train":
                train_videos,

            "validation":
                val_videos,

            "test":
                test_videos,
        },

        "train":
            calculate_split_stats(
                train_data
            ),

        "validation":
            calculate_split_stats(
                val_data
            ),

        "test":
            calculate_split_stats(
                test_data
            ),
    }

    save_json(
        STATS_FILE,
        stats,
    )

    # --------------------------------------------------------
    # Print
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("VIDEO SPLIT")
    print("=" * 70)

    print(
        f"Train videos: "
        f"{len(train_videos)}"
    )

    print(
        f"Validation videos: "
        f"{len(val_videos)}"
    )

    print(
        f"Test videos: "
        f"{len(test_videos)}"
    )

    print()
    print(
        f"Train comments: "
        f"{len(train_data)}"
    )

    print(
        f"Validation comments: "
        f"{len(val_data)}"
    )

    print(
        f"Test comments: "
        f"{len(test_data)}"
    )

    print()
    print("=" * 70)
    print("POLARITY DISTRIBUTION")
    print("=" * 70)

    print(
        "Train:",
        stats["train"][
            "polarity_distribution"
        ],
    )

    print(
        "Validation:",
        stats["validation"][
            "polarity_distribution"
        ],
    )

    print(
        "Test:",
        stats["test"][
            "polarity_distribution"
        ],
    )

    print()
    print("=" * 70)
    print("BIO DISTRIBUTION")
    print("=" * 70)

    print(
        "Train:",
        stats["train"][
            "bio_distribution"
        ],
    )

    print()
    print(
        "[OK] Files saved to:"
    )

    print(
        OUTPUT_DIR
    )


if __name__ == "__main__":
    main()