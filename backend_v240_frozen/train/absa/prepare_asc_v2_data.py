import json
from collections import Counter

from config_asc_v2 import (
    TRAIN_SOURCE,
    VAL_SOURCE,
    TEST_SOURCE,
    TRAIN_FILE,
    VAL_FILE,
    TEST_FILE,
    PREP_STATS_FILE,
    V2_DATA_DIR,
    VNCORENLP_DIR,
)

from vncorenlp_segmenter import (
    VnCoreNLPSegmenter,
)


VALID_LABELS = {
    "NEGATIVE",
    "NEUTRAL",
    "POSITIVE",
}


def load_json(path):
    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:
        return json.load(f)


def save_json(path, data):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

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


def convert_split(
    source_file,
    output_file,
    segmenter,
):
    data = load_json(source_file)

    examples = []
    counts = Counter()
    invalid = []

    for item in data:
        text = str(
            item.get("text", "")
            or ""
        )

        segmented_text = (
            segmenter.segment_text(text)
        )

        for aspect in (
            item.get("aspects")
            or []
        ):
            start = int(aspect["start"])
            end = int(aspect["end"])
            term = text[start:end]

            label = str(
                aspect.get(
                    "polarity",
                    "",
                )
            ).strip().upper()

            if label not in VALID_LABELS:
                invalid.append({
                    "comment_id": item.get("comment_id"),
                    "text": text,
                    "aspect": aspect,
                })
                continue

            segmented_aspect = (
                segmenter.segment_text(term)
            )

            examples.append({
                "comment_id": item.get("comment_id"),
                "video_id": item.get("video_id"),
                "text": text,
                "segmented_text": segmented_text,
                "aspect_term": term,
                "segmented_aspect": segmented_aspect,
                "start": start,
                "end": end,
                "label": label,
            })

            counts[label] += 1

    save_json(
        output_file,
        examples,
    )

    return {
        "examples": len(examples),
        "label_counts": {
            label: counts[label]
            for label in [
                "NEGATIVE",
                "NEUTRAL",
                "POSITIVE",
            ]
        },
        "invalid_count": len(invalid),
        "invalid_examples": invalid[:20],
    }


def main():
    V2_DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    segmenter = (
        VnCoreNLPSegmenter(
            VNCORENLP_DIR
        )
    )

    jobs = [
        ("train", TRAIN_SOURCE, TRAIN_FILE),
        ("validation", VAL_SOURCE, VAL_FILE),
        ("test", TEST_SOURCE, TEST_FILE),
    ]

    all_stats = {}

    for name, source, output in jobs:
        print()
        print(
            f"Preparing ASC {name}:",
            source,
        )

        stats = convert_split(
            source,
            output,
            segmenter,
        )

        all_stats[name] = stats

        print(
            "  examples:",
            stats["examples"],
        )

        print(
            "  labels:",
            stats["label_counts"],
        )

    save_json(
        PREP_STATS_FILE,
        all_stats,
    )

    print()
    print(
        "Saved:",
        V2_DATA_DIR,
    )


if __name__ == "__main__":
    main()
