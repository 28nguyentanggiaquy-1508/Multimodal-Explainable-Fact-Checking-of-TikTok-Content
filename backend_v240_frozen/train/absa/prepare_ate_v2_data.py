import json
from collections import Counter
from pathlib import Path

from config_ate_v2 import (
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


def overlaps(a_start, a_end, b_start, b_end):
    return (
        a_start < b_end
        and b_start < a_end
    )


def aspect_is_representable(
    aspect,
    segments,
):
    """
    A gold aspect is representable iff one or more VnCoreNLP words
    cover the gold span exactly.

    We do NOT alter gold spans to make metrics look better.
    """
    start = int(
        aspect["start"]
    )
    end = int(
        aspect["end"]
    )

    inside = [
        token
        for token in segments
        if (
            token["start"] >= start
            and token["end"] <= end
        )
    ]

    if not inside:
        return False

    return (
        inside[0]["start"] == start
        and inside[-1]["end"] == end
    )


def assign_labels(
    segments,
    aspects,
):
    labels = []
    ignore_mask = []

    for token in segments:
        t_start = token["start"]
        t_end = token["end"]

        overlapping = [
            aspect
            for aspect in aspects
            if overlaps(
                t_start,
                t_end,
                int(aspect["start"]),
                int(aspect["end"]),
            )
        ]

        if not overlapping:
            labels.append("O")
            ignore_mask.append(False)
            continue

        if len(overlapping) > 1:
            # Current annotation guideline does not allow overlap.
            labels.append("O")
            ignore_mask.append(True)
            continue

        aspect = overlapping[0]

        a_start = int(
            aspect["start"]
        )
        a_end = int(
            aspect["end"]
        )

        # Gold boundary cuts through a VnCoreNLP word.
        # Ignore this token in training instead of injecting wrong labels.
        if not (
            t_start >= a_start
            and t_end <= a_end
        ):
            labels.append("O")
            ignore_mask.append(True)
            continue

        if t_start == a_start:
            labels.append("B-ASP")
        else:
            labels.append("I-ASP")

        ignore_mask.append(False)

    return labels, ignore_mask


def convert_split(
    source_file,
    output_file,
    segmenter,
):
    data = load_json(
        source_file
    )

    output = []

    stats = Counter()

    conflict_examples = []

    for item in data:
        text = str(
            item.get(
                "text",
                "",
            )
            or ""
        )

        aspects = (
            item.get(
                "aspects"
            )
            or []
        )

        segments = (
            segmenter
            .segment_with_offsets(
                text
            )
        )

        labels, ignore_mask = (
            assign_labels(
                segments,
                aspects,
            )
        )

        representable = []

        for aspect in aspects:
            ok = (
                aspect_is_representable(
                    aspect,
                    segments,
                )
            )

            representable.append(
                ok
            )

            stats[
                "gold_aspects"
            ] += 1

            if ok:
                stats[
                    "representable_aspects"
                ] += 1
            else:
                stats[
                    "unrepresentable_aspects"
                ] += 1

                if len(
                    conflict_examples
                ) < 30:
                    conflict_examples.append({
                        "comment_id":
                            item.get(
                                "comment_id"
                            ),

                        "text":
                            text,

                        "aspect":
                            aspect,

                        "segments":
                            segments,
                    })

        stats[
            "comments"
        ] += 1

        stats[
            "segments"
        ] += len(
            segments
        )

        stats[
            "ignored_segments"
        ] += sum(
            bool(value)
            for value in ignore_mask
        )

        output.append({
            "comment_id":
                item.get(
                    "comment_id"
                ),

            "video_id":
                item.get(
                    "video_id"
                ),

            "text":
                text,

            # Original character-span ground truth remains authoritative.
            "aspects":
                aspects,

            "segments":
                segments,

            "segment_labels":
                labels,

            "segment_ignore":
                ignore_mask,

            "aspect_representable":
                representable,
        })

    save_json(
        output_file,
        output,
    )

    return {
        **dict(stats),

        "representability_rate":
            (
                stats[
                    "representable_aspects"
                ]
                / stats[
                    "gold_aspects"
                ]
                if stats[
                    "gold_aspects"
                ]
                else 1.0
            ),

        "conflict_examples":
            conflict_examples,
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

    all_stats = {}

    jobs = [
        (
            "train",
            TRAIN_SOURCE,
            TRAIN_FILE,
        ),
        (
            "validation",
            VAL_SOURCE,
            VAL_FILE,
        ),
        (
            "test",
            TEST_SOURCE,
            TEST_FILE,
        ),
    ]

    for (
        name,
        source,
        output,
    ) in jobs:
        print()
        print(
            f"Preparing {name}:",
            source,
        )

        stats = convert_split(
            source,
            output,
            segmenter,
        )

        all_stats[
            name
        ] = stats

        print(
            f"  comments={stats['comments']}"
        )

        print(
            f"  gold_aspects={stats['gold_aspects']}"
        )

        print(
            "  representable="
            f"{stats['representable_aspects']}"
            "/"
            f"{stats['gold_aspects']}"
            " = "
            f"{stats['representability_rate']:.2%}"
        )

        print(
            f"  ignored_segments="
            f"{stats['ignored_segments']}"
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

    print(
        "Stats:",
        PREP_STATS_FILE,
    )


if __name__ == "__main__":
    main()
