import json
import unicodedata

from config_ate_v2 import (
    TRAIN_SOURCE,
    VAL_SOURCE,
    TEST_SOURCE,
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


def main():
    segmenter = (
        VnCoreNLPSegmenter(
            VNCORENLP_DIR
        )
    )

    cases = [
        "tùy tỉ lệ cơ thể nữa",
        "tuỳ tỉ lệ cơ thể nữa",
        unicodedata.normalize(
            "NFD",
            "tùy tỉ lệ cơ thể nữa",
        ),
        "tùy\u00a0tỉ lệ cơ thể nữa",
        "tùy\u200btỉ lệ cơ thể nữa",
    ]

    print("=" * 72)
    print("REGRESSION CASES")
    print("=" * 72)

    for text in cases:
        segments = (
            segmenter
            .segment_with_offsets(text)
        )

        print(
            repr(text),
            "->",
            [
                (
                    x["word_form"],
                    x["surface"],
                    x["start"],
                    x["end"],
                    x["source"],
                )
                for x in segments
            ],
        )

    print()
    print("=" * 72)
    print("FULL DATASET ALIGNMENT CHECK")
    print("=" * 72)

    report = {}

    for name, path in [
        ("train", TRAIN_SOURCE),
        ("validation", VAL_SOURCE),
        ("test", TEST_SOURCE),
    ]:
        data = load_json(path)

        comments = 0
        fallback_comments = 0
        fallback_segments = 0

        for item in data:
            text = str(
                item.get("text", "")
                or ""
            )

            segments = (
                segmenter
                .segment_with_offsets(text)
            )

            comments += 1

            current_fallback = sum(
                1
                for x in segments
                if x.get("source")
                == "raw_fallback"
            )

            if current_fallback:
                fallback_comments += 1
                fallback_segments += (
                    current_fallback
                )

            # ASC aspect segmentation cũng phải chạy hết.
            for aspect in (
                item.get("aspects")
                or []
            ):
                start = int(aspect["start"])
                end = int(aspect["end"])

                segmenter.segment_text(
                    text[start:end]
                )

        report[name] = {
            "comments": comments,
            "fallback_comments": fallback_comments,
            "fallback_segments": fallback_segments,
        }

        print(
            f"{name}: "
            f"comments={comments}, "
            f"fallback_comments={fallback_comments}, "
            f"fallback_segments={fallback_segments}"
        )

    print()
    print(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
        )
    )

    total_fallback = sum(
        x["fallback_comments"]
        for x in report.values()
    )

    print()

    if total_fallback == 0:
        print(
            "[OK] Tất cả comments align bằng robust VnCoreNLP path."
        )
    else:
        print(
            "[WARN] Pipeline vẫn chạy được, nhưng có comments dùng "
            "raw_fallback. Gửi output để kiểm tra thêm."
        )


if __name__ == "__main__":
    main()
