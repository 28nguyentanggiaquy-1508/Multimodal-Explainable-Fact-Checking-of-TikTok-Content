from config_ate_v2 import VNCORENLP_DIR
from vncorenlp_segmenter import VnCoreNLPSegmenter


TESTS = [
    "Cíu taoo🥺",
    "tùy người thôi",
    "Ko ăn với lá lốt nhờn là thiếu rồi😏",
    "thay đổi cả cái vóc dáng nữa . dáng oke thì phối đồ cũng đẹp",
]


def main():
    segmenter = VnCoreNLPSegmenter(VNCORENLP_DIR)

    for text in TESTS:
        print("=" * 70)
        print("RAW:", repr(text))

        segments = segmenter.segment_with_offsets(text)

        for item in segments:
            print(
                item["word_form"],
                repr(item["surface"]),
                item["start"],
                item["end"],
                item["source"],
            )

        rebuilt = "".join(
            text[item["start"]:item["end"]]
            for item in segments
            if item["start"] < item["end"]
        )

        print("segments:", len(segments))
        print("OK")


if __name__ == "__main__":
    main()
