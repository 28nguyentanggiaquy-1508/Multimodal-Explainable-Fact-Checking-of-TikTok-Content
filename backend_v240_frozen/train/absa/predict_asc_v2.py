import json

from inference_asc_v2 import (
    ASCPhoBERTV2,
)


def main():
    engine = (
        ASCPhoBERTV2()
    )

    print(
        "PhoBERT ASC V2 ready."
    )

    while True:
        try:
            text = input(
                "\nComment: "
            ).strip()

            if not text:
                continue

            aspect = input(
                "Aspect: "
            ).strip()

        except KeyboardInterrupt:
            print()
            break

        if not aspect:
            continue

        result = engine.predict(
            text,
            aspect,
        )

        print(
            json.dumps(
                result,
                ensure_ascii=False,
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
