import json

from inference_ate_v2 import (
    ATEPhoBERTV2,
)


def main():
    engine = (
        ATEPhoBERTV2()
    )

    print(
        "PhoBERT ATE V2 ready."
    )

    print(
        "Type a Vietnamese comment. "
        "Ctrl+C to exit."
    )

    while True:
        try:
            text = input(
                "\nComment: "
            ).strip()

        except KeyboardInterrupt:
            print()
            break

        if not text:
            continue

        result = engine.predict(
            text
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
