
import json

from config_ate_v2 import BACKEND_DIR


V1_FILE = BACKEND_DIR / "evaluation_outputs" / "absa_v1" / "metrics.json"
V2_FILE = BACKEND_DIR / "evaluation_outputs" / "absa_v2_end_to_end_frozen" / "metrics.json"
OUT_FILE = BACKEND_DIR / "evaluation_outputs" / "absa_v1_vs_v2.md"


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def pct(x):
    return f"{x * 100:.2f}%"


def pp(v1, v2):
    return f"{(v2 - v1) * 100:+.2f} pp"


def main():
    v1 = load_json(V1_FILE)
    v2 = load_json(V2_FILE)

    a1 = v1["ate_exact_char_span"]
    a2 = v2["ate_exact_char_span"]

    s1 = v1["asc_oracle_gold_aspect"]
    s2 = v2["asc_oracle_gold_aspect"]

    e1 = v1["end_to_end_exact_span_sentiment"]
    e2 = v2["end_to_end_exact_span_sentiment"]

    lines = [
        "# ABSA V1 vs V2",
        "",
        "## ATE - Exact Character Span",
        "",
        "| Metric | V1 | V2 | Delta |",
        "|---|---:|---:|---:|",
        f"| Precision | {pct(a1['precision'])} | {pct(a2['precision'])} | {pp(a1['precision'], a2['precision'])} |",
        f"| Recall | {pct(a1['recall'])} | {pct(a2['recall'])} | {pp(a1['recall'], a2['recall'])} |",
        f"| F1 | {pct(a1['f1'])} | {pct(a2['f1'])} | {pp(a1['f1'], a2['f1'])} |",
        f"| FP | {a1['fp']} | {a2['fp']} | {a2['fp'] - a1['fp']:+d} |",
        f"| FN | {a1['fn']} | {a2['fn']} | {a2['fn'] - a1['fn']:+d} |",
        "",
        "## ASC - Gold Aspect / Oracle ATE",
        "",
        "| Metric | V1 | V2 | Delta |",
        "|---|---:|---:|---:|",
        f"| Accuracy | {pct(s1['accuracy'])} | {pct(s2['accuracy'])} | {pp(s1['accuracy'], s2['accuracy'])} |",
        f"| Macro-F1 | {pct(s1['macro_f1'])} | {pct(s2['macro_f1'])} | {pp(s1['macro_f1'], s2['macro_f1'])} |",
        f"| NEGATIVE F1 | {pct(s1['per_class']['NEGATIVE']['f1'])} | {pct(s2['per_class']['NEGATIVE']['f1'])} | {pp(s1['per_class']['NEGATIVE']['f1'], s2['per_class']['NEGATIVE']['f1'])} |",
        f"| NEUTRAL F1 | {pct(s1['per_class']['NEUTRAL']['f1'])} | {pct(s2['per_class']['NEUTRAL']['f1'])} | {pp(s1['per_class']['NEUTRAL']['f1'], s2['per_class']['NEUTRAL']['f1'])} |",
        f"| POSITIVE F1 | {pct(s1['per_class']['POSITIVE']['f1'])} | {pct(s2['per_class']['POSITIVE']['f1'])} | {pp(s1['per_class']['POSITIVE']['f1'], s2['per_class']['POSITIVE']['f1'])} |",
        "",
        "## End-to-End ABSA",
        "",
        "Exact match = `(start, end, sentiment)`.",
        "",
        "| Metric | V1 | V2 | Delta |",
        "|---|---:|---:|---:|",
        f"| Precision | {pct(e1['precision'])} | {pct(e2['precision'])} | {pp(e1['precision'], e2['precision'])} |",
        f"| Recall | {pct(e1['recall'])} | {pct(e2['recall'])} | {pp(e1['recall'], e2['recall'])} |",
        f"| F1 | {pct(e1['f1'])} | {pct(e2['f1'])} | {pp(e1['f1'], e2['f1'])} |",
        "",
        "Frozen-test results are for final comparison, not further tuning.",
    ]

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUT_FILE.write_text("\n".join(lines), encoding="utf-8")

    print("Saved:", OUT_FILE)


if __name__ == "__main__":
    main()
