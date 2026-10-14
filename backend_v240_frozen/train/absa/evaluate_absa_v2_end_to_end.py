
import json

from config_ate_v2 import BACKEND_DIR, TEST_SOURCE
from config_asc_v2 import LABEL2ID
from inference_ate_v2 import ATEPhoBERTV2
from inference_asc_v2 import ASCPhoBERTV2
from metrics_ate_v2 import exact_span_metrics
from metrics_asc_v2 import classification_metrics


OUTPUT_DIR = BACKEND_DIR / "evaluation_outputs" / "absa_v2_end_to_end"


def safe_div(a, b):
    return a / b if b else 0.0


def prf(tp, fp, fn):
    p = safe_div(tp, tp + fp)
    r = safe_div(tp, tp + fn)
    f1 = safe_div(2 * p * r, p + r)
    return {
        "precision": p,
        "recall": r,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def norm(label):
    return str(label or "").strip().upper()


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    data = load_json(TEST_SOURCE)
    ate_engine = ATEPhoBERTV2()
    asc_engine = ASCPhoBERTV2()

    ate_gold_all = []
    ate_pred_all = []

    oracle_gold = []
    oracle_pred = []

    matched_gold = []
    matched_pred = []

    e2e_tp = e2e_fp = e2e_fn = 0

    breakdown = {
        "gold_aspects": 0,
        "ate_exact_tp": 0,
        "ate_false_positive": 0,
        "ate_false_negative": 0,
        "matched_span_sentiment_correct": 0,
        "matched_span_sentiment_wrong": 0,
    }

    errors = {
        "ate_false_negatives": [],
        "ate_false_positives": [],
        "matched_span_sentiment_errors": [],
        "end_to_end_errors": [],
    }

    for i, item in enumerate(data, start=1):
        text = str(item.get("text", "") or "")
        gold_aspects = item.get("aspects") or []

        gold_spans = set()
        gold_by_span = {}
        gold_triples = set()

        for aspect in gold_aspects:
            start = int(aspect["start"])
            end = int(aspect["end"])
            sentiment = norm(aspect.get("polarity"))

            span = (start, end)
            gold_spans.add(span)
            gold_by_span[span] = sentiment
            gold_triples.add((start, end, sentiment))

            # ASC Oracle: gold aspect -> ASC V2
            result = asc_engine.predict(text, text[start:end])
            pred_sentiment = norm(result["sentiment"])

            oracle_gold.append(LABEL2ID[sentiment])
            oracle_pred.append(LABEL2ID[pred_sentiment])

        breakdown["gold_aspects"] += len(gold_spans)

        # ATE V2
        ate_result = ate_engine.predict(text)
        predicted_aspects = ate_result["aspects"]

        pred_spans = {
            (int(a["start"]), int(a["end"]))
            for a in predicted_aspects
        }

        ate_gold_all.append(gold_spans)
        ate_pred_all.append(pred_spans)

        tp_spans = gold_spans & pred_spans
        fp_spans = pred_spans - gold_spans
        fn_spans = gold_spans - pred_spans

        breakdown["ate_exact_tp"] += len(tp_spans)
        breakdown["ate_false_positive"] += len(fp_spans)
        breakdown["ate_false_negative"] += len(fn_spans)

        for start, end in sorted(fn_spans):
            errors["ate_false_negatives"].append({
                "comment_id": item.get("comment_id"),
                "text": text,
                "term": text[start:end],
                "start": start,
                "end": end,
                "gold_sentiment": gold_by_span[(start, end)],
            })

        for start, end in sorted(fp_spans):
            errors["ate_false_positives"].append({
                "comment_id": item.get("comment_id"),
                "text": text,
                "term": text[start:end],
                "start": start,
                "end": end,
            })

        # End-to-end: ATE V2 -> ASC V2
        pred_triples = set()
        pred_detail = []

        for predicted_aspect in predicted_aspects:
            start = int(predicted_aspect["start"])
            end = int(predicted_aspect["end"])
            term = text[start:end]

            asc_result = asc_engine.predict(text, term)
            pred_sentiment = norm(asc_result["sentiment"])

            pred_triples.add((start, end, pred_sentiment))

            pred_detail.append({
                "term": term,
                "start": start,
                "end": end,
                "sentiment": pred_sentiment,
                "ate_confidence": predicted_aspect.get("confidence"),
                "asc_confidence": asc_result.get("confidence"),
                "asc_probabilities": asc_result.get("probabilities"),
            })

            span = (start, end)
            if span in gold_by_span:
                gold_sentiment = gold_by_span[span]

                matched_gold.append(LABEL2ID[gold_sentiment])
                matched_pred.append(LABEL2ID[pred_sentiment])

                if gold_sentiment == pred_sentiment:
                    breakdown["matched_span_sentiment_correct"] += 1
                else:
                    breakdown["matched_span_sentiment_wrong"] += 1
                    errors["matched_span_sentiment_errors"].append({
                        "comment_id": item.get("comment_id"),
                        "text": text,
                        "term": term,
                        "start": start,
                        "end": end,
                        "gold": gold_sentiment,
                        "pred": pred_sentiment,
                        "probabilities": asc_result.get("probabilities"),
                    })

        cur_tp = len(gold_triples & pred_triples)
        cur_fp = len(pred_triples - gold_triples)
        cur_fn = len(gold_triples - pred_triples)

        e2e_tp += cur_tp
        e2e_fp += cur_fp
        e2e_fn += cur_fn

        if cur_fp or cur_fn:
            errors["end_to_end_errors"].append({
                "comment_id": item.get("comment_id"),
                "video_id": item.get("video_id"),
                "text": text,
                "gold": [
                    {
                        "term": text[start:end],
                        "start": start,
                        "end": end,
                        "sentiment": sentiment,
                    }
                    for start, end, sentiment in sorted(gold_triples)
                ],
                "pred": pred_detail,
            })

        if i % 20 == 0 or i == len(data):
            print(
                f"[{i}/{len(data)}] "
                f"ATE TP={breakdown['ate_exact_tp']} "
                f"FP={breakdown['ate_false_positive']} "
                f"FN={breakdown['ate_false_negative']} | "
                f"E2E TP={e2e_tp} FP={e2e_fp} FN={e2e_fn}"
            )

    ate_metrics = exact_span_metrics(ate_gold_all, ate_pred_all)
    asc_oracle = classification_metrics(oracle_gold, oracle_pred)

    asc_on_exact = (
        classification_metrics(matched_gold, matched_pred)
        if matched_gold
        else {"accuracy": 0.0, "macro_f1": 0.0}
    )

    e2e = prf(e2e_tp, e2e_fp, e2e_fn)

    breakdown["end_to_end_tp"] = e2e_tp
    breakdown["end_to_end_fp"] = e2e_fp
    breakdown["end_to_end_fn"] = e2e_fn
    breakdown["asc_accuracy_on_exact_ate_matches"] = safe_div(
        breakdown["matched_span_sentiment_correct"],
        breakdown["matched_span_sentiment_correct"]
        + breakdown["matched_span_sentiment_wrong"],
    )

    metrics = {
        "dataset": {
            "test_file": str(TEST_SOURCE),
            "comments": len(data),
            "gold_aspects": breakdown["gold_aspects"],
        },
        "models": {
            "ate": "ate_phobert_v2",
            "asc": "asc_phobert_v2",
        },
        "ate_exact_char_span": ate_metrics,
        "asc_oracle_gold_aspect": asc_oracle,
        "asc_on_exact_ate_matches": asc_on_exact,
        "end_to_end_exact_span_sentiment": e2e,
        "error_propagation": breakdown,
    }

    with open(OUTPUT_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    with open(OUTPUT_DIR / "errors.json", "w", encoding="utf-8") as f:
        json.dump(errors, f, ensure_ascii=False, indent=2)

    print()
    print("=" * 72)
    print("ABSA V2 END-TO-END RESULTS")
    print("=" * 72)
    print(f"ATE V2 F1               : {ate_metrics['f1']:.4f}")
    print(f"ASC V2 Oracle Macro-F1  : {asc_oracle['macro_f1']:.4f}")
    print(f"ASC exact-span Accuracy : {asc_on_exact.get('accuracy', 0):.4f}")
    print(f"End-to-End V2 Precision : {e2e['precision']:.4f}")
    print(f"End-to-End V2 Recall    : {e2e['recall']:.4f}")
    print(f"End-to-End V2 F1        : {e2e['f1']:.4f}")
    print("Saved:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
