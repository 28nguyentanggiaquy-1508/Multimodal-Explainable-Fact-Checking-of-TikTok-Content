def _collapse_other(labels):
    upper = [str(l).upper() for l in labels]
    if "OTHER" in upper and len(upper) > 1:
        return ["OTHER"]
    return upper


def _set_label(prediction, new_label):
    new_label = new_label.upper()
    labels = list(prediction.get("all_labels") or [])
    if new_label not in labels:
        labels.insert(0, new_label)
    else:
        labels = [new_label] + [l for l in labels if l != new_label]

    prediction["label"] = new_label
    prediction["all_labels"] = _collapse_other(labels)


def refine_prediction(comment, prediction, context):
    """
    Context-aware sentiment correction.
    """

    text = (comment or "").lower()
    label = prediction.get("label", "")
    tone = (context or {}).get("tone", "") if isinstance(context, dict) else ""

    if tone == "sad":
        sarcasm_words = ["=))", "haha", "hahaha", "cười", "hài", "đáng đời"]
        if any(w in text for w in sarcasm_words) and label == "ENJOYMENT":
            _set_label(prediction, "DISGUST")

        sympathy_words = ["tội", "thương", "khóc", "buồn", "đau lòng", "xót"]
        if any(w in text for w in sympathy_words):
            _set_label(prediction, "SADNESS")

    if tone == "funny":
        funny_words = ["=))", "haha", "cười chết", "hài", "mắc cười"]
        if any(w in text for w in funny_words):
            _set_label(prediction, "ENJOYMENT")

    if tone == "angry":
        toxic_words = ["ngu", "toxic", "ghét", "xàm", "khùng"]
        if any(w in text for w in toxic_words):
            _set_label(prediction, "ANGER")

    # Final guard: ensure all_labels respects the OTHER-collapse rule even
    # when the adapter is older or refine_prediction didn't touch the label.
    prediction["all_labels"] = _collapse_other(prediction.get("all_labels") or [prediction.get("label", "OTHER")])
    if prediction["all_labels"]:
        prediction["label"] = prediction["all_labels"][0]

    return prediction
