def filter_comments_by_context(comments, context):
    """
    Lọc comment theo ngữ cảnh video.
    """

    tone = context.get("tone", "")

    filtered = []

    for c in comments:
        text = c["text"].lower()

        # spam
        if len(text) < 2:
            continue

        # context aware
        if tone == "sad":
            if "=))" in text and "khóc" in text:
                c["boost_negative"] = True

        filtered.append(c)

    return filtered