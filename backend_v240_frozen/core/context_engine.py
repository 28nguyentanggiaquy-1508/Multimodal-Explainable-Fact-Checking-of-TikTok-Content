import re

def extract_video_context(comments):
    """
    Tạo ngữ cảnh video từ comments.
    """

    full_text = " ".join(comments).lower()

    keywords = []

    rules = {
        "sad": ["buồn", "khóc", "chia tay", "đau lòng"],
        "funny": ["haha", "cười", "=))", "hài"],
        "angry": ["toxic", "ghét", "tệ", "ức chế"],
        "love": ["yêu", "cute", "dễ thương"]
    }

    scores = {}

    for label, words in rules.items():
        count = sum(full_text.count(w) for w in words)
        scores[label] = count

    dominant = max(scores, key=scores.get)

    if dominant == "sad":
        topic = "Tâm trạng / cảm xúc"
    elif dominant == "funny":
        topic = "Giải trí"
    elif dominant == "angry":
        topic = "Tranh cãi / toxic"
    else:
        topic = "Đời sống"

    return {
        "topic": topic,
        "tone": dominant,
        "scores": scores
    }