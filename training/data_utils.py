"""Dataset readers for the supplied English MCQ + Hindi/Telugu SQuAD-style data."""
import json
from pathlib import Path

LANGUAGE_FILES = {
    "English": {
        "train": "data/English/train-00000-of-00001 (1).json",
        "validation": "data/English/validation-00000-of-00001 (1).json",
        "test": "data/English/test-00000-of-00001.json",
    },
    "Hindi": {"train": "data/Hindi/train.json", "validation": "data/Hindi/val.json", "test": "data/Hindi/test.json"},
    "Telugu": {"train": "data/Telugu/train.json", "validation": "data/Telugu/val.json", "test": "data/Telugu/test.json"},
}

def _choice_text(record):
    raw = record.get("choices")
    if not raw:
        return []
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            return []
    texts = raw.get("text", []) if isinstance(raw, dict) else []
    labels = raw.get("label", []) if isinstance(raw, dict) else []
    return [(str(labels[i]) if i < len(labels) else chr(65+i), str(t)) for i, t in enumerate(texts)]

def normalize_record(record, language):
    """Return (input_text, answer) or None for unanswerable/malformed examples."""
    question = str(record.get("question") or record.get("question_stem") or "").strip()
    if not question:
        return None
    if language == "English":
        choices = _choice_text(record)
        answer_key = str(record.get("answerKey") or "").strip()
        answer = ""
        for label, text in choices:
            if label.casefold() == answer_key.casefold():
                answer = text.strip()
                break
        if not answer and answer_key:
            # Some variants store the answer itself rather than a choice label.
            answer = next((t for _, t in choices if t.casefold() == answer_key.casefold()), "")
        if not answer:
            return None
        context = str(record.get("context") or "").strip()
        fact = str(record.get("fact1") or "").strip()
        options = "\n".join(f"{label}. {text}" for label, text in choices)
        context_parts = [p for p in [context, fact, ("Options:\n" + options if options else "")] if p]
        context = "\n".join(context_parts)
        if not context:
            context = "Options:\n" + options if options else ""
    else:
        context = str(record.get("context") or "").strip()
        answers = record.get("answers") or {}
        texts = answers.get("text", []) if isinstance(answers, dict) else []
        answer = str(texts[0]).strip() if texts and texts[0] is not None else ""
        if not context or not answer:
            return None
    if not context or not answer:
        return None
    prompt = f"answer the question in {language}. question: {question} context: {context}"
    return {"input_text": prompt, "target_text": answer, "language": language}

def load_split(split, data_root="data", languages=None, limit_per_language=0):
    languages = languages or ["English", "Hindi", "Telugu"]
    base = Path(data_root)
    rows = []
    counts = {}
    for language in languages:
        rel = LANGUAGE_FILES[language][split]
        path = base.parent / rel if not base.is_absolute() and str(base) == "data" else base / Path(rel).relative_to("data")
        # If data_root is the repository's data directory, paths above resolve correctly.
        if not path.exists():
            path = Path(rel)
        if not path.exists():
            raise FileNotFoundError(f"Dataset file not found: {path}. Run python scripts/prepare_data.py first.")
        with path.open("r", encoding="utf-8") as f:
            records = json.load(f)
        n_before = len(rows)
        for record in records:
            item = normalize_record(record, language)
            if item:
                rows.append(item)
                if limit_per_language and len(rows) - n_before >= limit_per_language:
                    break
        counts[language] = len(rows) - n_before
        print(f"{split}: {language}: loaded {counts[language]} answerable examples")
    if not rows:
        raise ValueError(f"No usable records found for split {split}")
    return rows, counts
