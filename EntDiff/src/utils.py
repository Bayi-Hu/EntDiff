"""Utility helpers shared across diffusion training scripts.

Contains dataset-agnostic I/O, logging, metric, and metadata helpers.
Model / trainer / collator code stays in ``diffusion_train.py``.
"""

import json
from pathlib import Path


DATA_DIR = Path(__file__).resolve().parent.parent / "data"


# ─── stdout / stderr → console.log ──────────────────────────────────

class Tee:
    """Duplicate a stream to a log file so all prints and tracebacks
    are persisted alongside the run.
    """

    def __init__(self, log_path, stream):
        self.log_file = open(log_path, "a", encoding="utf-8")
        self.stream = stream

    def write(self, data):
        self.stream.write(data)
        self.log_file.write(data)
        self.log_file.flush()

    def flush(self):
        self.stream.flush()
        self.log_file.flush()

    def close(self):
        self.log_file.close()


# ─── JSONL loaders ───────────────────────────────────────────────────

def load_jsonl(path):
    """Load one JSON object per line from ``path``."""
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def _split_of(record):
    """Return the ``split`` field of a QA / corpus record, tolerating
    both top-level and ``metadata.split`` conventions.
    """
    return record.get("split") or record.get("metadata", {}).get("split")


def load_test_qa(qa_file, dataset):
    """Load the test-split QA rows for ``dataset``.

    QuALITY records carry ``metadata.split``; LiteraryQA records may carry
    a top-level ``split`` or none at all. Rows are kept if their split
    equals ``test``, or if there is no split field present.
    """
    items = load_jsonl(DATA_DIR / qa_file)
    kept = []
    for d in items:
        split = _split_of(d)
        if split is None or split == "test":
            kept.append(d)
    print(f"Loaded {len(kept)} test questions from {qa_file} (dataset={dataset})")
    return kept


# ─── Metric ──────────────────────────────────────────────────────────

def evaluate(predictions, label=None, dataset="quality"):
    """Compute total MC accuracy (single ``acc_all`` metric)."""
    correct = 0
    total = 0
    for p in predictions:
        total += 1
        if p["prediction"] == p.get("gold_letter"):
            correct += 1
    acc = correct / total if total else 0.0

    header = "  Results" + (f"  [{label}]" if label else "") + f"  (dataset={dataset})"
    print(f"\n{'=' * 60}")
    print(header)
    print(f"{'=' * 60}")
    print(f"  acc={acc:.3f} ({correct}/{total})")

    return {"acc_all": acc}


# ─── Corpus / meta helpers ───────────────────────────────────────────

def build_article_tokens(corpus):
    """Build the mapping ``article_id → "<article_id>"`` (per-article
    special token name). Deterministic (sorted by ``article_id``).
    """
    article_ids = sorted(set(item["article_id"] for item in corpus))
    return {aid: f"<{aid}>" for aid in article_ids}


def _meta_prefix(item):
    """Build a title / author preamble prepended to article-grounded
    prompts. Returns "" when neither is present.
    """
    parts = []
    if item.get("title"):
        parts.append(f"Title: {item['title']}")
    if item.get("author"):
        parts.append(f"Author: {item['author']}")
    return "\n".join(parts) + "\n" if parts else ""


def _get_nested(d, dotted):
    """Get a nested field via a dotted key (e.g., ``metadata.split``).

    Returns ``None`` when any intermediate lookup fails.
    """
    cur = d
    for key in dotted.split("."):
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
        if cur is None:
            return None
    return cur


def _filter_corpus_by_split(all_corpus, split_field, split_value):
    """Filter corpus rows by ``split_field == split_value``.

    Returns ``all_corpus`` unchanged when either ``split_field`` or
    ``split_value`` is ``None`` (dataset opts out of split filtering).
    """
    if split_field is None or split_value is None:
        return list(all_corpus)
    return [it for it in all_corpus if _get_nested(it, split_field) == split_value]
