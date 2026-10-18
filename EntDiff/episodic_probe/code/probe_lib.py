"""Shared config, IO and metrics for the episodic probe."""
import json, re
from collections import Counter
from pathlib import Path

# parents[2] is the package root; paths resolve relative to this file.
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
PROBE_DIR = ROOT / "episodic_probe"
ITEMS_DIR = PROBE_DIR / "items"        # probe questions + reference facts
ANSWERS_DIR = PROBE_DIR / "answers"    # model generations
JUDGE_DIR = PROBE_DIR / "judge"        # grading output, one dir per run
ARCHIVE_DIR = PROBE_DIR / "archive"    # superseded versions

CORPUS_FILE = DATA_DIR / "quality_corpus_clean.jsonl"

# ─── the three probed checkpoints (+ base control) ───────────────────────

SEED = 20260807

FAMILY = "episodic_context"


def cudnn_workaround():
    try:
        import torch as _t
        _t.backends.cuda.enable_cudnn_sdp(False)
    except Exception:
        pass


# ─── IO ──────────────────────────────────────────────────────────────────

def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def write_jsonl(path, rows, meta=None):
    with open(path, "w", encoding="utf-8") as f:
        if meta is not None:
            f.write(json.dumps({"_meta": meta}, ensure_ascii=False) + "\n")
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def iter_items(path):
    for r in load_jsonl(path):
        if "_meta" in r and len(r) == 1:
            continue
        yield r



def load_meta():
    return {it["article_id"]: {"title": it.get("title", ""),
                               "author": it.get("author", ""),
                               "text": it.get("text", "")}
            for it in load_jsonl(CORPUS_FILE)}



# ─── episodic_context prompt (this family only) ──────────────────────────
# Built from the item's own fields, NOT from the pre-baked `question` string, so
# the story is named in exactly ONE place and Qwen/Dream differ only by the
# per-article token in the header — the asymmetry actually being probed.

def article_header(title, author="", art_tok=None):
    """`Based on Article <tok> "TITLE" by AUTHOR,`"""
    by = (f" by {author}" if author and str(author).strip()
          and str(author).lower() != "unknown" else "")
    tok_part = f"{art_tok} " if art_tok else ""
    t = (title or "").strip().strip('"')   # one title is itself quoted in the corpus
    return f'Based on Article {tok_part}"{t}"{by},'


def epctx_prompt(it, art_tok=None):
    """Full user content for one episodic_context item."""
    head = article_header(it.get("title", ""), it.get("author", ""), art_tok)
    sent = " ".join((it.get("sentence") or "").split())
    # Many anchors are dialogue and already carry their own '"' at one or both
    # ends; wrapping blindly would emit `""`. Only add the quote where missing.
    op = "" if sent.startswith('"') else '"'
    cl = "" if sent.endswith('"') else '"'
    return (f'{head} what is the context that comes immediately '
            f'{it["type"]} the sentence: {op}{sent}{cl}?')


# ─── metrics ─────────────────────────────────────────────────────────────

def norm(s):
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", (s or "").lower())).strip()


def tok(s):
    return norm(s).split()


def exact_span_match(pred, gold):
    g = norm(gold)
    return 1.0 if g and g in norm(pred) else 0.0


def unigram_f1(pred, gold):
    p, g = tok(pred), tok(gold)
    if not p or not g:
        return 0.0
    cp, cg = Counter(p), Counter(g)
    common = sum((cp & cg).values())
    if common == 0:
        return 0.0
    prec, rec = common / len(p), common / len(g)
    return 2 * prec * rec / (prec + rec)


def _lcs(a, b):
    if not a or not b:
        return 0
    dp = [0] * (len(b) + 1)
    for x in a:
        prev = 0
        for j, y in enumerate(b, 1):
            cur = dp[j]
            dp[j] = prev + 1 if x == y else max(dp[j], dp[j - 1])
            prev = cur
    return dp[len(b)]


def rougeL(pred, gold):
    p, g = tok(pred), tok(gold)
    if not p or not g:
        return 0.0
    l = _lcs(p, g)
    prec, rec = l / len(p), l / len(g)
    return 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0


def lexical_metrics(pred, gold):
    return {"exact": exact_span_match(pred, gold),
            "rougeL": round(rougeL(pred, gold), 4),
            "uf1": round(unigram_f1(pred, gold), 4),
            "pred_ntok": len(tok(pred)), "gold_ntok": len(tok(gold))}



