"""
probe_lib.py — shared config/IO/metrics for the episodic_probe suite
(full-265 QuALITY corpus).

ONE task: episodic_context — quote a sentence from a story, ask for the exact
context immediately before/after it (530 items = 265 articles x before/after).
Other families (completion / infill / attribution / temporal_order / ...) were
retired to archive/items_other_families/.

ONE prompt: epctx_prompt(). Earlier variants (duplicated title/author, trailing
"Give the exact words..." instruction) are gone — they are not reproducible from
this file, only from archive/.

Scoring: rougeL / uf1 / exact inline; Claude-judge (method-blind, 0/1/2 + halluc)
per judge/judge_protocol.md.
"""
import json, re
from collections import Counter
from pathlib import Path

# release 版:相对本文件定位,整个包可以整体搬走。
# episodic_probe/code/probe_lib.py → parents[2] 就是 release 根。
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
PROBE_DIR = ROOT / "episodic_probe"
ITEMS_DIR = PROBE_DIR / "items"        # 题库 (成品族)
ANSWERS_DIR = PROBE_DIR / "answers"    # 现役模型输出
JUDGE_DIR = PROBE_DIR / "judge"        # 判分协议 + 各 round
ARCHIVE_DIR = PROBE_DIR / "archive"    # 被取代的旧版本 (只移不删)

CORPUS_FILE = DATA_DIR / "quality_corpus_clean.jsonl"

# ─── the three probed checkpoints (+ base control) ───────────────────────
CKPT = {
    # ep40 run finished (MC 0.542@e40, peak 0.546@e27). The earlier checkpoint-epoch20
    # copy was wiped by HF Trainer checkpoint rotation (it globs checkpoint-*), so use final.
    "diffusion_c64": str(ROOT / "src_diffusion/diffusion_train/"
                         "fmtexp_uc2000_bs4x4_dream_ep40_seed42_chunk64/checkpoint_final"),
    "active_reading": str(ROOT / "src_baseline/cpt_align/"
                          "quality_active_reading_455M_lr2e-05_bs4x4_ep2_seed42_0803_2200/checkpoint_final"),
    "synthqa": str(ROOT / "src_baseline/cpt_align/"
                   "quality_synthqa_455M_lr2e-05_bs4x4_ep2_seed42_0803_2200/checkpoint_final"),
    "entigraph": str(ROOT / "src_baseline/cpt_align/"
                     "quality_entigraph_455M_lr2e-05_bs4x4_ep2_seed42_0803_2200/checkpoint_final"),
    # 续跑中; 用当前可用的 step-84000 存档 (epoch 1.78)
    "rephrase84k": str(ROOT / "src_baseline/cpt_align/"
                       "quality_rephrase_455M_lr2e-05_bs4x4_ep2_seed42_0803_2200/checkpoint-84000"),
    # UltraChat post-trained variants (instruct-only, 1 epoch, lr 5e-6, bs 4x4).
    # Same alignment data for every model, so what differs afterwards is what
    # injection stored rather than answering style.
    "synthqa_uc": str(ROOT / "src_baseline/cpt_align/"
                      "ucpost_synthqa_lr5e-6_bs4x4_ep1_seed42/checkpoint_final"),
    "entigraph_uc": str(ROOT / "src_baseline/cpt_align/"
                        "ucpost_entigraph_lr5e-6_bs4x4_ep1_seed42/checkpoint_final"),
    "active_reading_uc": str(ROOT / "src_baseline/cpt_align/"
                             "ucpost_active_reading_lr5e-6_bs4x4_ep1_seed42/checkpoint_final"),
    "dream_ep40_uc": str(ROOT / "src_diffusion/uc_post/"
                         "ucpost_dream_ep40_lr5e-6_bs4x4_ep1_seed42/checkpoint-750"),
    "base": "Qwen/Qwen2.5-7B-Instruct",

    # ─── len83 (83 books, 3.9k-225k words) ───────────────────────────────
    # A different experiment from the QuALITY rows above: different corpus,
    # different item set (items_len83_epctx.jsonl), so these never mix with the
    # 265-article results. Training length is NOT matched across these four and
    # any comparison has to say so:
    #   entigraph      2.00 epoch, ran to completion
    #   active_reading 1.48 epoch, died at step 66000 (exit 120, disk)
    #   span           18.3 epoch (step 138000), pre-resume checkpoint
    #   btok           19.9 epoch (step 150000), POST-resume checkpoint -- its
    #                  pre-resume copy was deleted during the 8/24 disk cleanup.
    #                  It did not degenerate after the resume (MC output stayed
    #                  A-D throughout, unlike `orig`, which emitted "the" on
    #                  2426/2452), but it is the only arm on the far side of that
    #                  boundary.
    "len83_entigraph": str(ROOT / "src_baseline/cpt_align/"
                           "len83_entigraph_518M_lr2e-05_bs4x4_ep2_seed42/checkpoint_final"),
    "len83_active_reading": str(ROOT / "src_baseline/cpt_align/"
                                "len83_active_reading_518M_lr2e-05_bs4x4_ep2_seed42/checkpoint-66000"),
    # 上面那条是 8/20 崩溃时的断点(epoch 1.48),当时 run 还没跑完,现有的 episodic 答案
    # 就是它生成的。2026-08-26 补跑到 89206 步(epoch 2.0)完成,checkpoint_final 才是
    # 与其余三臂(entigraph 用 final、span 用 138000、btok 用 150000 —— 各自的最终产出)
    # 对等的模型。旧条目保留,以便复核已发布的那批答案。
    # RawFT: 直接在 83 本书原文上 CPT 3 epoch(峰值位置,由 10-epoch 那轮定出:
    # MC 底座 0.4056 -> ep3 0.4387 峰 -> ep5 0.4268)。它是"不用任何合成语料"的对照。
    # 这个 checkpoint 是临时的 —— genQA 和 episodic probe 评完即删,只留评估产物。
    "len83_rawft_ep3": str(ROOT / "src_baseline/cpt_align/"
                           "len83_rawcorpus2048_lr2e-05_bs4x4_ep3_seed42/checkpoint_final"),
    "len83_active_reading_final": str(ROOT / "src_baseline/cpt_align/"
                                      "len83_active_reading_518M_lr2e-05_bs4x4_ep2_seed42/checkpoint_final"),
    # len83 的两个 CPT baseline。它们的 MC 一直在跑,但 episodic 从没探过 ——
    # cpt_align.py 里 literaryqa 的 episodic_items_file 是 None,训练中不接线,
    # 只能事后对 checkpoint_final 单独跑。synthqa 2 epoch 跑满 89,638 步收于
    # MC 0.5492(峰 0.5612 @24k);rephrase 95,858 步,跑完再补。
    "len83_synthqa_final": str(ROOT / "src_baseline/cpt_align/"
                               "len83_synthqa_518M_lr2e-05_bs4x4_ep2_seed42/checkpoint_final"),
    "len83_rephrase_final": str(ROOT / "src_baseline/cpt_align/"
                                "len83_rephrase_518M_lr2e-05_bs4x4_ep2_seed42/checkpoint_final"),
    # 20260826 那一轮 40-epoch 的终点。此前 len83 的 episodic 表里 CPT 各 arm 用的是
    # 训练终点,而 span/btok 用的是 20-epoch 中点(s150000 / s153000),口径不一致。
    # 补上这两个 final 之后,七个 arm 才能在同一条线上比。
    # span 不 desentinel(词表不改写正文,输出本来就是明文);btok 必须 desentinel
    # —— 它的词表里每个 (书,人物) 一个哨兵 token,生成出来是哨兵而 gold 是明文。
    "len83_span_final": str(ROOT / "src_diffusion/diffusion_train/20260826/"
                            "len83_dreamDNA_ENTSPAN_ep40_seed42/checkpoint_final"),
    "len83_btok_final": str(ROOT / "src_diffusion/diffusion_train/20260826/"
                            "len83_dreamDNA_BOOKTOK_ep40_seed42/checkpoint_final"),
    "len83_span": str(ROOT / "src_diffusion/diffusion_train/20260821/"
                      "len83_dreamDNA_ENTSPAN_ep80_seed42/checkpoint-138000"),
    "len83_btok": str(ROOT / "src_diffusion/diffusion_train/20260821/"
                      "len83_dreamDNA_BOOKTOK_ep80_seed42/checkpoint-150000"),
}

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



