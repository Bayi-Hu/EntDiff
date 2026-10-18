"""
diffusion_train_entity.py — diffusion_train.py plus the two entity
experiments, kept as a separate file so the baseline trainer can stay untouched
while jobs that already depend on it are running.

With ``--entity_mode none`` (the default) this is the baseline path unchanged.
``span`` masks each proper noun as a unit; ``token`` gives each multi-token
proper noun its own vocabulary entry. See entity_mask.py for why.

Overview
--------
Mirrors the style of `src_difussion/diffusion_instruct_qa_origin.py` (chat-
template wrapped corpus recitation + MC-letter + question→passage generation)
but is fully self-contained: every helper it needs from `diffusion_base`,
`diffusion_instruct`, and the origin qa_gen builders is inlined below. No
imports from those modules; no imports from a project-local `utils.py`.

Dataset dispatch
----------------
Both QuALITY and LiteraryQA are supported via ``--dataset {quality,literaryqa}``.
All file paths and split-filter conventions come from ``DATASETS[args.dataset]``.

Training tasks
--------------
  • corpus_chat  : "Recite the article <art_token>" → chunked story text
                   (masked at rate t ~ U(t_min, t_max)). Injects article
                   knowledge into the article's dedicated token embedding.
  • qa           : MC-letter — user turn with (A)(B)(C)(D), gold letter is
                   the only token that receives loss.
  • qa_gen       : free-form generation — question → relevant chunk_text,
                   art-token-grounded (from ``chunkq_file``).

Post-train mode
---------------
``--post_train`` runs an *instruct-only* variant (UltraChat instruction ↔
answer pairs, no corpus recitation, no MC-letter). Useful as a control that
teaches the answering *behavior* on top of an already-memorized checkpoint.

Recall eval
-----------
For each test QA item: (1) Dream diffusion-fills a recalled passage from the
``question`` cue conditioned on the article token, then (2) an untrained
Qwen2.5-Instruct judge takes (meta, recalled_passage, question, options) and
returns a letter A-D that is compared against ``gold_letter``.
"""

import argparse
import json
import os
import random
import re
import shutil
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.distributions as dists
import torch.nn as nn
import torch.nn.functional as F
from datasets import Dataset

import entity_mask
from transformers import AutoModel, AutoTokenizer, Trainer, TrainerCallback, set_seed
from trl import SFTConfig

from utils import (
    DATA_DIR,
    Tee,
    _meta_prefix,
    build_article_tokens,
    evaluate,
    load_jsonl,
)


# ─── Constants + dataset registry ────────────────────────────────────

# Hardcoded external QA judge: untrained Qwen2.5-Instruct on recalled content.
# Probes recall quality independent of trained Dream's in-corpus memorization.
QWEN_JUDGE_MODEL = "Qwen/Qwen2.5-7B-Instruct"

# ── RELEASE NOTE ─────────────────────────────────────────────────────
# Only two entries below have their data shipped in this release:
#
#   quality_deanaph        QuALITY,     265 articles, 1.51M tokens
#   literaryqa50_deanaph   NarrativeQA,  50 books,    5.49M tokens
#
# The other entries are kept so the registry matches the code used for the
# reported experiments (and so the ablation names in the paper resolve), but
# their corpus files are not distributed -- selecting one will fail on a
# missing file in data/. The two shipped corpora are the de-anaphorised
# versions; their unmodified counterparts (quality_corpus_clean.jsonl,
# lit_len50_corpus_clean.jsonl) ship too, since the episodic probe reads the
# original text.
# ─────────────────────────────────────────────────────────────────────
DATASETS = {
    "quality": {
        "corpus_file":         "quality_corpus_clean.jsonl",
        "qa_test_file":        "quality_qa.jsonl",
        "chunkq_file":         "chunk_query/chunk_queries_quality_all_chunk64.jsonl",
        "episodic_items_file": "items_epctx_1000.jsonl",
    },
    "literaryqa": {
        "corpus_file":         "literaryqa_exp50_corpus.jsonl",
        "qa_test_file":        "literaryqa_exp50_mc.jsonl",
        "chunkq_file":         "chunk_query/chunk_queries_literaryqa_exp50_chunk64.jsonl",
        "episodic_items_file": None,
    },
    # Scale-control: same code / same chunk-query task / same 853 recall QAs as
    # `quality`, but memorize only the original 50 sample50 articles (vs 186) —
    # to test whether the recall gap (0.431 vs 0.508) is purely data scale.
    # corpus + chunk_queries are subsets of the `quality` files.
    "literaryqa40": {
        "corpus_file":         "lit_exp40_corpus.jsonl",
        "qa_test_file":        "lit_exp40_mc.jsonl",
        "chunkq_file":         "chunk_query/chunk_queries_lit_exp40_chunk64.jsonl",
        "episodic_items_file": None,
    },
    # Regenerated MC for the same 40 books, built under the 18-book protocol
    # (gold_answer = answers[0] with only the trailing period stripped). The old
    # lit_exp40_mc.jsonl rewrote 20.8% of its golds and left them systematically
    # half a word longer than the distractors; this one does neither.
    # De-anaphorised lit_exp40: same 40 books, same MC questions, same chunk size.
    # Third-person pronouns whose referent is a named character were replaced with
    # the name (14,668 edits, verified byte-for-byte against the original except at
    # pronoun positions), so a 64-word training window can be attributed to a
    # character 87% of the time instead of 72%. The chunk_query file is regenerated
    # from this text -- reusing the original one would feed old prose through qa_gen
    # while corpus_chat feeds new prose.
    "literaryqa40_deanaph": {
        "corpus_file":         "lit_exp40_corpus_deanaph.jsonl",
        "qa_test_file":        "lit_exp40_mc.jsonl",
        "chunkq_file":         "chunk_query/chunk_queries_lit_exp40_deanaph_chunk64.jsonl",
        "episodic_items_file": None,
    },
    "literaryqa40_v2": {
        "corpus_file":         "lit_exp40_corpus.jsonl",
        "qa_test_file":        "lit_exp40_mc_v2.jsonl",
        "chunkq_file":         "chunk_query/chunk_queries_lit_exp40_chunk64.jsonl",
        "episodic_items_file": None,
    },
    # The 83 books that survived the corpus/QA consistency audit, out of the
    # length-stratified 100. LiteraryQA inherits NarrativeQA's title-matching
    # errors -- One Shot/One-Shot, Fugitive Pieces/Fugitive Poetry -- where the
    # questions belong to a different book than the text.
    "literaryqa83": {
        "corpus_file":         "lit_len83_corpus.jsonl",
        "qa_test_file":        "lit_len83_mc.jsonl",
        "chunkq_file":         "chunk_query/chunk_queries_lit_len83_chunk64.jsonl",
        "episodic_items_file": None,
    },
    # Same 83 books with subject and object pronouns resolved to names by Claude:
    # 141,553 edits, 46.1% of he/she/him/her-objective replaced. Possessives and
    # reflexives are untouched by construction, so there is no "Name's" to form and
    # none of the morphology damage the Qwen versions carried (v2 broke grammar on
    # 39.5% of its edits; this measures 100% grammatical, 99.3% correct referent, on
    # a 150-item audit with the referent and grammar axes judged separately).
    #
    # chunkq_file points at the ORIGINAL len83 queries on purpose. This entry exists
    # for the --chunkq_per_chunk 0 ablation, where the file is read and validated but
    # contributes no samples. If chunk queries are ever switched back on for this
    # corpus they must be regenerated from the de-anaphorised text first -- otherwise
    # qa_gen feeds old prose while corpus_chat feeds new, which is the mistake
    # literaryqa40_deanaph was set up to avoid.
    # Same corpus and questions, with every person rewritten to a token that
    # belongs to one book: the Mary of lit_1050 and the Mary of lit_7481 are
    # different tokens rather than one shared row. Places, organisations and
    # book-specific peoples keep the global atomic tokens, because a place
    # really is the same place across books -- splitting London 44 ways would
    # divide 1,653 occurrences into tokens of 38 and Europe into tokens of 8.
    #
    # Both files come from build_booktok_data.py. The MC file has its question
    # and option text rewritten but gold_letter and gold_answer untouched, and
    # the genqa reference answers live in literaryqa_qa.jsonl, which is never
    # rewritten -- so scoring compares plain names against plain names once
    # entity_mask.desentinel has undone the substitution in the prediction.
    "literaryqa83_booktok": {
        "corpus_file":         "lit_len83_corpus_deanaph_v5_booktok.jsonl",
        "qa_test_file":        "lit_len83_mc_booktok.jsonl",
        "chunkq_file":         "chunk_query/chunk_queries_lit_len83_chunk64.jsonl",
        "episodic_items_file": None,
    },
    "literaryqa83_deanaph": {
        "corpus_file":         "lit_len83_corpus_deanaph_v5.jsonl",
        "qa_test_file":        "lit_len83_mc.jsonl",
        "chunkq_file":         "chunk_query/chunk_queries_lit_len83_chunk64.jsonl",
        "episodic_items_file": None,
    },
    # len83 按分层均匀降采到 50 本 (每层 10 本, seed 42) 的去代词版。
    # 5,403,185 tok (len83 的 52.2%),MC 1,460 题,与 AR 侧 literaryqa_len50 用的是
    # 同一份 lit_len50_mc.jsonl,所以两边的 MC 计分口径完全一致,曲线可以直接比。
    #
    # entity_vocab 由 --entity_vocab 传 len50dna_entity_vocab.json —— 3,682 个名字
    # (2,656 个多 token),是 len83 那份 6,068 个按 50 本过滤来的,keep 仍是
    # PERSON/PLACE/ORG/BOOK_ENTITY。**必须显式传**:参数默认值指向 len83 的文件,
    # 不传会拿 83 本的词表去 mask 50 本的语料,不报错但 span 覆盖是错的。
    #
    # chunkq_file 设 None 而不是挂 len83 那份:那份 queries 是对原始(未去代词)文本生成的,
    # 挂着它等于埋一个静默陷阱 —— 谁把 --chunkq_per_chunk 调回 1,就会拿旧文本喂 qa_gen、
    # 新文本喂 corpus_chat,不报错但结果是错的。
    # episodic_items_file 同样设 None:len50 的探针题集 items_len50_epctx.jsonl 虽然在盘上,
    # 但锚点句取自原始文本,训练中在线跑会有同样的新旧错配;探针照旧对 checkpoint 事后单独跑。
    "literaryqa50_deanaph": {
        "corpus_file":         "lit_len50_corpus_deanaph_v5.jsonl",
        "qa_test_file":        "lit_len50_mc.jsonl",
        "chunkq_file":         None,
        "episodic_items_file": None,
    },
    # QuALITY 265 篇,人称代词由 Claude 解析成名字后的版本(流程见 deanaph_claude.py):
    # 18,034 个待判槽位,13,025 处实际替换,"64 词窗口内没有任何人名"的窗口降了 31.1%
    # —— 64 正是注入时的 chunk 大小,所以这个窗口就是模型每次实际看到的上下文。5 条安全
    # 护栏里 name_not_in_book / possessive_her / in_compound 在 15,152 个答案上一次都没触发。
    #
    # article_id 与 quality_qa.jsonl 的 corpus_id 完全对齐(265/265),MC 题面未改写,
    # 所以评测口径和上面 "quality" 那一条一致,唯一差别是训练语料换成去代词版。
    #
    # entity_vocab 由 --entity_vocab 传入,对应 data/quality_dna_entity_vocab_reviewed.json:
    # 规则粗筛 3,923 个候选,agent 逐词分 10 类后保留 PERSON/PLACE/ORG/BOOK_ENTITY 共
    # 2,332 个(59.4%,与 len83 的 57.3% 同口径),其中 1,659 个多 token。审查砍掉的 33%
    # 出现次数里有 3,563 次是 I'm / I'll / I've 这类缩写 —— 它们全是多 token,不砍会严重
    # 污染 span 实验。
    #
    # chunkq_file 是 None,不是漏填:本方法跑的是 --chunkq_per_chunk 0,qa_gen 一条样本
    # 都不产生,上面的组装代码会直接短路、根本不读这个文件。这里刻意**不**像
    # literaryqa83_deanaph 那样挂一个原始(未去代词)的 queries 文件 —— 挂着它等于埋一个
    # 静默陷阱:谁把 --chunkq_per_chunk 调回 1,就会拿旧文本喂 qa_gen、新文本喂
    # corpus_chat,不报错但结果是错的。真要为这个语料开 chunk query,先从去代词文本重新
    # 生成一份,再把路径填到这里。
    #
    # episodic_items_file 先设 None,与 literaryqa83_deanaph 一致:探针 item 的锚点句同样
    # 取自原始文本,训练中在线跑会有和 chunkq 相同的新旧错配。要打开就改成
    # "items_epctx_1000.jsonl"(文件在 episodic_probe/items/),是一行的事;在那之前
    # episodic 结果照旧对 checkpoint 事后单独跑,和 len83 的做法相同。
    "quality_deanaph": {
        "corpus_file":         "quality_corpus_deanaph.jsonl",
        "qa_test_file":        "quality_qa.jsonl",
        "chunkq_file":         None,
        "episodic_items_file": None,
    },
    # d27(span)的 book-token 对照臂。同一份去代词语料,但正文里每个 (篇, 人物) 被换成
    # 专属哨兵 token —— art_0 的 Mary 和 art_7 的 Mary 是不同 token,身份不再需要模型
    # 从上下文推断。只拆 PERSON:地名跨篇是同一个地方,拆开只会把出现次数摊到训不动。
    # build_booktok_data.py 生成,实测替换掉 40,669/41,448 = 98.1% 的实体出现,
    # 1,966 个人物哨兵 + 780 个非人物全局名 = 2,746 个新 token。
    #
    # qa_test_file 的题面和选项同样被改写,但 gold_letter / gold_answer 未动,所以 MC
    # 的计分口径和 d27 完全一致,两条曲线可以直接比。
    # chunkq_file / episodic_items_file 与 quality_deanaph 同样设 None,理由见上一条。
    "quality_deanaph_booktok": {
        "corpus_file":         "quality_corpus_deanaph_booktok.jsonl",
        "qa_test_file":        "quality_qa_booktok.jsonl",
        "chunkq_file":         None,
        "episodic_items_file": None,
    },
    "quality_sample50": {
        "corpus_file":         "quality_corpus_sample50_clean.jsonl",
        "qa_test_file":        "quality_qa_sample50.jsonl",  # 853, MC + recall
        "chunkq_file":         "chunk_query/chunk_queries_sample50_chunk64.jsonl",
        "episodic_items_file": None,   # probe spans all 265; this trains on 50
    },
}


# ─── Model loading (Dream-7B + article tokens) ───────────────────────

def load_dream_model_with_tokens(model_name, corpus_file, max_chunk_words):
    """Load Dream-7B and add per-article special tokens.

    Returns:
        model, tokenizer, article_token_ids, mask_token_id
    where ``article_token_ids`` maps ``article_id → token_id`` for the newly
    added tokens. If the tokenizer lacks a native mask token, a ``<mask>``
    token is added.
    """
    print(f"Loading Dream model: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"

    corpus = load_jsonl(DATA_DIR / corpus_file)
    art_token_map = build_article_tokens(corpus)

    new_tokens = list(art_token_map.values())
    mask_token_id = getattr(tokenizer, "mask_token_id", None)
    if mask_token_id is None:
        new_tokens.append("<mask>")
        print("  [warn] tokenizer has no built-in mask_token; adding '<mask>'")

    n_added = tokenizer.add_tokens(new_tokens, special_tokens=True)
    print(f"  Added {n_added} tokens "
          f"({len(art_token_map)} article"
          f"{' + 1 <mask>' if mask_token_id is None else ''})")

    article_token_ids = {
        aid: tokenizer.convert_tokens_to_ids(token_name)
        for aid, token_name in art_token_map.items()
    }
    if mask_token_id is None:
        mask_token_id = tokenizer.convert_tokens_to_ids("<mask>")
    print(f"  mask_token_id = {mask_token_id}  "
          f"({tokenizer.convert_ids_to_tokens(mask_token_id)})")

    model = AutoModel.from_pretrained(
        model_name, device_map="auto", trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    print(f"  Loaded model ({type(model).__name__}, "
          f"{model.num_parameters():,} params)")

    old_vocab_size = model.config.vocab_size
    model.resize_token_embeddings(len(tokenizer))

    with torch.no_grad():
        embed_weight = model.get_input_embeddings().weight
        std = embed_weight[:old_vocab_size].std().item()
        nn.init.normal_(embed_weight[old_vocab_size:], mean=0.0, std=std)
        lm_head = model.get_output_embeddings()
        if lm_head is not None and lm_head.weight.data_ptr() != embed_weight.data_ptr():
            lm_std = lm_head.weight[:old_vocab_size].std().item()
            nn.init.normal_(lm_head.weight[old_vocab_size:], mean=0.0, std=lm_std)

    print(f"  Vocab: {old_vocab_size} → {len(tokenizer)} (+{n_added})")

    # Dream's DreamGenerationConfig.validate() rejects the strict= kwarg newer
    # transformers passes; patch it so save_pretrained doesn't crash. (Same
    # patch as RECALL/src_difussion/diffusion_instruct_qa.py — applied here at
    # load time because save_strategy="steps" saves DURING training.)
    if getattr(model, "generation_config", None) is not None:
        gc_cls = type(model.generation_config)
        if not getattr(gc_cls, "_recall_validate_patched", False):
            _orig = gc_cls.validate
            def _patched(self, *a, **kw):
                kw.pop("strict", None)
                return _orig(self, *a, **kw)
            gc_cls.validate = _patched
            gc_cls._recall_validate_patched = True

    return model, tokenizer, article_token_ids, mask_token_id


# ─── Raw-text QA content (used inside chat-wrap for MC) ──────────────

def build_qa_content_with_meta(art_token_str, question, options, meta_prefix=""):
    """Build the free-text QA prompt body (article/meta/question/options).

    This is the *content* of the user turn in the chat-wrapped MC task; the
    chat wrapping is done separately by ``apply_chat_template``.
    """
    context = (
        f"{art_token_str}\n{meta_prefix}".rstrip()
        if meta_prefix else art_token_str
    )
    options_text = "\n".join(f"({letter}) {text}" for letter, text in sorted(options.items()))
    return (
        f"Based on the following context:\n\n"
        f"{context}\n\n"
        f"Question: {question}\n\n"
        f"{options_text}\n\n"
        f"Answer with the letter of the correct option (A, B, C, or D) and nothing else."
    )


# ─── Dream trainer plumbing: attention mask + logit shift + loss ─────

def _to_dream_attn_mask(attention_mask_2d):
    """Convert HF 2D long attention_mask [B, L] → SDPA-compatible 4D bool
    [B, 1, 1, L]. Dream's modeling forwards ``attention_mask`` straight to
    SDPA without dtype/shape conversion, so we do it here.
    """
    if attention_mask_2d is None:
        return None
    return attention_mask_2d[:, None, None, :].bool()


def _dream_shift_logits(logits):
    """Re-align Dream logits so position ``i`` predicts token ``i``.

    Dream uses Qwen2.5's AR LM head, so the raw output ``logits[i]`` is the
    next-token distribution at position ``i + 1``. Both training and
    inference apply the same left-pad-by-one shift.
    """
    return torch.cat([logits[:, 0:1], logits[:, :-1]], dim=1).contiguous()


class DreamMLMTrainer(Trainer):
    """MLM-style loss for Dream diffusion CPT.

    Steps: shift logits (Dream AR-head convention) → per-token CE ignoring
    ``-100`` → optional ``1/t`` re-weighting → per-sample SUM → batch mean.
    ``loss_weight_t`` is a class-level toggle so callers can flip it without
    re-instantiating the trainer.
    """

    loss_weight_t = False

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        t_per_sample = inputs.pop("t_per_sample", None)
        attn_mask = _to_dream_attn_mask(inputs.get("attention_mask"))
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=attn_mask,
        )
        logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
        logits = _dream_shift_logits(logits)

        B, L, V = logits.shape
        per_token_ce = F.cross_entropy(
            logits.reshape(-1, V).float(),
            labels.reshape(-1),
            ignore_index=-100,
            reduction="none",
        ).view(B, L)

        if self.loss_weight_t and t_per_sample is not None:
            t = t_per_sample.to(per_token_ce.device).clamp(min=1e-3)
            weight = (1.0 / t)[:, None].expand_as(per_token_ce)
        else:
            weight = torch.ones_like(per_token_ce)

        loss = (per_token_ce * weight).sum(dim=-1).mean()

        return (loss, outputs) if return_outputs else loss


def load_test_qa(qa_file, dataset=None):
    """Every MC question in the file, both splits. Only the QA mixed into
    training is split-filtered; scoring covers the whole injected corpus.
    `dataset` is accepted for call-site compatibility and unused."""
    rows = [q for q in load_jsonl(DATA_DIR / qa_file)
            if q.get("options") and q.get("gold_letter")]
    print(f"MC eval: {len(rows)} questions from {qa_file} (all splits)")
    return rows


# ─── Base diffusion masking collator ─────────────────────────────────

class DiffusionMaskingCollator:
    """Base collator: raw ``[art_token | chunk_ids]`` corpus samples, masked at
    a per-sample rate ``t ~ U(t_min, t_max)``. Subclasses add branches keyed on
    ``task`` (see ``ChatDiffusionMaskingCollator``).
    """

    def __init__(self, pad_token_id, mask_token_id, max_length,
                 t_min=0.05, t_max=0.95):
        self.pad_token_id = pad_token_id
        self.mask_token_id = mask_token_id
        self.max_length = max_length
        self.t_min = t_min
        self.t_max = t_max

    def __call__(self, features):
        processed = []
        t_per_sample = []
        for f in features:
            chunk_ids = list(f["chunk_ids"])
            art_token_id = f["art_token_id"]
            meta_len = int(f.get("meta_len", 0))
            ids = ([art_token_id] + chunk_ids)[:self.max_length]
            n = len(ids)
            t = random.uniform(self.t_min, self.t_max)
            input_ids = list(ids)
            labels = [-100] * n
            mask_start = 1 + meta_len          # art token + meta prefix stay visible
            for i in range(min(mask_start, n), n):
                if random.random() < t:
                    labels[i] = ids[i]
                    input_ids[i] = self.mask_token_id
            processed.append({"input_ids": input_ids, "labels": labels})
            t_per_sample.append(float(t))

        if not processed:
            return {}

        max_len = max(len(p["input_ids"]) for p in processed)
        padded_input_ids, padded_labels, padded_attn = [], [], []
        for p in processed:
            pad_len = max_len - len(p["input_ids"])
            padded_input_ids.append(p["input_ids"] + [self.pad_token_id] * pad_len)
            padded_labels.append(p["labels"] + [-100] * pad_len)
            padded_attn.append([1] * len(p["input_ids"]) + [0] * pad_len)

        return {
            "input_ids": torch.tensor(padded_input_ids, dtype=torch.long),
            "labels": torch.tensor(padded_labels, dtype=torch.long),
            "attention_mask": torch.tensor(padded_attn, dtype=torch.long),
            "t_per_sample": torch.tensor(t_per_sample, dtype=torch.float),
        }


# ─── Chat-wrapped collator: corpus_chat ──────────────────────────────

class ChatDiffusionMaskingCollator(DiffusionMaskingCollator):
    """Extends the base collator with a ``corpus_chat`` branch.

    ``corpus_chat`` samples arrive pre-assembled: ``input_ids`` already
    contains the chat-templated user prompt + assistant-header prefix
    (length ``protect_len``) followed by the content tokens to be masked.
    """

    def __call__(self, features):
        processed = []
        t_per_sample = []
        for f in features:
            task = f["task"]

            if task == "corpus_chat":
                ids = list(f["input_ids"])[:self.max_length]
                n = len(ids)
                protect_len = int(f["protect_len"])
                t = random.uniform(self.t_min, self.t_max)
                input_ids = list(ids)
                labels = [-100] * n
                mask_start = min(protect_len, n)
                for i in range(mask_start, n):
                    if random.random() < t:
                        labels[i] = ids[i]
                        input_ids[i] = self.mask_token_id
                processed.append({"input_ids": input_ids, "labels": labels})
                t_per_sample.append(float(t))

            else:
                raise ValueError(f"Unknown task type: {task!r}")

        if not processed:
            return {}

        max_len = max(len(p["input_ids"]) for p in processed)
        padded_input_ids, padded_labels, padded_attn = [], [], []
        for p in processed:
            pad_len = max_len - len(p["input_ids"])
            padded_input_ids.append(p["input_ids"] + [self.pad_token_id] * pad_len)
            padded_labels.append(p["labels"] + [-100] * pad_len)
            padded_attn.append([1] * len(p["input_ids"]) + [0] * pad_len)

        return {
            "input_ids": torch.tensor(padded_input_ids, dtype=torch.long),
            "labels": torch.tensor(padded_labels, dtype=torch.long),
            "attention_mask": torch.tensor(padded_attn, dtype=torch.long),
            "t_per_sample": torch.tensor(t_per_sample, dtype=torch.float),
        }


# ─── qa_gen-aware collator: adds a `qa_gen` branch ───────────────────

class InstructQADiffusionMaskingCollator(ChatDiffusionMaskingCollator):
    """Adds a ``qa_gen`` branch (identical masking rules to ``corpus_chat``:
    protect the prompt prefix, mask each assistant-answer position i.i.d.
    with probability ``t`` sampled per sample).
    """

    def __call__(self, features):
        processed = []
        t_per_sample = []
        for f in features:
            task = f["task"]

            if task in ("corpus_chat", "qa_gen"):
                ids = list(f["input_ids"])[:self.max_length]
                n = len(ids)
                protect_len = int(f["protect_len"])
                t = random.uniform(self.t_min, self.t_max)
                input_ids = list(ids)
                labels = [-100] * n
                mask_start = min(protect_len, n)
                for i in range(mask_start, n):
                    if random.random() < t:
                        labels[i] = ids[i]
                        input_ids[i] = self.mask_token_id
                processed.append({"input_ids": input_ids, "labels": labels})
                t_per_sample.append(float(t))

            else:
                raise ValueError(f"Unknown task type: {task!r}")

        if not processed:
            return {}

        max_len = max(len(p["input_ids"]) for p in processed)
        padded_input_ids, padded_labels, padded_attn = [], [], []
        for p in processed:
            pad_len = max_len - len(p["input_ids"])
            padded_input_ids.append(p["input_ids"] + [self.pad_token_id] * pad_len)
            padded_labels.append(p["labels"] + [-100] * pad_len)
            padded_attn.append([1] * len(p["input_ids"]) + [0] * pad_len)

        return {
            "input_ids": torch.tensor(padded_input_ids, dtype=torch.long),
            "labels": torch.tensor(padded_labels, dtype=torch.long),
            "attention_mask": torch.tensor(padded_attn, dtype=torch.long),
            "t_per_sample": torch.tensor(t_per_sample, dtype=torch.float),
        }


# ─── corpus_chat builder (chat-template wrapped recitation) ──────────

def build_corpus_data_chat(tokenizer, article_token_ids, corpus_items,
                           max_chunk_words=64, max_length=4096, seed=None):
    """Build chat-wrapped corpus recitation samples.

    Per-sample layout after ``apply_chat_template(add_generation_prompt=True)``:

        <|im_start|>user
        Recite the article <art_token>
        Title: ...
        Author: ...<|im_end|>
        <|im_start|>assistant
        {content tokens}          ← masked at rate t

    ``protect_len`` is the length of the prompt through the assistant header;
    the collator masks only positions ``i >= protect_len``.
    """
    rng = random.Random(seed) if seed is not None else None

    formatted = []
    for item in corpus_items:
        article_id = item["article_id"]
        art_token_id = article_token_ids.get(article_id)
        if art_token_id is None:
            continue

        art_token_str = tokenizer.convert_ids_to_tokens(art_token_id)
        meta_prefix = _meta_prefix(item).rstrip()
        user_msg = f"Recite the article {art_token_str}\n{meta_prefix}"

        prompt_ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": user_msg}],
            tokenize=True, add_generation_prompt=True,
        )
        protect_len = len(prompt_ids)

        words = item["text"].split()
        n_words = len(words)
        if rng is not None:
            offset = rng.randint(0, max(0, max_chunk_words - 1))
        else:
            offset = 0

        word_chunks = []
        if offset > 0:
            word_chunks.append(words[:offset])
        for i in range(offset, n_words, max_chunk_words):
            word_chunks.append(words[i:i + max_chunk_words])
        if not word_chunks:
            word_chunks = [[]]

        for wc in word_chunks:
            content_text = " ".join(wc)
            content_ids = tokenizer.encode(content_text, add_special_tokens=False)
            input_ids = (list(prompt_ids) + content_ids)[:max_length]
            formatted.append({
                "task": "corpus_chat",
                "input_ids": input_ids,
                "protect_len": protect_len,
            })
    return formatted


# ─── MC-letter builder (chat-template wrapped) ───────────────────────

def build_qa_content_generic(question, options):
    """MC prompt body WITHOUT any article/context (generic MC, e.g. synthetic
    calibration questions that aren't tied to a corpus article)."""
    options_text = "\n".join(f"({letter}) {text}"
                             for letter, text in sorted(options.items()))
    return (
        f"Question: {question}\n\n"
        f"{options_text}\n\n"
        f"Answer with the letter of the correct option (A, B, C, or D) and nothing else."
    )


# ─── instruct 样本构建(chat-wrapped)──────────────────────────────────
# 原来这一节还有 build_chunk_query_gen_data_chat(question → chunk_text),已随
# qa_gen 一起移除:本方法不合成问答样本,只训练语料本身 + 少量 instruct 防遗忘。

def build_instruct_gen_data_chat(tokenizer, instruct_file, seed=42,
                                 max_length=4096, limit=0):
    """Build free-form instruction→answer samples from a general instruction set
    (Alpaca / UltraChat-style JSONL: ``instruction`` / ``input`` / ``output``).

    No article token is used — this teaches the *generic* answering behavior.
    Article grounding happens elsewhere via ``corpus_chat`` memorization.
    """
    items = load_jsonl(DATA_DIR / instruct_file)
    rng = random.Random(seed)
    rng.shuffle(items)

    formatted = []
    for r in items:
        instr = (r.get("instruction") or "").strip()
        inp = (r.get("input") or "").strip()
        answer = (r.get("output") or r.get("response") or r.get("answer") or "").strip()
        if not instr or not answer:
            continue

        user_msg = instr if not inp else f"{instr}\n\n{inp}"
        prompt_ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": user_msg}],
            tokenize=True, add_generation_prompt=True,
        )
        answer_ids = tokenizer.encode(answer, add_special_tokens=False)
        input_ids = (list(prompt_ids) + answer_ids)[:max_length]
        if len(input_ids) <= len(prompt_ids):
            continue
        formatted.append({
            "task": "qa_gen",
            "input_ids": input_ids,
            "protect_len": len(prompt_ids),
        })
        if limit and len(formatted) >= limit:
            break
    print(f"Instruct-gen (chat-wrapped): {len(formatted)} samples from {instruct_file}")
    return formatted


# ─── Dream sampling helpers (mirror generation_utils.py) ─────────────

def _top_p_logits(logits, top_p):
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cum = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    remove = cum > top_p
    remove[..., 1:] = remove[..., :-1].clone()
    remove[..., 0] = 0
    mask = torch.zeros_like(logits, dtype=torch.bool).scatter_(-1, sorted_indices, remove)
    return logits.masked_fill(mask, torch.finfo(logits.dtype).min)


def _sample_tokens(logits, temperature=0.0, top_p=None,
                   margin_confidence=False, neg_entropy=False):
    if temperature > 0:
        logits = logits / temperature
    if top_p is not None and top_p < 1:
        logits = _top_p_logits(logits, top_p)
    probs = torch.softmax(logits, dim=-1)
    if temperature > 0:
        try:
            x0 = dists.Categorical(probs=probs).sample()
            confidence = torch.gather(probs, -1, x0.unsqueeze(-1)).squeeze(-1)
        except Exception:
            confidence, x0 = probs.max(dim=-1)
    else:
        confidence, x0 = probs.max(dim=-1)
    if margin_confidence:
        sp, _ = torch.sort(probs, dim=-1, descending=True)
        confidence = sp[:, 0] - sp[:, 1]
    if neg_entropy:
        confidence = torch.sum(probs * torch.log(probs + 1e-10), dim=-1)
    return confidence, x0


# ─── External Qwen-Instruct judge ────────────────────────────────────

class QwenInstructJudge:
    """Untrained Qwen2.5-Instruct judge. Loaded lazily on first
    ``predict_letter`` call. Provides an ``unload`` method to free VRAM.
    """

    def __init__(self, model_name, device=None):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    def lazy_load(self):
        if self.model is not None:
            return
        from transformers import AutoModelForCausalLM
        print(f"  Loading external judge: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name, device_map="auto",
            torch_dtype=torch.bfloat16, trust_remote_code=True,
        ).eval()
        print(f"    Loaded ({self.model.num_parameters():,} params)")

    def unload(self):
        """Release the judge weights and clear the CUDA cache."""
        if self.model is not None:
            del self.model
            self.model = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @torch.no_grad()
    def predict_letter(self, meta_prefix, passage, question, options,
                       max_new_tokens=16):
        self.lazy_load()
        context = (meta_prefix.rstrip() + "\n" + passage) if meta_prefix else passage
        opts = "\n".join(f"({letter}) {text}" for letter, text in sorted(options.items()))
        content = (
            f"Based on the following context:\n\n{context}\n\n"
            f"Question: {question}\n\n{opts}\n\n"
            f"Answer with the letter of the correct option (A, B, C, or D) and nothing else."
        )
        input_ids = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=True, add_generation_prompt=True, return_tensors="pt",
        ).to(self.model.device)
        out = self.model.generate(
            input_ids, attention_mask=torch.ones_like(input_ids),
            max_new_tokens=max_new_tokens, do_sample=False,
            pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
        )
        text = self.tokenizer.decode(
            out[0][input_ids.shape[1]:], skip_special_tokens=True,
        ).strip()
        m = re.match(r"^\(?([A-Da-d])\)?", text) or re.search(r"\b([A-Da-d])\b", text)
        return (m.group(1).upper() if m else (text[:1].upper() if text else "A")), text


# ─── Direct QA: Dream single-step letter prediction ──────────────────

@torch.no_grad()
def qa_select_dream(model, tokenizer, art_token_str, question, options,
                    mask_token_id, meta_prefix=""):
    """Chat-template MC eval: wrap the QA content in the chat template, append
    a single ``<mask>`` at the assistant's first-output position, take the
    argmax, and extract the A-D letter.

    MC format is never taught during training, so this is zero-shot format
    handling off the instruct prior — the same ask the AR baselines get.
    """
    qa_content = build_qa_content_with_meta(
        art_token_str, question, options, meta_prefix,
    )
    prompt_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": qa_content}],
        tokenize=True, add_generation_prompt=True,
    )
    mask_pos = len(prompt_ids)
    input_ids = list(prompt_ids) + [mask_token_id]

    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=model.device)
    outputs = model(input_ids=input_tensor)
    logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
    logits = _dream_shift_logits(logits)
    pred_token_id = int(logits[0, mask_pos].argmax().item())
    raw_token = tokenizer.decode([pred_token_id], skip_special_tokens=False)
    text = tokenizer.decode([pred_token_id], skip_special_tokens=True).strip()

    m = re.match(r"^\(?([A-Da-d])\)?", text)
    if m:
        return m.group(1).upper(), raw_token
    m = re.search(r"\b([A-Da-d])\b", text)
    if m:
        return m.group(1).upper(), raw_token
    return (text[:1].upper() if text else "A"), raw_token


@torch.no_grad()
def qa_select_dream_gen(model, tokenizer, art_token_str, question, options,
                        mask_token_id, meta_prefix="", n_gen=6, n_steps=6, eps=1e-3):
    """MC eval by GENERATION (mirrors lit_mc_eval's Dream path): append
    ``n_gen`` masks after the cue, diffusion-fill them ('ar' schedule), decode,
    and ``re.search`` the first A-D letter.

    Unlike ``qa_select_dream`` (single-mask argmax), the model writes out the
    option content, so the chosen letter is content-driven. This avoids the
    single-token failure mode where an un-injected model emits ``<|endoftext|>``
    (scoring ~chance) and blunts the answer-order prior that biases the
    single-token argmax.
    """
    qa_content = build_qa_content_with_meta(
        art_token_str, question, options, meta_prefix,
    )
    prompt_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": qa_content}],
        tokenize=True, add_generation_prompt=True,
    )
    gen_ids = diffusion_fill_from_cue(
        model, list(prompt_ids), mask_token_id,
        n_gen, n_steps, 0.0, "ar", None, None, eps,
    )
    text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
    # Robust letter extraction: prefer "(A)" then "answer is X" / ": X", then a
    # standalone \b[A-D]\b. A naive re.search("[A-D]") would wrongly grab the
    # "A" of "Answer" or the "C" of "Correct" in preamble like "The answer is B.".
    m = (re.search(r"\(([A-Da-d])\)", text)
         or re.search(r"(?:answer|option)\D{0,6}\(?([A-Da-d])\b", text, re.I)
         or re.search(r":\s*\(?([A-Da-d])\b", text)
         or re.search(r"\b([A-Da-d])\b", text))
    return (m.group(1).upper() if m else "A"), text[:40]


# ─── Diffusion fill from cue (recall Stage 2) ────────────────────────

@torch.no_grad()
def diffusion_fill_from_cue(model, cue_ids, mask_id, n_gen, n_steps, temperature,
                            alg, alg_temp, top_p, eps):
    """Fill ``n_gen`` masked positions after a cue prefix.

    Supported algorithms:
      • ``ar``          — left-to-right one-token-at-a-time unmasking; the
                          full ``[cue, MASK*n_gen]`` stays in-distribution
                          for Dream. Requires ``n_gen`` forward passes.
      • ``origin``      — LLaDA-style random transfer schedule.
      • ``maskgit_plus`` / ``topk_margin`` / ``entropy`` — confidence-based
                          batched unmasking.
    """
    device = next(model.parameters()).device
    cue_len = len(cue_ids)
    x = torch.tensor([list(cue_ids) + [mask_id] * n_gen], dtype=torch.long, device=device)
    timesteps = torch.linspace(1.0, eps, n_steps + 1, device=device)

    if alg == "ar":
        for _ in range(n_gen):
            mask_index = (x == mask_id)
            if not mask_index.any():
                break
            logits = _dream_shift_logits(model(input_ids=x).logits)
            first_mask_pos = mask_index[0].nonzero(as_tuple=True)[0][0].item()
            pos_logits = logits[0, first_mask_pos].unsqueeze(0)
            _, sampled = _sample_tokens(pos_logits, temperature, top_p)
            x[0, first_mask_pos] = sampled.item()
        return x[0, cue_len:].tolist()

    for i in range(n_steps):
        mask_index = (x == mask_id)
        if not mask_index.any():
            break
        logits = _dream_shift_logits(model(input_ids=x).logits)
        mask_logits = logits[mask_index]
        t, s = timesteps[i], timesteps[i + 1]
        if alg == "origin":
            p_transfer = 1 - s / t if i < n_steps - 1 else 1.0
            x0 = torch.full_like(x[mask_index], mask_id)
            sel = torch.rand(*x0.shape, device=device) < p_transfer
            if sel.any():
                _, sampled = _sample_tokens(mask_logits[sel], temperature, top_p)
                x0[sel] = sampled
            x[mask_index] = x0
            continue
        if alg == "maskgit_plus":
            confidence, x0 = _sample_tokens(mask_logits, temperature, top_p)
        elif alg == "topk_margin":
            confidence, x0 = _sample_tokens(
                mask_logits, temperature, top_p, margin_confidence=True,
            )
        elif alg == "entropy":
            confidence, x0 = _sample_tokens(
                mask_logits, temperature, top_p, neg_entropy=True,
            )
        else:
            raise ValueError(f"Unknown alg: {alg}")
        n_mask = mask_index.sum() / mask_index.shape[0]
        n_transfer = int(n_mask * (1 - s / t)) if i < n_steps - 1 else int(n_mask)
        if n_transfer == 0:
            continue
        full = torch.full_like(x, -float("inf"), dtype=logits.dtype)
        full[mask_index] = confidence
        if alg_temp is None or alg_temp == 0:
            _, transfer_idx = torch.topk(full, n_transfer)
        else:
            transfer_idx = torch.multinomial(
                F.softmax(full / alg_temp, dim=-1), num_samples=n_transfer,
            )
        x_ = torch.full_like(x, mask_id)
        x_[mask_index] = x0
        rows = torch.arange(x.size(0), device=device).unsqueeze(1).expand_as(transfer_idx)
        x[rows, transfer_idx] = x_[rows, transfer_idx]
    return x[0, cue_len:].tolist()


# ─── Recall pipeline: Dream generate → Qwen judge ────────────────────

@torch.no_grad()
def eval_recall_qwen_judge(model, tokenizer, test_qa, article_token_ids,
                           corpus_meta, mask_token_id,
                           n_gen_tokens, n_steps, temperature,
                           alg, alg_temp, top_p, eps,
                           label="epoch", qwen_judge=None):
    """Single-cue recall eval with an external Qwen judge.

    Per QA item:
      1. Dream diffusion-fills a recalled passage from the ``question`` cue,
         conditioned on the article's special token.
      2. The Qwen judge sees (meta, recalled_passage, question, options) and
         emits a single letter A-D, compared against ``gold_letter``.

    Returns ``(aggregate_metrics, per_QA_records)``. The judge is unloaded at
    the end to free VRAM before training resumes.
    """
    model.eval()
    records = []
    n_correct = 0
    n_total = len(test_qa)
    t0 = time.time()

    def _build_cue(art_token_str, meta_prefix_str, query):
        """Return prompt token ids for the recall cue: user message
        "Based on the article <art>\\n{meta}\\n\\n{query}" through
        <|im_start|>assistant\\n. ``diffusion_fill_from_cue`` then appends
        ``n_gen_tokens`` masks and fills them.
        """
        user_msg = f"Based on the article {art_token_str}\n{meta_prefix_str}\n\n{query}"
        prompt_ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": user_msg}],
            tokenize=True, add_generation_prompt=True,
        )
        return list(prompt_ids)

    for i, q in enumerate(test_qa):
        aid = q["corpus_id"]
        art_id = article_token_ids.get(aid)
        if art_id is None:
            continue

        item = corpus_meta[aid]
        meta_str = _meta_prefix(item)
        art_token_str = tokenizer.convert_ids_to_tokens(art_id)
        question = q["question"]
        options = q.get("options") or {}

        cue = _build_cue(art_token_str, meta_str.rstrip(), question)
        gen_ids = diffusion_fill_from_cue(
            model, cue, mask_token_id,
            n_gen_tokens, n_steps, temperature, alg, alg_temp, top_p, eps,
        )
        recalled = tokenizer.decode(gen_ids, skip_special_tokens=True)

        pred_letter, _ = qwen_judge.predict_letter(
            meta_str, recalled, question, options,
        )

        gold = q.get("gold_letter")
        correct = int(bool(gold) and pred_letter == str(gold).upper())
        n_correct += correct

        records.append({
            "id": q.get("id"),
            "corpus_id": aid,
            "question": question,
            "gold_letter": gold,
            "options": options,
            "recalled": recalled,
            "pred_qwen": pred_letter,
            "correct_qwen": correct,
        })
        if (i + 1) % 500 == 0:
            print(f"    [{i + 1}/{n_total}] {time.time() - t0:.1f}s")

    aggregate = {
        "n": n_total,
        "acc_qwen": (n_correct / n_total) if n_total > 0 else 0.0,
    }

    print(f"\n  Recall via Qwen judge ({label}, n={n_total}):")
    print(f"    acc_qwen: {aggregate['acc_qwen']:.4f}  ({n_correct}/{n_total})")

    if qwen_judge is not None:
        qwen_judge.unload()

    return aggregate, records


# ─── Run name ────────────────────────────────────────────────────────

def build_run_name(args):
    """Compose a run name that encodes dataset + core hyperparams. The
    dataset prefix keeps runs from different datasets disambiguated on disk.
    """
    ts = datetime.now().strftime("%m%d_%H%M")
    rtag = "rfull" if args.recall_n_eval_samples <= 0 else f"r{args.recall_n_eval_samples}"
    mode = "_post" if args.post_train else ""
    return (
        f"{args.dataset}{mode}"
        f"_t{args.t_min}-{args.t_max}"
        f"_lr{args.lr}_bs{args.batch_size}x{args.grad_accum}_ep{args.epochs}_{rtag}"
        f"_chunk{args.max_chunk_words}_seed{args.random_seed}_{ts}"
    )


# ─── Main ────────────────────────────────────────────────────────────

def main():
    # H100 + recent torch: cuDNN SDPA backend fails with "No valid execution
    # plans" on Dream's attention. Fall back to math / memory-efficient SDPA.
    try:
        torch.backends.cuda.enable_cudnn_sdp(False)
    except Exception:
        pass

    parser = argparse.ArgumentParser(
        description="Dream-Instruct diffusion CPT trainer (self-contained), "
                    "supporting QuALITY + LiteraryQA via --dataset."
    )
    parser.add_argument("--dataset", type=str, default="quality",
                        choices=list(DATASETS.keys()))
    parser.add_argument("--run_name", type=str, default=None,
                        help="Override the auto-generated run name. Use a FIXED "
                             "name (no timestamp) so restarts land in the same "
                             "output_dir and --resume_from auto can continue.")
    parser.add_argument("--out_root", type=str, default="./diffusion_train",
                        help="Root dir for run outputs; output_dir = out_root/run_name. "
                             "Pass e.g. ./diffusion_train/20260807 to date-organize. "
                             "Default keeps the historical flat layout. Pin it (don't "
                             "recompute the date on resume) so --resume_from auto works.")
    parser.add_argument("--model", type=str, default="Dream-org/Dream-v0-Instruct-7B")

    parser.add_argument("--instruct_file", type=str, default="instruct_ultrachat.jsonl")
    parser.add_argument("--chunkq_per_chunk", type=int, default=1)
    parser.add_argument("--chunkq_select", choices=["first", "random"], default="first",
                        help="Which of a chunk's generated questions to keep. "
                             "'first' (default, historical) is a prefix slice and is "
                             "biased to who-questions by the generator's fixed ordering; "
                             "'random' samples uniformly so all question types appear.")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--max_seq_length", type=int, default=4096)
    parser.add_argument("--max_chunk_words", type=int, default=64)
    parser.add_argument("--entity_mode",
                        choices=["none", "span", "token", "booktok"],
                        default="none",
                        help="none = baseline; span = mask each proper noun as a "
                             "unit; token = one vocab entry per multi-token name")
    parser.add_argument("--entity_token_scope", choices=["all", "multi"],
                        default="all",
                        help="token mode only: which names get an atomic token. "
                             "'all' keeps the treatment uniform across the "
                             "single- vs multi-token strata; 'multi' converts "
                             "only the names that currently cost >1 token")
    parser.add_argument("--booktok_vocab", default="len83dna_booktok_vocab.json",
                        help="booktok mode only: written by build_booktok_data.py")
    parser.add_argument("--entity_vocab", default="len83_entity_vocab.json",
                        help="built by build_entity_vocab.py, read from DATA_DIR")
    parser.add_argument("--dynamic_chunking", action="store_true",
                        help="Re-chunk the corpus before every epoch with a fresh "
                             "per-article random word offset, so chunk boundaries "
                             "move instead of staying fixed for the whole run.")
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--t_min", type=float, default=0.05)
    parser.add_argument("--t_max", type=float, default=0.95)
    parser.add_argument("--qa_eval_every_n_epochs", type=int, default=1)
    parser.add_argument("--eval_every_n_steps", type=int, default=0,
                        help="If >0, run MC+recall eval every N optimizer steps "
                             "instead of at epoch boundaries (used for short "
                             "post-train runs where 1 epoch spans many steps).")
    parser.add_argument("--save_every_n_epochs", type=int, default=20,
                        help="Save a (model-only) resumable checkpoint every N "
                             "epochs.")
    parser.add_argument("--save_every_n_steps", type=int, default=0,
                        help="If >0, save a (model-only) resumable checkpoint every "
                             "N optimizer steps (overrides the epoch-based save "
                             "cadence; needed in step-eval mode where epoch saves "
                             "are otherwise disabled).")
    parser.add_argument("--no_save", action="store_true",
                        help="Never write any checkpoint — disables periodic saves "
                             "AND the final checkpoint_final. Eval-only run; NOT "
                             "resumable past a wall.")
    parser.add_argument("--chunkq_file", type=str, default=None,
                        help="Override dataset chunk-query file (e.g. test-only injection).")
    # Answer format comes from UltraChat alone; training on gold MC letters
    # would inflate MC accuracy without adding knowledge.
    parser.add_argument("--n_instruct_mix", type=int, default=0,
                        help="If >0, mix this many general UltraChat instruction "
                             "samples into training (from --instruct_file). Keeps "
                             "the model's instruction-following / answer-format "
                             "ability without using task-specific MC QA.")
    parser.add_argument("--disable_recall", action="store_true",
                        help="Never run the recall (Dream-fill + Qwen judge) eval, "
                             "even in step-eval mode.")
    # Default is argmax: one mask, one forward, read the letter. `gen` fills
    # mc_gen_tokens masks and re-searches the text, which is 20x the forward
    # passes (3235 MC items took ~19 min per eval vs ~1 min) and is a different
    # measurement from the first-position argmax the fmt-exp runs were scored
    # with -- numbers from the two modes are not comparable.
    parser.add_argument("--mc_decode", type=str, default="argmax",
                        choices=["gen", "argmax", "both"],
                        help="MC letter decode: 'gen' = diffusion-fill n_gen tokens "
                             "then re.search A-D (content-driven, matches lit_mc_eval; "
                             "avoids endoftext/single-token bias). 'argmax' = legacy "
                             "single-mask first-position argmax.")
    parser.add_argument("--mc_gen_tokens", type=int, default=20,
                        help="n_gen for --mc_decode gen (masks filled per MC item).")
    parser.add_argument("--save_total_limit", type=int, default=1,
                        help="How many recent checkpoints to keep on disk "
                             "(each ~15G for 7B). >1 lets you inspect several "
                             "training stages to locate convergence.")
    parser.add_argument("--resume_from", type=str, default=None,
                        help="Checkpoint dir to resume from, or 'auto' to pick "
                             "up the latest checkpoint in the run's output_dir.")
    # Post-train mode: instruct-only, no corpus, no MC letter.
    parser.add_argument("--post_train", action="store_true")
    # Recall pipeline
    parser.add_argument("--recall_n_gen_tokens", type=int, default=100)
    parser.add_argument("--recall_n_steps", type=int, default=100)
    parser.add_argument("--recall_temperature", type=float, default=0.0)
    parser.add_argument("--recall_alg", type=str, default="ar",
                        choices=["origin", "entropy", "maskgit_plus", "topk_margin", "ar"])
    parser.add_argument("--recall_alg_temp", type=float, default=None)
    parser.add_argument("--recall_top_p", type=float, default=None)
    parser.add_argument("--recall_eval_every_n_epochs", type=int, default=20)
    # Episodic probe: same prompt / gold / mask budget as the standalone
    # eval_dream.py, so mid-training points are comparable to post-hoc ones.
    parser.add_argument("--genqa_every_n_steps", type=int, default=0,
                        help="Free-form generation every N steps. Needed when the "
                             "run uses --eval_every_n_steps, because that mode "
                             "returns early from on_epoch_end and the epoch-based "
                             "cadence never fires.")
    parser.add_argument("--genqa_every_n_epochs", type=int, default=0,
                        help="Free-form LiteraryQA generation every N epochs.")
    parser.add_argument("--genqa_at_epochs", type=str, default="",
                        help="Comma-separated epochs; overrides the stride.")
    parser.add_argument("--genqa_skip_pretrain", action="store_true",
                        help="skip the free-form genqa at epoch0_pretrain "
                             "(22 minutes on Dream, measuring the untrained "
                             "base model). MC and recall at pretrain are "
                             "unaffected.")
    parser.add_argument("--genqa_n_samples", type=int, default=0,
                        help="0 = every question in the MC file, the same set the "
                             "MC eval scores. Positive takes a fixed subsample.")
    parser.add_argument("--genqa_max_new_tokens", type=int, default=32,
                        help="Mask budget. References are short: median 3 words.")
    parser.add_argument("--episodic_at_epochs", type=str, default="",
                        help="Comma-separated epochs to probe at, e.g. 2,5. "
                             "Overrides --episodic_every_n_epochs; use it when "
                             "the points wanted are not evenly spaced and each "
                             "probe costs 1038 open-ended generations.")
    parser.add_argument("--episodic_items_file", type=str, default=None,
                        help="覆盖 DATASETS[dataset]['episodic_items_file']。加这个开关是因为"
                             "训练中的 episodic 探针只从 ds_cfg 取题库(:1752),没有 CLI 入口;"
                             "quality_deanaph 那条把它设成 None(去代词语料的探针 item 尚未重新"
                             "生成),但按原文本的 item 去探是 len83 那批 DNA 模型一直在用的口径,"
                             "要跨数据集可比就得能显式传进来。")
    parser.add_argument("--episodic_every_n_epochs", type=int, default=0,
                        help="Run the episodic probe every N epochs (0 = off). "
                             "Also fires at epoch0_pretrain and at the final epoch.")
    parser.add_argument("--episodic_skip_pretrain", action="store_true",
                        help="Skip the episodic probe at epoch0_pretrain. The "
                             "untrained point is the same for every run, so it is "
                             "worth the GPU time only once.")
    parser.add_argument("--recall_n_eval_samples", type=int, default=-1)
    parser.add_argument("--recall_qa_file", type=str, default=None,
                        help="if set, recall eval uses ALL QAs from this file "
                             "(e.g. quality_qa_sample50.jsonl) instead of "
                             "sampling from qa_test_file")
    args = parser.parse_args()
    args.recall_eps = 1e-3

    ds_cfg = DATASETS[args.dataset]
    set_seed(args.random_seed)

    run_name = args.run_name or build_run_name(args)
    args.output_dir = f"{args.out_root.rstrip('/')}/{run_name}"
    os.makedirs(args.output_dir, exist_ok=True)

    log_path = os.path.join(args.output_dir, "console.log")
    tee_out = Tee(log_path, sys.stdout)
    tee_err = Tee(log_path, sys.stderr)
    sys.stdout = tee_out
    sys.stderr = tee_err

    config = vars(args).copy()
    config["run_name"] = run_name
    config["timestamp"] = datetime.now().isoformat()
    config["dataset_config"] = ds_cfg
    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2, default=str)

    src_dir = Path(__file__).resolve().parent
    snapshot_dir = os.path.join(args.output_dir, "src_snapshot")
    os.makedirs(snapshot_dir, exist_ok=True)
    for py_file in src_dir.glob("*.py"):
        shutil.copy2(py_file, snapshot_dir)

    print(f"Run name: {run_name}")
    print(f"Output dir: {args.output_dir}")
    print(f"Dataset: {args.dataset}")

    # ─── Model + article tokens ─────────────────────────────────────
    model, tokenizer, article_token_ids, mask_token_id = load_dream_model_with_tokens(
        args.model, ds_cfg["corpus_file"], args.max_chunk_words,
    )

    # ─── Entity experiments ─────────────────────────────────────────
    # Loaded here, after the article tokens exist, so `token` mode resizes on
    # top of them rather than fighting them for row indices.
    entity_vocab = None
    if args.entity_mode in ("span", "token"):
        entity_vocab = entity_mask.load_entity_vocab(DATA_DIR / args.entity_vocab)
        print(f"Entity mode: {args.entity_mode}  "
              f"({len(entity_vocab['per_book'])} books, "
              f"{len(entity_vocab['multi_tok']):,} multi-token names)")
    elif args.entity_mode == "booktok":
        print(f"Entity mode: booktok")
        if args.entity_mode == "booktok":
            bt = entity_mask.load_booktok_vocab(DATA_DIR / args.booktok_vocab)
            entity_mask.add_booktok_tokens(model, tokenizer, bt)
        elif args.entity_mode == "token":
            key = "all_tok" if args.entity_token_scope == "all" else "multi_tok"
            forms = entity_vocab[key]
            print(f"  scope={args.entity_token_scope}: {len(forms):,} names")
            entity_mask.add_entity_tokens(model, tokenizer, forms)

    # ─── Corpus + meta ──────────────────────────────────────────────
    # All articles are memorized: chunk-query already covers the whole corpus,
    # so split-filtering here only halved exposure, it never held anything out.
    all_corpus = load_jsonl(DATA_DIR / ds_cfg["corpus_file"])
    corpus_meta = {item["article_id"]: item for item in all_corpus}
    mem_corpus = all_corpus
    print(f"Corpus: {len(mem_corpus)} articles to memorize (all)")

    # ─── Recall pipeline prep (Qwen judge) ──────────────────────────
    qwen_judge = QwenInstructJudge(QWEN_JUDGE_MODEL)
    print(f"External Qwen judge: {QWEN_JUDGE_MODEL} (lazy load)")

    # ─── Training data assembly ─────────────────────────────────────
    # chunk-query 那一路(qa_gen)已从本脚本移除。本方法只训练语料本身 + 少量 instruct
    # 防遗忘,不再合成 question→passage 样本;所有在跑的 run 都传 --chunkq_per_chunk 0,
    # 那条分支产出恒为 [],是死代码。留着它的代价不只是白读一遍 chunkq 文件,更是配置里
    # 必须挂一个用不上的路径 —— 而去代词语料的 chunk queries 尚未重新生成,谁把
    # --chunkq_per_chunk 调回非 0,就会拿旧文本喂 qa_gen、新文本喂 corpus_chat,不报错
    # 但结果是错的(literaryqa40_deanaph 踩过这个坑)。
    #
    # --chunkq_per_chunk / --chunkq_file / --chunkq_select 三个参数**保留但不再生效**:
    # 正在跑的 d26_span40 / d26_btok40 命令行里带着 --chunkq_per_chunk 0,链式续跑时会
    # 重新解析本文件的 argparse,删掉参数会让 resume 直接报 unrecognized argument 而死。
    # 传 0 静默通过(与移除前逐字等价),传非 0 立刻报错而不是悄悄忽略。
    if not args.post_train and args.chunkq_per_chunk:
        raise SystemExit(
            f"--chunkq_per_chunk {args.chunkq_per_chunk}: chunk-query 生成已从本脚本移除,"
            f"只接受 0。需要这条路径请用 diffusion_train.py,或先从当前语料重新生成 "
            f"chunk queries 再把该功能单独实现在新脚本里。")

    # post_train 模式:instruct-only,不用语料。
    instruct_only_data = build_instruct_gen_data_chat(
        tokenizer, args.instruct_file, seed=args.random_seed,
        max_length=args.max_seq_length,
    ) if args.post_train else []

    def make_corpus_data(epoch):
        """Rebuild corpus_chat data for the given epoch. In post-train mode,
        no corpus is used (returns empty list).
        """
        if args.post_train:
            return []
        # seed=None keeps offset 0 for every article (the fixed-chunk behaviour).
        # With --dynamic_chunking each epoch gets its own seed, so every article
        # draws a fresh offset in [0, max_chunk_words-1] and the chunk boundaries
        # shift; a fact that sat mid-chunk in one epoch straddles a boundary in
        # the next.
        epoch_seed = (args.random_seed * 10000 + epoch) if args.dynamic_chunking else None
        if args.entity_mode == "span":
            return entity_mask.build_corpus_data_chat_entity(
                tokenizer, article_token_ids, mem_corpus, entity_vocab,
                _meta_prefix, max_chunk_words=args.max_chunk_words,
                max_length=args.max_seq_length, seed=epoch_seed,
            )
        return build_corpus_data_chat(
            tokenizer, article_token_ids, mem_corpus,
            max_chunk_words=args.max_chunk_words,
            max_length=args.max_seq_length, seed=epoch_seed,
        )

    # Optional: mix in general UltraChat instruction samples to preserve the
    # model's instruction-following / answer-format ability without task-specific
    # MC QA. (post_train mode already uses the full instruct set, so skip there.)
    instruct_mix = []
    if not args.post_train and args.n_instruct_mix > 0:
        instruct_mix = build_instruct_gen_data_chat(
            tokenizer, args.instruct_file, seed=args.random_seed,
            max_length=args.max_seq_length,
            limit=args.n_instruct_mix,
        )

    corpus_data = make_corpus_data(epoch=0)
    all_data = corpus_data + instruct_only_data + instruct_mix
    random.seed(args.random_seed)
    random.shuffle(all_data)
    dataset = Dataset.from_list(all_data)
    print(f"Total: {len(all_data)} ({len(corpus_data)} corpus_chat + "
          f"{len(instruct_only_data)} instruct_only + {len(instruct_mix)} instruct)")

    # Save a resumable (model-only) checkpoint every `save_every_n_epochs`
    # epochs, keeping only the latest on disk. Full optimizer states for a 7B
    # model are ~56GB and would blow the disk quota, so we save model weights
    # only; with a constant_with_warmup LR (warmup done in epoch 0) resuming
    # with a fresh optimizer is a negligible perturbation.
    eff_bs = args.batch_size * args.grad_accum
    steps_per_epoch = max(1, (len(all_data) + eff_bs - 1) // eff_bs)
    save_steps = max(1, steps_per_epoch * args.save_every_n_epochs)
    print(f"Steps/epoch={steps_per_epoch}; checkpointing every "
          f"{args.save_every_n_epochs} epochs = every {save_steps} steps "
          f"(model-only, keep latest {args.save_total_limit})")

    training_args = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="constant_with_warmup",
        warmup_steps=args.warmup_steps,
        weight_decay=0.01,
        bf16=True,
        logging_steps=50,
        # Short step-eval (post-train) runs don't wall and don't need resumable
        # checkpoints; disabling the periodic/end-of-training HF checkpoint
        # avoids a redundant ~15G save on top of the explicit checkpoint_final.
        # If --save_every_n_steps is given, checkpoint on that step cadence (works
        # even in step-eval mode). Otherwise keep the epoch-based behaviour, which
        # disables periodic saves during short step-eval runs.
        save_strategy=("no" if args.no_save else
                       ("steps" if (args.save_every_n_steps > 0 or args.eval_every_n_steps <= 0) else "no")),
        save_steps=(args.save_every_n_steps if args.save_every_n_steps > 0 else save_steps),
        save_total_limit=args.save_total_limit,
        save_only_model=True,
        report_to="none",
        run_name=run_name,
        seed=args.random_seed,
        remove_unused_columns=False,
    )

    test_qa = load_test_qa(ds_cfg["qa_test_file"])
    if args.recall_qa_file:
        # Recall on a fixed QA file (ALL of it, no sampling) — e.g. the 853
        # sample50 QAs, so the recall curve is directly comparable to the
        # earlier sample50 experiment.
        recall_test_qa = load_jsonl(DATA_DIR / args.recall_qa_file)
        print(f"Recall eval: ALL {len(recall_test_qa)} QAs from "
              f"{args.recall_qa_file}")
    elif args.recall_n_eval_samples > 0 and args.recall_n_eval_samples < len(test_qa):
        recall_rng = random.Random(42)
        recall_test_qa = recall_rng.sample(test_qa, args.recall_n_eval_samples)
        print(f"Recall eval subset: {len(recall_test_qa)} / {len(test_qa)} QAs")
    else:
        recall_test_qa = list(test_qa)
        print(f"Recall eval subset: {len(recall_test_qa)} / {len(test_qa)} QAs")

    cli_args = args

    class EvalCallback(TrainerCallback):
        """Runs MC-letter (``run_qa``) and recall pipeline (``run_recall``)
        evals at configurable cadences. Also rotates the training dataset
        each epoch when dynamic chunking is on.
        """

        _last_preds = None
        trainer = None

        def _run_eval(self, label, hf_args, model, state=None,
                      run_qa=True, run_recall=True, run_episodic=False,
                      run_genqa=False):
            model.eval()
            model.to("cuda")
            t0 = time.time()
            preds = None

            if run_qa:
                print(f"\n>>> {label} evaluation "
                      f"(direct QA — Dream letter prediction)")
                preds = []
                for i, q in enumerate(test_qa):
                    aid = q["corpus_id"]
                    art_id = article_token_ids.get(aid)
                    options = q.get("options") or {}
                    if art_id is None or not options:
                        prediction, raw = "A", ""
                    else:
                        art_token_str = tokenizer.convert_ids_to_tokens(art_id)
                        meta = _meta_prefix(corpus_meta.get(aid, {}))
                        pred_gen = pred_am = None; raw = ""; raw_am = ""
                        if cli_args.mc_decode in ("gen", "both"):
                            pred_gen, raw = qa_select_dream_gen(
                                model, tokenizer, art_token_str,
                                q["question"], options, mask_token_id,
                                meta_prefix=meta, n_gen=cli_args.mc_gen_tokens,
                                n_steps=cli_args.mc_gen_tokens, eps=cli_args.recall_eps,
                            )
                        if cli_args.mc_decode in ("argmax", "both"):
                            pred_am, raw_am = qa_select_dream(
                                model, tokenizer, art_token_str,
                                q["question"], options, mask_token_id,
                                meta_prefix=meta,
                            )
                        # primary prediction: gen if computed, else argmax
                        prediction = pred_gen if pred_gen is not None else pred_am
                    rec = {
                        "id": q.get("id"),
                        "corpus_id": aid,
                        "question": q["question"],
                        "options": options,
                        "gold_letter": q.get("gold_letter"),
                        "answer": q.get("answer") or q.get("gold_answer"),
                        "prediction": prediction,
                        "raw_output": raw if raw else raw_am,
                        "metadata": q.get("metadata", {}),
                    }
                    if cli_args.mc_decode == "both":
                        rec["prediction_argmax"] = (pred_am if art_id is not None
                                                    and options else "A")
                        rec["raw_argmax"] = raw_am
                    preds.append(rec)
                    if (i + 1) % 500 == 0:
                        print(f"    [{i + 1}/{len(test_qa)}] "
                              f"{time.time() - t0:.1f}s")
                metrics = evaluate(preds, label=label, dataset=cli_args.dataset)
                if cli_args.mc_decode == "both":
                    na = sum(1 for p in preds if p["prediction"] in "ABCD"
                             and p["prediction"] == p.get("gold_letter"))
                    aa = sum(1 for p in preds if p.get("prediction_argmax") in "ABCD"
                             and p.get("prediction_argmax") == p.get("gold_letter"))
                    agree = sum(1 for p in preds
                                if p["prediction"] == p.get("prediction_argmax"))
                    tot = len(preds) or 1
                    print(f"  [decode compare] gen acc={na/tot:.4f}  "
                          f"argmax acc={aa/tot:.4f}  agreement={agree/tot:.4f}")
                path = os.path.join(hf_args.output_dir, f"acc_{label}.jsonl")
                with open(path, "w", encoding="utf-8") as f:
                    for p in preds:
                        f.write(json.dumps(p, ensure_ascii=False) + "\n")
                print(f"  Saved to {path}")
                self.__class__._last_preds = preds
            else:
                print(f"\n>>> {label} QA eval SKIPPED; "
                      f"using last preds for direct lookup")
                preds = self._last_preds or []

            if run_recall:
                print(f"\n>>> {label} recall-pipeline eval "
                      f"(Dream cue → Qwen judge)")
                try:
                    agg, recs = eval_recall_qwen_judge(
                        model, tokenizer, recall_test_qa, article_token_ids,
                        corpus_meta, mask_token_id,
                        cli_args.recall_n_gen_tokens, cli_args.recall_n_steps,
                        cli_args.recall_temperature, cli_args.recall_alg,
                        cli_args.recall_alg_temp, cli_args.recall_top_p,
                        cli_args.recall_eps,
                        label=label, qwen_judge=qwen_judge,
                    )
                except Exception as e:
                    # Recall runs a 2nd 7B (Qwen judge) alongside Dream +
                    # optimizer states; if it OOMs or errors, skip it rather
                    # than crash the run — MC metric + checkpoints are intact.
                    import traceback
                    traceback.print_exc()
                    print(f">>> [WARN] recall eval failed ({e}); skipping recall "
                          f"this round and continuing training")
                    try:
                        qwen_judge.unload()
                    except Exception:
                        pass
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    agg, recs = {"acc_qwen_q": None, "error": str(e)}, []
                rec_path = os.path.join(hf_args.output_dir, f"recall_{label}.jsonl")
                _meta = {"_meta": {
                    "pipeline": "diffusion_train_recall_qwen_judge",
                    "dataset": cli_args.dataset,
                    "label": label,
                    "qa_test_file": ds_cfg["qa_test_file"],
                    "judge": QWEN_JUDGE_MODEL,
                    "n_gen_tokens": cli_args.recall_n_gen_tokens,
                    "n_steps": cli_args.recall_n_steps,
                    "alg": cli_args.recall_alg,
                    "alg_temp": cli_args.recall_alg_temp,
                    "eps": cli_args.recall_eps,
                    "temperature": cli_args.recall_temperature,
                    "top_p": cli_args.recall_top_p,
                    "recall_n_eval_samples": cli_args.recall_n_eval_samples,
                    "corpus_injection": "chat_template_assistant_wrap+qa_gen_instruct",
                    "instruct_file": cli_args.instruct_file,
                    **agg,
                }}
                with open(rec_path, "w", encoding="utf-8") as f:
                    f.write(json.dumps(_meta, ensure_ascii=False, default=str) + "\n")
                    for r in recs:
                        f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
                print(f"  Saved recall to {rec_path}")
            else:
                print(f">>> {label} recall-pipeline eval SKIPPED")

            if run_episodic:
                try:
                    self._run_episodic(label, model, hf_args)
                except Exception as e:
                    # Same policy as recall: never kill a 12-hour run over eval.
                    import traceback
                    traceback.print_exc()

            if run_genqa:
                try:
                    self._run_genqa(label, model, hf_args)
                except Exception:
                    # Same policy: an eval must never end a training run.
                    import traceback
                    traceback.print_exc()
                    print(f">>> [WARN] episodic probe failed ({e}); skipping this "
                          f"round and continuing training")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
            else:
                print(f">>> {label} episodic probe SKIPPED")

            model.train()

        def on_train_begin(self, hf_args, state, control, model=None, **kwargs):
            # HF restores global_step before this fires, so on a wall-resume the
            # weights are mid-training and none of these are "pretrain": running
            # them would overwrite the real baseline files with the wrong point.
            if state.global_step > 0:
                print(f">>> resumed at step {state.global_step}; "
                      f"skipping the epoch0_pretrain evals")
                return
            # Baseline before any post-training. When step-based eval is on,
            # also measure recall so we get a clean before/after comparison.
            self._run_eval(
                "epoch0_pretrain", hf_args, model, state,
                run_qa=True,
                run_recall=(cli_args.eval_every_n_steps > 0 and not cli_args.disable_recall),
                run_episodic=(cli_args.episodic_every_n_epochs > 0
                              and not cli_args.episodic_skip_pretrain),
                # Free-form generation belongs in the baseline whenever the run asks
                # for it at all. Post-training is the case that needs it: the whole
                # question is what UltraChat changes, and "before" has to come off
                # the same weights rather than a nearby step of the injection run.
                # --genqa_skip_pretrain drops this one call. On Dream a full
                # genqa is 2,452 questions filled one at a time into a 32-mask
                # canvas -- 22 minutes, against 16 minutes to train the 2,000
                # steps between evals. The pretrain point costs the same 22
                # minutes and measures the untrained base model, which is the
                # same weights in every run of a comparison and is already on
                # record; paying for it once per run is 22 minutes each time
                # for a number we can read off any earlier run.
                run_genqa=(not cli_args.genqa_skip_pretrain
                           and (cli_args.genqa_every_n_epochs > 0
                                or cli_args.genqa_every_n_steps > 0
                                or bool(cli_args.genqa_at_epochs.strip()))),
            )

        def on_epoch_begin(self, hf_args, state, control, **kwargs):
            epoch = int(state.epoch) if state.epoch else 0
            if epoch == 0:
                return
            if self.trainer is None:
                print(">>> [WARN] EvalCallback.trainer not set; cannot rotate dataset")
                return

            epoch_seed = cli_args.random_seed * 10000 + epoch
            print(f"\n>>> Reshuffling dataset for epoch {epoch} "
                  f"(epoch_seed={epoch_seed}"
                  f"{', dynamic re-chunk' if cli_args.dynamic_chunking else ''})")
            if cli_args.dynamic_chunking:
                # Rebuild corpus_chat with new chunk boundaries. qa_gen is left
                # alone on purpose: its targets come from a pre-generated chunk
                # query file keyed to the fixed chunking, so re-chunking it would
                # decouple each question from the text it was written against.
                epoch_corpus = make_corpus_data(epoch)
            else:
                epoch_corpus = corpus_data

            new_all = list(epoch_corpus) + instruct_only_data + instruct_mix
            random.seed(epoch_seed)
            random.shuffle(new_all)

            new_dataset = Dataset.from_list(new_all)
            self.trainer.train_dataset = new_dataset
            self.trainer._train_dataloader = None
            print(f"  Dataset: {len(epoch_corpus)} corpus_chat + "
                  f"{len(instruct_only_data)} instruct_only + "
                  f"{len(instruct_mix)} instruct = {len(new_all)}")

        def on_step_end(self, hf_args, state, control, model=None, **kwargs):
            n = cli_args.eval_every_n_steps
            if not n or n <= 0:
                return
            step = int(state.global_step)
            if step > 0 and step % n == 0:
                m = model if model is not None else (
                    self.trainer.model if self.trainer else None)
                # on_epoch_end is skipped in step mode, so convert the cadence.
                n_ep = cli_args.episodic_every_n_epochs
                ep_steps = n_ep * steps_per_epoch if n_ep > 0 else 0
                ng = cli_args.genqa_every_n_steps
                self._run_eval(
                    f"step{step}", hf_args, m, state,
                    run_qa=True, run_recall=(not cli_args.disable_recall),
                    run_episodic=(ep_steps > 0 and step % ep_steps < n),
                    run_genqa=(ng > 0 and step % ng < n),
                )

        def on_train_end(self, hf_args, state, control, model=None, **kwargs):
            n = cli_args.eval_every_n_steps
            if not n or n <= 0:
                return
            step = int(state.global_step)
            if step % n != 0:
                # Final step wasn't a multiple of N (e.g. data filtered below a
                # round count) — make sure we still get an end-of-run eval.
                m = model if model is not None else (
                    self.trainer.model if self.trainer else None)
                self._run_eval(
                    f"step{step}_final", hf_args, m, state,
                    run_qa=True, run_recall=(not cli_args.disable_recall),
                    run_episodic=(cli_args.episodic_every_n_epochs > 0),
                    run_genqa=(cli_args.genqa_every_n_steps > 0
                               or cli_args.genqa_every_n_epochs > 0),
                )

        def _run_genqa(self, label, model, hf_args):
            """Free-form LiteraryQA answers, scored the NarrativeQA way.

            Must be called from inside ``_run_eval`` for the same reason as
            ``_run_episodic``: it relies on that method's ``model.eval()``.

            The prompt carries the book's ``<lit_NNNNN>`` token, because that is
            where injection put the knowledge -- asking without it is asking a
            question the model was never trained on. Answers are short, so the
            mask budget is 32 rather than the episodic probe's 160.

            Scoring is shared with the AR side (eval_genqa_lit.write_results) so
            a diffusion number and a baseline number mean the same thing.
            """
            base = str(Path(__file__).resolve().parent.parent / "src_baseline")
            if base not in sys.path:
                sys.path.insert(0, base)
            import eval_genqa_lit as G

            items = G.load_items(n=cli_args.genqa_n_samples, seed=42,
                                 mc_file=ds_cfg["qa_test_file"],
                                 corpus_file=ds_cfg["corpus_file"])
            print(f"\n>>> {label} genqa ({len(items)} LiteraryQA test questions)",
                  flush=True)
            stop_tokens = {tokenizer.eos_token_id, tokenizer.pad_token_id,
                           tokenizer.convert_tokens_to_ids("<|endoftext|>"),
                           tokenizer.convert_tokens_to_ids("<|im_end|>")}
            stop_tokens.discard(None)

            n_gen = cli_args.genqa_max_new_tokens
            preds, missing, started = [], 0, time.time()
            for i, item in enumerate(items, 1):
                tid = tokenizer.convert_tokens_to_ids(f"<{item['corpus_id']}>")
                if tid is None or tid == tokenizer.unk_token_id:
                    missing += 1
                    preds.append("")
                    continue
                cue = tokenizer.apply_chat_template(
                    [{"role": "user",
                      "content": G.build_prompt(item,
                                                tokenizer.convert_ids_to_tokens(tid))}],
                    tokenize=True, add_generation_prompt=True)
                filled = diffusion_fill_from_cue(
                    model, list(cue), mask_token_id, n_gen=n_gen, n_steps=n_gen,
                    temperature=0.0, alg="ar", alg_temp=None, top_p=None, eps=1e-3)
                stop_at = next((k for k, t in enumerate(filled)
                                if t in stop_tokens), len(filled))
                # Undo the per-book substitution before scoring. The gold
                # answers were never rewritten, so a prediction still carrying
                # " <e1050_Mary>" scores zero on an answer that says "Mary".
                #
                # Gated on the mode rather than left to run everywhere. It is a
                # regex that does nothing to text without sentinels, but a model
                # that has read enough of them starts imitating the shape -- one
                # booktok answer at step 2,000 opened with "<e2066_D.>" -- and
                # "a model trained on plain text has no reason to emit that" is
                # weaker than not running the substitution at all. With this
                # gate, --entity_mode none is byte-for-byte the baseline path.
                text = tokenizer.decode(filled[:stop_at], skip_special_tokens=True)
                if cli_args.entity_mode == "booktok":
                    text = entity_mask.desentinel(text)
                preds.append(G.clean_prediction(text))
                if i % 100 == 0:
                    print(f"    [{i}/{len(items)}] {time.time() - started:.0f}s",
                          flush=True)

            path = os.path.join(hf_args.output_dir, f"genqa_{label}.jsonl")
            agg = G.write_results(path, items, preds)
            print(f">>> {label} genqa  meteor={agg['meteor']:.4f} "
                  f"rougeL={agg['rougeL']:.4f} bleu1={agg['bleu1']:.4f} "
                  f"bleu4={agg['bleu4']:.4f}  empty={agg['empty']}/{agg['n']}  "
                  f"missing_token={missing}  {time.time() - started:.0f}s", flush=True)

        def _run_episodic(self, label, model, hf_args):
            """Answer the episodic_context probe with the live weights.

            Must be called from inside ``_run_eval`` — it relies on that method's
            ``model.eval()``; running in train() would leave dropout on.

            Writes ``episodic_<label>.jsonl`` into the run dir.
            """
            # probe_lib lives with the probe; imported lazily so a broken probe
            # checkout cannot stop a training run from starting.
            probe_code = str(Path(__file__).resolve().parent.parent
                             / "episodic_probe" / "code")
            if probe_code not in sys.path:
                sys.path.insert(0, probe_code)
            import probe_lib

            items_file = cli_args.episodic_items_file or ds_cfg["episodic_items_file"]
            items = list(probe_lib.iter_items(probe_lib.ITEMS_DIR / items_file))
            print(f"\n>>> {label} episodic probe ({len(items)} items, {items_file})",
                  flush=True)

            stop_tokens = {tokenizer.eos_token_id, tokenizer.pad_token_id,
                           tokenizer.convert_tokens_to_ids("<|endoftext|>"),
                           tokenizer.convert_tokens_to_ids("<|im_end|>")}
            stop_tokens.discard(None)

            rows = []
            n_missing_token = 0
            started = time.time()
            for i, item in enumerate(items, 1):
                article_id = item["article_id"]
                gold = item.get("gold_span", "")
                token_id = tokenizer.convert_tokens_to_ids(f"<{article_id}>")
                if token_id is None or token_id == tokenizer.unk_token_id:
                    n_missing_token += 1
                    answer = ""
                else:
                    article_token = tokenizer.convert_ids_to_tokens(token_id)
                    n_gold = len(tokenizer.encode(gold, add_special_tokens=False))
                    n_gen = min(n_gold * 3 + 40, 160)
                    cue = tokenizer.apply_chat_template(
                        [{"role": "user",
                          "content": probe_lib.epctx_prompt(item, article_token)}],
                        tokenize=True, add_generation_prompt=True)
                    filled = diffusion_fill_from_cue(
                        model, list(cue), mask_token_id,
                        n_gen=n_gen, n_steps=n_gen, temperature=0.0,
                        alg="ar", alg_temp=None, top_p=None, eps=1e-3)
                    stop_at = next((k for k, t in enumerate(filled)
                                    if t in stop_tokens), len(filled))
                    answer = tokenizer.decode(filled[:stop_at],
                                              skip_special_tokens=True).strip()
                    # 与 genQA 路径同样的还原,同样按 mode 门控。漏掉这一步时
                    # btok 会答出 " <e0_Hepzibah>" 而 gold_span 写的是
                    # "Hepzibah",lexical_metrics 和后续的 judge_facts3 都判零 ——
                    # 看起来像"模型什么都没记住",实际是没解码回明文。
                    if cli_args.entity_mode == "booktok":
                        answer = entity_mask.desentinel(answer).strip()
                rows.append({
                    "qid": item["qid"],
                    "article_id": article_id,
                    "family": "episodic_context",
                    "type": item.get("type"),
                    "sentence": item["sentence"],
                    "gold": gold,
                    "answer": answer,
                    "metrics": probe_lib.lexical_metrics(answer, gold),
                })
                if i % 100 == 0:
                    print(f"    [{i}/{len(items)}] {time.time() - started:.0f}s",
                          flush=True)

            summary = {
                "rougeL": statistics.mean(r["metrics"]["rougeL"] for r in rows),
                "words": statistics.mean(len(probe_lib.tok(r["answer"])) for r in rows),
                "empty": sum(1 for r in rows if not r["answer"]) / len(rows),
            }
            meta = {
                "method": f"{args.run_name}_{label}",
                "label": label,
                "family": "episodic_context",
                "items_file": items_file,
                "prompt": "probe_lib.epctx_prompt + article token",
                "n_gen": "min(3*n_gold+40, 160)",
                "n": len(rows),
                "missing_art_token": n_missing_token,
                "secs": round(time.time() - started, 1),
                **summary,
            }
            path = os.path.join(hf_args.output_dir, f"episodic_{label}.jsonl")
            probe_lib.write_jsonl(path, rows, meta=meta)
            print(f">>> {label} episodic  rougeL={summary['rougeL']:.4f}  "
                  f"words={summary['words']:.1f}  empty={summary['empty']:.1%}  "
                  f"missing_art={n_missing_token}  {time.time() - started:.0f}s")
            print(f"  Saved to {path}", flush=True)

        def on_epoch_end(self, hf_args, state, control, model=None, **kwargs):
            if cli_args.eval_every_n_steps and cli_args.eval_every_n_steps > 0:
                # Step-based cadence already handles evaluation; the final
                # step is a multiple of N so no epoch-boundary eval is needed.
                return
            epoch = int(state.epoch)
            final_epoch = int(cli_args.epochs)
            n_qa = max(1, cli_args.qa_eval_every_n_epochs)
            n_recall = max(1, cli_args.recall_eval_every_n_epochs)
            run_qa = (epoch % n_qa == 0) or (epoch == final_epoch)
            run_recall = ((not cli_args.disable_recall)
                          and ((epoch % n_recall == 0) or (epoch == final_epoch)))
            at_ep = {int(x) for x in cli_args.episodic_at_epochs.split(",") if x.strip()}
            n_ep = cli_args.episodic_every_n_epochs
            run_episodic = ((epoch in at_ep) if at_ep
                            else (n_ep > 0
                                  and ((epoch % n_ep == 0) or (epoch == final_epoch))))
            at_g = {int(x) for x in cli_args.genqa_at_epochs.split(",") if x.strip()}
            n_g = cli_args.genqa_every_n_epochs
            run_genqa = ((epoch in at_g) if at_g
                         else (n_g > 0
                               and ((epoch % n_g == 0) or (epoch == final_epoch))))
            self._run_eval(
                f"epoch{epoch}", hf_args, model, state,
                run_qa=run_qa, run_recall=run_recall, run_episodic=run_episodic,
                run_genqa=run_genqa,
            )

    collator_cls = (entity_mask.make_entity_span_collator(
                        InstructQADiffusionMaskingCollator)
                    if args.entity_mode == "span"
                    else InstructQADiffusionMaskingCollator)
    collator = collator_cls(
        pad_token_id=tokenizer.pad_token_id,
        mask_token_id=mask_token_id,
        max_length=args.max_seq_length,
        t_min=args.t_min, t_max=args.t_max,
    )

    eval_callback = EvalCallback()
    trainer = DreamMLMTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        args=training_args,
        data_collator=collator,
        callbacks=[eval_callback],
    )
    eval_callback.trainer = trainer

    print(f"\n{'=' * 60}")
    print(f"  diffusion_train: dataset={args.dataset}  "
          f"mode={'post_train' if args.post_train else 'instructqa'}")
    print(f"  model={args.model}")
    print(f"  t=[{args.t_min}, {args.t_max}], chunk_words={args.max_chunk_words}")
    print(f"  {args.epochs} epochs, bs={args.batch_size}x{args.grad_accum}, "
          f"lr={args.lr}, seq={args.max_seq_length}, seed={args.random_seed}")
    print(f"  {len(corpus_data)} corpus_chat + {len(instruct_only_data)} instruct_only + "
          f"{len(instruct_mix)} instruct = {len(all_data)} total")
    print(f"  Recall eval: Dream cue → Qwen judge ({QWEN_JUDGE_MODEL})")
    print(f"{'=' * 60}\n")

    # Resolve resume target: explicit dir, or 'auto' → latest checkpoint in
    # this run's output_dir (re-running the same command auto-continues).
    resume_arg = args.resume_from
    if resume_arg in ("auto", "latest", "true", "True"):
        from transformers.trainer_utils import get_last_checkpoint
        last = get_last_checkpoint(args.output_dir)
        if last:
            print(f">>> Auto-resume from {last}")
            resume_arg = last
        else:
            print(">>> No checkpoint found for auto-resume; starting fresh")
            resume_arg = None

    trainer.train(resume_from_checkpoint=resume_arg)

    if args.no_save:
        print(">>> --no_save set: skipping final checkpoint save")
    else:
        final_dir = os.path.join(args.output_dir, "checkpoint_final")
        trainer.save_model(final_dir)
        tokenizer.save_pretrained(final_dir)
        print(f">>> Saved final model to {final_dir}")


if __name__ == "__main__":
    main()
