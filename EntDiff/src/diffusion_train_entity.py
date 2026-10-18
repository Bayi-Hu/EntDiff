"""Diffusion continued pre-training for closed-book knowledge injection.

Corpus chunks are chat-wrapped behind a per-document special token and trained
with a masked-diffusion loss. --entity_mode span masks each name as one unit.
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

DATASETS = {
    "quality_deanaph": {
        "corpus_file":         "quality_corpus_deanaph.jsonl",
        "qa_test_file":        "quality_qa.jsonl",
        "chunkq_file":         None,
        "episodic_items_file": None,
    },
    # 50 books sampled from a larger pool (stratified, seed 42), de-anaphorised.
    "literaryqa50_deanaph": {
        "corpus_file":         "lit_len50_corpus_deanaph_v5.jsonl",
        "qa_test_file":        "lit_len50_mc.jsonl",
        "chunkq_file":         None,
        "episodic_items_file": None,
    },
}


# ─── Model loading (Dream-7B + article tokens) ───────────────────────

def load_dream_model_with_tokens(model_name, corpus_file, max_chunk_words):
    """Load Dream-7B, add one special token per article, return (model, tokenizer, token_ids, mask_id)."""
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
    """Build the MC prompt body (article token, title/author, question, options)."""
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
    """Convert a 2D HF attention mask to the 4D bool mask Dream's SDPA path expects."""
    if attention_mask_2d is None:
        return None
    return attention_mask_2d[:, None, None, :].bool()


def _dream_shift_logits(logits):
    """Re-align Dream logits so position i predicts token i."""
    return torch.cat([logits[:, 0:1], logits[:, :-1]], dim=1).contiguous()


class DreamMLMTrainer(Trainer):
    """MLM-style loss: per-token CE over masked positions, summed per sequence, averaged over the batch."""

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
    """Every MC question in the file, both splits."""
    rows = [q for q in load_jsonl(DATA_DIR / qa_file)
            if q.get("options") and q.get("gold_letter")]
    print(f"MC eval: {len(rows)} questions from {qa_file} (all splits)")
    return rows


# ─── Base diffusion masking collator ─────────────────────────────────

class DiffusionMaskingCollator:
    """Base collator: mask corpus tokens at a per-sample rate t ~ U(t_min, t_max)."""

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
    """Adds the corpus_chat branch (prompt through the assistant header is never masked)."""

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
    """Adds the qa_gen branch; same masking rules as corpus_chat."""

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
    """Build chat-wrapped corpus recitation samples, one per max_chunk_words-word chunk."""
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
    """MC prompt body with no article token or context."""
    options_text = "\n".join(f"({letter}) {text}"
                             for letter, text in sorted(options.items()))
    return (
        f"Question: {question}\n\n"
        f"{options_text}\n\n"
        f"Answer with the letter of the correct option (A, B, C, or D) and nothing else."
    )


# --- instruct samples (chat-wrapped) ---

def build_instruct_gen_data_chat(tokenizer, instruct_file, seed=42,
                                 max_length=4096, limit=0):
    """Build instruction->answer samples from a general instruction set."""
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
    """Untrained Qwen2.5-Instruct judge, loaded lazily."""

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
    """MC eval: append one <mask> at the assistant position and take the argmax there."""
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
    """MC eval by generation rather than single-mask argmax."""
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
    """Fill n_gen masked positions after a cue prefix."""
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
    """Single-cue recall eval scored by an external Qwen judge."""
    model.eval()
    records = []
    n_correct = 0
    n_total = len(test_qa)
    t0 = time.time()

    def _build_cue(art_token_str, meta_prefix_str, query):
        """Return prompt token ids for the recall cue."""
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
    """Compose a run name encoding dataset and core hyperparameters."""
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
    parser.add_argument("--booktok_vocab", default=None,
                        help="booktok mode only")
    parser.add_argument("--entity_vocab", default="len50dna_entity_vocab.json",
                        help="read from DATA_DIR; must match the corpus -- a "
                             "mismatch masks the wrong spans without erroring")
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
                        help="override DATASETS[dataset]['episodic_items_file']")
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
                             "instead of sampling from qa_test_file")
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
    # chunk-query (qa_gen) was removed. --chunkq_per_chunk / --chunkq_file /
    # --chunkq_select are kept but inert so chained resume does not die on an
    # unrecognized argument: 0 passes silently, non-zero raises.
    if not args.post_train and args.chunkq_per_chunk:
        raise SystemExit(
            f"--chunkq_per_chunk {args.chunkq_per_chunk}: chunk-query generation was removed from this script; only 0 is accepted.")

    # post_train: instruct-only, no corpus.
    instruct_only_data = build_instruct_gen_data_chat(
        tokenizer, args.instruct_file, seed=args.random_seed,
        max_length=args.max_seq_length,
    ) if args.post_train else []

    def make_corpus_data(epoch):
        """Rebuild corpus_chat data for the given epoch."""
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
        # Recall on a fixed QA file, all of it, no sampling.
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
        """Runs the MC, recall, genQA and episodic evaluations on the configured cadence."""

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
            """Free-form answers scored with METEOR / ROUGE-L / BLEU."""
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
            """Answer the episodic probe with the live weights."""
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
                    # de-sentinelise before scoring, as in the genQA path
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
