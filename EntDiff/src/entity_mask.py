"""Two ways to stop diffusion training from cheating on multi-token entities.

The problem both address, measured on len83 (LLM-judge containment, 2452 test
questions, Dream s78000 @64 tokens vs the Qwen2.5 AR baseline):

                     单tok    多tok   非专名    全量
    AR s42000       0.3862   0.4393  0.3380   0.3789
    Dream s78000    0.4115   0.3446  0.3484   0.3626

Dream matches AR on names that cost one token and collapses on names that cost
more; AR's ordering is the *opposite* (longer names are easier for it, being
rarer and less confusable), so this is not a difficulty effect but something
specific to the diffusion objective. McNemar on the interaction: chi2 = 8.11.

The suspected cause is sibling leakage. For "Hepzibah" = [' Hep','z','ib','ah'],
when ' Hep' is masked at rate t its three siblings survive with probability
(1-t)^3, and predicting ' Hep' from "___ z ib ah" is orthographic completion,
not recall. The gradient that should force a description->name binding is spent
on a task that does not need one. A one-token name has no siblings and cannot
leak, which is exactly where the two models are level.

Two independent fixes, one per experiment:

  span   Mask an entity's tokens as a unit -- all masked or none. The siblings
         are gone when the first piece is a target, so the only route to it is
         the context. Vocabulary, corpus and tokenizer are untouched, so the
         AR baseline stays comparable without being re-run.

  token  Give each multi-token name its own vocabulary entry, so it *is* one
         token and the leak is structurally impossible. Only names that
         currently cost >= 2 tokens are converted: a name that is already one
         token would gain nothing and would trade a pretrained embedding for a
         fresh one. New embeddings are initialised to the mean of the subwords
         they replace rather than to noise -- half the names occur fewer than
         five times, and a random vector seen five times learns nothing, while
         the subword mean starts where the model already is.

Neither is on by default; ``--entity_mode none`` reproduces the baseline path
byte for byte.
"""
import json
import random
import re

import torch

WORD = re.compile(r"[A-Za-z][A-Za-z'\-]*")


def load_entity_vocab(path):
    """Read the file build_entity_vocab.py writes."""
    with open(path) as f:
        v = json.load(f)
    v["per_book"] = {k: set(w) for k, w in v["per_book"].items()}
    return v


# ─── span mode: group an entity's tokens so they share one mask draw ──

def entity_group_ids(tokenizer, text, names, offset=0):
    """Return (token_ids, group_ids) for ``text``.

    ``group_ids[i]`` is 0 outside a name and a positive integer shared by every
    token of one name occurrence.

    Dream ships a slow (pure-Python) tokenizer, so there is no offset map to
    read spans off. Instead the text is cut into alternating non-name / name
    segments and encoded piecewise. The cut before a name is placed *before*
    its leading space, so the name segment is " Hepzibah" -- the same
    space-prefixed string byte-level BPE would see mid-sentence -- and the
    piecewise encoding is expected to equal the whole-string one.

    Expected, not assumed: the result is compared against ``encode(text)`` and,
    if they differ, that chunk falls back to no grouping. A chunk that falls
    back trains exactly as it does in the baseline, so a rare mismatch costs
    coverage and never correctness.
    """
    spans = []
    for m in WORD.finditer(text):
        if m.group(0) in names:
            s = m.start()
            if s > 0 and text[s - 1] == " ":
                s -= 1
            if not spans or s >= spans[-1][1]:
                spans.append((s, m.end()))

    whole = tokenizer.encode(text, add_special_tokens=False)
    if not spans:
        return whole, [0] * len(whole)

    ids, groups, cur, g = [], [], 0, offset
    for s, e in spans:
        if s > cur:
            piece = tokenizer.encode(text[cur:s], add_special_tokens=False)
            ids += piece
            groups += [0] * len(piece)
        g += 1
        piece = tokenizer.encode(text[s:e], add_special_tokens=False)
        ids += piece
        groups += [g] * len(piece)
        cur = e
    if cur < len(text):
        piece = tokenizer.encode(text[cur:], add_special_tokens=False)
        ids += piece
        groups += [0] * len(piece)

    if ids != whole:
        return whole, [0] * len(whole)
    return ids, groups


def build_corpus_data_chat_entity(tokenizer, article_token_ids, corpus_items,
                                  entity_vocab, meta_prefix_fn,
                                  max_chunk_words=64, max_length=4096, seed=None):
    """``build_corpus_data_chat`` plus a ``span_ids`` field.

    Kept as a separate function rather than a flag on the original so that the
    baseline code path is not touched at all; the two differ only in the two
    lines that call ``entity_group_ids`` instead of ``tokenizer.encode``.
    """
    rng = random.Random(seed) if seed is not None else None
    formatted = []
    for item in corpus_items:
        article_id = item["article_id"]
        art_token_id = article_token_ids.get(article_id)
        if art_token_id is None:
            continue
        names = entity_vocab["per_book"].get(article_id, set())

        art_token_str = tokenizer.convert_ids_to_tokens(art_token_id)
        user_msg = f"Recite the article {art_token_str}\n{meta_prefix_fn(item).rstrip()}"
        prompt_ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": user_msg}],
            tokenize=True, add_generation_prompt=True,
        )
        protect_len = len(prompt_ids)

        words = item["text"].split()
        offset = rng.randint(0, max(0, max_chunk_words - 1)) if rng is not None else 0
        word_chunks = []
        if offset > 0:
            word_chunks.append(words[:offset])
        for i in range(offset, len(words), max_chunk_words):
            word_chunks.append(words[i:i + max_chunk_words])
        if not word_chunks:
            word_chunks = [[]]

        for wc in word_chunks:
            content_ids, groups = entity_group_ids(
                tokenizer, " ".join(wc), names, offset=0)
            input_ids = (list(prompt_ids) + content_ids)[:max_length]
            span_ids = ([0] * protect_len + groups)[:max_length]
            formatted.append({
                "task": "corpus_chat",
                "input_ids": input_ids,
                "protect_len": protect_len,
                "span_ids": span_ids,
            })
    return formatted


def make_entity_span_collator(base_cls):
    """Subclass the project's collator, replacing the i.i.d. mask draw with a
    per-group draw. Built as a factory so this module never imports the
    training script (which would be circular).
    """

    class EntitySpanCollator(base_cls):
        """corpus_chat samples carrying ``span_ids`` get one mask decision per
        entity; every other sample keeps the i.i.d. behaviour, so qa_gen and
        the instruct mix are unchanged.
        """

        def __call__(self, features):
            processed, t_per_sample = [], []
            for f in features:
                task = f["task"]
                if task not in ("corpus_chat", "qa_gen"):
                    raise ValueError(f"Unknown task type: {task!r}")
                ids = list(f["input_ids"])[:self.max_length]
                n = len(ids)
                protect_len = int(f["protect_len"])
                t = random.uniform(self.t_min, self.t_max)
                input_ids, labels = list(ids), [-100] * n
                mask_start = min(protect_len, n)

                spans = f.get("span_ids")
                decided = {}
                for i in range(mask_start, n):
                    g = spans[i] if spans is not None and i < len(spans) else 0
                    if g:
                        if g not in decided:
                            decided[g] = random.random() < t
                        hit = decided[g]
                    else:
                        hit = random.random() < t
                    if hit:
                        labels[i] = ids[i]
                        input_ids[i] = self.mask_token_id
                processed.append({"input_ids": input_ids, "labels": labels})
                t_per_sample.append(float(t))

            if not processed:
                return {}
            max_len = max(len(p["input_ids"]) for p in processed)
            pi, pl, pa = [], [], []
            for p in processed:
                pad = max_len - len(p["input_ids"])
                pi.append(p["input_ids"] + [self.pad_token_id] * pad)
                pl.append(p["labels"] + [-100] * pad)
                pa.append([1] * len(p["input_ids"]) + [0] * pad)
            return {
                "input_ids": torch.tensor(pi, dtype=torch.long),
                "labels": torch.tensor(pl, dtype=torch.long),
                "attention_mask": torch.tensor(pa, dtype=torch.long),
                "t_per_sample": torch.tensor(t_per_sample, dtype=torch.float),
            }

    return EntitySpanCollator


# ─── token mode: one vocabulary entry per multi-token name ────────────

def _mean_keep_norm(W, idx):
    """Mean of the rows at ``idx``, rescaled back to their average norm.

    The plain mean is the wrong magnitude. Averaging k roughly independent
    vectors shrinks the norm by about 1/sqrt(k), and measured on Dream-7B the
    lm_head rows come out at 0.765 / 0.654 / 0.608 / 0.573 of a normal row for
    2 / 3 / 4 / 5 pieces. A short output row is a small logit, so the new token
    loses argmax to whatever it competes with -- and it loses in proportion to
    how many pieces the name had, which is exactly the population the
    experiment is about. The smoke run showed it: before a single training
    step, multi-token names appeared in 5 of 20 answers under token mode
    against 10 of 20 under span, and by step 210 it was 1 of 20.

    Rescaling keeps the direction the mean gives -- the right neighbourhood of
    the embedding space -- and restores the magnitude. For a name that was
    already one token the mean is that row, its norm is unchanged, and this is
    the identity.
    """
    rows = W[idx].float()
    v = rows.mean(0)
    n = v.norm()
    if n > 0:
        v = v * (rows.norm(dim=1).mean() / n)
    return v.to(W.dtype)


def add_entity_tokens(model, tokenizer, forms):
    """Add ``forms`` (already carrying their leading space) to the tokenizer and
    seed each new row with the mean of the subwords it replaces.

    The leading space matters: " Hepzibah" added as a special token makes
    "saw Hepzibah" one token and round-trips exactly, while a bare "Hepzibah"
    would strand the space as its own token and make the sequence *longer*.
    Possessives still work -- "Hepzibah's" becomes [' Hepzibah', "'s"].

    Added as ordinary tokens, not special ones. ``special_tokens=True`` marks a
    token as a control symbol, and every generation path here decodes with
    ``skip_special_tokens=True`` so the article token and the chat markers do
    not leak into an answer -- which means an entity added that way is deleted
    from the output the moment the model gets it right. "It was Hepzibah
    there." decodes as "It was there." The first token run showed exactly that:
    multi-token names fell from 5 of 20 answers to 1 and stayed, because the
    better the model got at emitting the atomic token, the more often the name
    was stripped before scoring. These are content, not control.
    """
    old = [tokenizer.encode(w, add_special_tokens=False) for w in forms]
    n_added = tokenizer.add_tokens(forms, special_tokens=False)
    # mean_resizing=False:transformers 新版默认会用现有词表的均值和协方差初始化新行,
    # 那要对 152k x 3584 的嵌入矩阵求协方差再做 Cholesky 检查,走 cuSOLVER。在
    # atl1-1-03-012-18-0 上它直接抛 CUSOLVER_STATUS_INTERNAL_ERROR(cusolverDnCreate
    # 建句柄就失败),同样的代码在别的节点能跑 —— 是节点相关的,不是配置问题。
    # 关掉它在这里零语义代价:下面的循环对每个新 id 都用 _mean_keep_norm 从"它所代表的
    # 那个名字的子词"重新赋值,把 resize 给的初始化整个覆盖掉。唯一不被覆盖的是
    # `not pieces`(surface 编码为空)那一支,而真实名字不会出现这种情况。
    # 顺带省掉一次对整个嵌入矩阵的协方差计算。
    model.resize_token_embeddings(len(tokenizer), mean_resizing=False)
    new_ids = [tokenizer.convert_tokens_to_ids(w) for w in forms]

    with torch.no_grad():
        emb = model.get_input_embeddings().weight
        head = model.get_output_embeddings()
        tie = head is None or head.weight.data_ptr() == emb.data_ptr()
        for nid, pieces in zip(new_ids, old):
            if not pieces:
                continue
            idx = torch.tensor(pieces, device=emb.device)
            emb[nid] = _mean_keep_norm(emb, idx)
            if not tie:
                hidx = torch.tensor(pieces, device=head.weight.device)
                head.weight[nid] = _mean_keep_norm(head.weight, hidx)

    n_pieces = sum(len(p) for p in old)
    print(f"  entity tokens: +{n_added} "
          f"(mean-init from {n_pieces:,} subword rows, "
          f"{n_pieces / max(1, len(forms)):.2f} pieces/name)")
    return n_added


# ─── booktok mode: a person token per book ───────────────────────────

def load_booktok_vocab(path):
    """Read what build_booktok_data.py wrote."""
    with open(path) as f:
        return json.load(f)


def add_booktok_tokens(model, tokenizer, vocab):
    """Add the per-book person sentinels and the global non-person names.

    A sentinel is initialised from the subwords of the *name it stands for*,
    not from the sentinel's own characters -- ` <e1050_Mary>` has to start
    where ` Mary` starts, otherwise the 28 Marys begin as 28 arbitrary vectors
    instead of 28 copies of one sensible one that training then pulls apart.
    That identical start is the point: any difference between two books' Marys
    at the end of training was learned, not initialised.

    Added as ordinary tokens, not special ones. ``special_tokens=True`` marks a
    token as a control symbol, and every generation path here decodes with
    ``skip_special_tokens=True`` so the article token and the chat markers do
    not leak into an answer -- which means an entity added that way is deleted
    from the output the moment the model gets it right. "It was Hepzibah
    there." decodes as "It was there." The first token run showed exactly that:
    multi-token names fell from 5 of 20 answers to 1 and stayed, because the
    better the model got at emitting the atomic token, the more often the name
    was stripped before scoring. These are content, not control.
    """
    forms = list(vocab["add"]) + list(vocab["global_all"])
    surface = vocab["surface"]
    src = [tokenizer.encode(surface.get(w, w), add_special_tokens=False)
           for w in forms]
    n_added = tokenizer.add_tokens(forms, special_tokens=False)
    # mean_resizing=False:transformers 新版默认会用现有词表的均值和协方差初始化新行,
    # 那要对 152k x 3584 的嵌入矩阵求协方差再做 Cholesky 检查,走 cuSOLVER。在
    # atl1-1-03-012-18-0 上它直接抛 CUSOLVER_STATUS_INTERNAL_ERROR(cusolverDnCreate
    # 建句柄就失败),同样的代码在别的节点能跑 —— 是节点相关的,不是配置问题。
    # 关掉它在这里零语义代价:下面的循环对每个新 id 都用 _mean_keep_norm 从"它所代表的
    # 那个名字的子词"重新赋值,把 resize 给的初始化整个覆盖掉。唯一不被覆盖的是
    # `not pieces`(surface 编码为空)那一支,而真实名字不会出现这种情况。
    # 顺带省掉一次对整个嵌入矩阵的协方差计算。
    model.resize_token_embeddings(len(tokenizer), mean_resizing=False)
    new_ids = [tokenizer.convert_tokens_to_ids(w) for w in forms]

    with torch.no_grad():
        emb = model.get_input_embeddings().weight
        head = model.get_output_embeddings()
        tie = head is None or head.weight.data_ptr() == emb.data_ptr()
        for nid, pieces in zip(new_ids, src):
            if not pieces:
                continue
            emb[nid] = _mean_keep_norm(emb, torch.tensor(pieces, device=emb.device))
            if not tie:
                head.weight[nid] = _mean_keep_norm(
                    head.weight, torch.tensor(pieces, device=head.weight.device))

    print(f"  booktok tokens: +{n_added} "
          f"({len(vocab['add']):,} per-book person, "
          f"{len(vocab['global_all']):,} global non-person)")
    return n_added


SENTINEL = re.compile(r"\s*<e\d+_([^>]*)>")


def desentinel(text):
    """Turn ` <e1050_Mary>` back into ` Mary`.

    Generated text has to pass through this before METEOR, ROUGE, BLEU or the
    judge see it -- the gold answers are plain names and were never rewritten,
    so a prediction still carrying sentinels scores zero on an answer that is
    in fact correct. Applied to every generation path, including the ones that
    produce nothing but a letter, where it is a no-op.

    The inner pattern is deliberately loose. Two thousand steps in, the model
    has seen several hundred thousand sentinels and has learned to imitate the
    shape, so it emits malformed ones character by character -- one step-2000
    answer opened with "<e2066_D.>", a book number that did not match the
    question and a name with a full stop in it, neither of which any real
    sentinel has. Matching only well-formed sentinels leaves that debris in the
    text to be scored. Keeping whatever sits after the underscore recovers the
    model's intent where there is one and costs nothing where there is not;
    the string "<e<digits>_" does not occur in English.
    """
    return SENTINEL.sub(lambda m: " " + m.group(1), text)
