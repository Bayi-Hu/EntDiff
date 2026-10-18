"""Entity-aware masking for diffusion training.

Span mode masks all tokens of one name together; token mode gives each name
its own vocabulary entry. Both exist to stop the model completing a name from
its own first subword.
"""
import json
import random
import re

import torch

WORD = re.compile(r"[A-Za-z][A-Za-z'\-]*")


def load_entity_vocab(path):
    """Read the entity vocabulary JSON."""
    with open(path) as f:
        v = json.load(f)
    v["per_book"] = {k: set(w) for k, w in v["per_book"].items()}
    return v


# ─── span mode: group an entity's tokens so they share one mask draw ──

def entity_group_ids(tokenizer, text, names, offset=0):
    """Return (token_ids, group_ids); group_ids[i] is 0 outside a name, else a per-occurrence id."""
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
    """build_corpus_data_chat plus a span_ids field."""
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
    """Factory: subclass the collator, replacing the i.i.d. mask draw with a per-entity draw."""

    class EntitySpanCollator(base_cls):
        """One mask decision per entity for corpus_chat; other sample types stay i.i.d."""

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
    """Mean of the rows at idx, rescaled back to their average norm."""
    rows = W[idx].float()
    v = rows.mean(0)
    n = v.norm()
    if n > 0:
        v = v * (rows.norm(dim=1).mean() / n)
    return v.to(W.dtype)


def add_entity_tokens(model, tokenizer, forms):
    """Add forms (leading space included) to the tokenizer, seeding each row from its subwords."""
    old = [tokenizer.encode(w, add_special_tokens=False) for w in forms]
    n_added = tokenizer.add_tokens(forms, special_tokens=False)
    # mean_resizing=False: the covariance init goes through cuSOLVER and fails
    # on some nodes. Harmless here -- the loop below overwrites every new row.
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
    """Read the booktok vocabulary JSON."""
    with open(path) as f:
        return json.load(f)


def add_booktok_tokens(model, tokenizer, vocab):
    """Add per-book person sentinels and global non-person names."""
    forms = list(vocab["add"]) + list(vocab["global_all"])
    surface = vocab["surface"]
    src = [tokenizer.encode(surface.get(w, w), add_special_tokens=False)
           for w in forms]
    n_added = tokenizer.add_tokens(forms, special_tokens=False)
    # mean_resizing=False: the covariance init goes through cuSOLVER and fails
    # on some nodes. Harmless here -- the loop below overwrites every new row.
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
    """Turn ` <e1050_Mary>` back into ` Mary`; run before any scoring."""
    return SENTINEL.sub(lambda m: " " + m.group(1), text)
