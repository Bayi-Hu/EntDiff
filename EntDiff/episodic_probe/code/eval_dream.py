"""
eval_dream.py — answer the episodic_context probe with a Dream-7B diffusion
checkpoint. Closed-book via the per-article special token.

Prompt (built by pl.epctx_prompt with art_tok, the ONLY prompt version in use):
    Based on Article <quality_art_N> "TITLE" by AUTHOR, what is the context that
    comes immediately {before|after} the sentence: "<anchor>"?

Byte-identical to what eval_qwen.py feeds the AR models except for the article
token — that token is the asymmetry being probed.

Decoding: diffusion_fill_from_cue with alg="ar" (leftmost-first unmasking of a
contiguous mask block). NOTE the mask count IS part of the input: n_gen is fixed
before the first forward pass and the model fills essentially all of it (528/530
items hit the cap), so n_gen is a length COMMAND, not a ceiling. It is currently
derived from the gold length, which leaks how long the answer should be.

Usage:
  python eval_dream.py --checkpoint <dir> --method_tag dream_ep40
"""
import argparse, sys, time
from pathlib import Path
import probe_lib as pl
pl.cudnn_workaround()
import torch
from transformers import AutoModel, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
# diffusion_train.py 未随包发布。这两个函数在 diffusion_train_entity.py 里是
# 逐字相同的实现(已按 AST 源码段 SHA 核对),所以改从后者导入,行为不变。
from diffusion_train_entity import (diffusion_fill_from_cue,  # noqa: E402
                                    load_dream_model_with_tokens)
import entity_mask  # noqa: E402

cli_desentinel = False

FAMILY = "episodic_context"
ITEMS_FILE = "items_epctx_1000.jsonl"


def art_token(tok, aid):
    tid = tok.convert_tokens_to_ids(f"<{aid}>")
    if tid is None or tid == tok.unk_token_id:
        return None
    return tok.convert_ids_to_tokens(tid)


@torch.no_grad()
def fill_open(model, tok, cue, mask_id, n_gen):
    """Fill n_gen masks, then cut at the first EOS. The EOS trim almost never
    fires (5/530) — the model fills the whole mask block."""
    gen = diffusion_fill_from_cue(model, list(cue), mask_id, n_gen, n_gen,
                                  0.0, "ar", None, None, 1e-3)
    stops = {tok.eos_token_id, tok.pad_token_id,
             tok.convert_tokens_to_ids("<|endoftext|>"),
             tok.convert_tokens_to_ids("<|im_end|>")}
    stops.discard(None)
    cut = next((i for i, t in enumerate(gen) if t in stops), len(gen))
    return tok.decode(gen[:cut], skip_special_tokens=True).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=pl.CKPT["diffusion_c64"])
    ap.add_argument("--method_tag", default="dream_ep40")
    ap.add_argument("--out_tag", default="")
    ap.add_argument("--items_file", default=ITEMS_FILE)
    ap.add_argument("--corpus_file", default="quality_corpus_clean.jsonl",
                    help="corpus whose per-article tokens to add when the "
                         "checkpoint has none (untrained-Dream floor row only); "
                         "len83 runs need lit_len83_corpus.jsonl")
    ap.add_argument("--art_probe", default="quality_art_0",
                    help="article id used to test whether this checkpoint "
                         "already carries per-article tokens")
    ap.add_argument("--desentinel", action="store_true",
                    help="turn ` <e1050_Mary>` back into ` Mary` in the answer. "
                         "REQUIRED for the booktok arm: its vocabulary holds one "
                         "token per (book, person), so it generates sentinels, "
                         "while gold spans and gold facts are plain text. Without "
                         "this a correct answer scores zero. The genqa path does "
                         "the same thing inside diffusion_train_entity.py.")
    ap.add_argument("--n_gen", type=int, default=0,
                    help="fixed mask count for every item; 0 = min(n_gold*3+40,160) "
                         "(the gold-length-derived budget used so far)")
    args = ap.parse_args()

    global cli_desentinel
    cli_desentinel = args.desentinel
    items = list(pl.iter_items(pl.ITEMS_DIR / args.items_file))
    print(f"{FAMILY}: {len(items)} items"
          f"{'  [desentinel ON]' if cli_desentinel else ''}")
    print(f"Loading Dream: {args.checkpoint}")
    tok = AutoTokenizer.from_pretrained(args.checkpoint, trust_remote_code=True)
    has_art = art_token(tok, args.art_probe) is not None
    if has_art:
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        model = AutoModel.from_pretrained(args.checkpoint, device_map="cuda",
                                          trust_remote_code=True,
                                          torch_dtype=torch.bfloat16).eval()
        mask_id = getattr(tok, "mask_token_id", None) or tok.convert_tokens_to_ids("<mask>")
    else:
        # Untrained Dream: the article tokens are not in the vocab yet. Add them
        # with random embeddings, exactly as eval_dream_mc_base.py does, so the
        # floor row has the same prompt shape as the injected rows.
        print("  no article tokens in this checkpoint -> adding untrained ones")
        model, tok, _art_ids, mask_id = load_dream_model_with_tokens(
            args.checkpoint, args.corpus_file, 64)
        model.eval()
    print(f"mask_id={mask_id}  n_gen={'fixed ' + str(args.n_gen) if args.n_gen else 'min(n_gold*3+40,160)'}")

    rows, t0, missing = [], time.time(), 0
    for i, it in enumerate(items, 1):
        art = art_token(tok, it["article_id"])
        if art is None:
            missing += 1
            ans = ""
        else:
            gold = it.get("gold_span", "")
            n_gold = len(tok.encode(gold, add_special_tokens=False)) if gold else 0
            n_gen = args.n_gen or min(n_gold * 3 + 40, 160)
            cue = tok.apply_chat_template(
                [{"role": "user", "content": pl.epctx_prompt(it, art)}],
                tokenize=True, add_generation_prompt=True)
            ans = fill_open(model, tok, cue, mask_id, n_gen)
        raw = ans
        if cli_desentinel:
            ans = entity_mask.desentinel(ans)
        rows.append({"qid": it["qid"], "article_id": it["article_id"],
                     "answer_raw": raw,
                     "family": FAMILY, "type": it.get("type"),
                     "sentence": it.get("sentence"),
                     "gold": it.get("gold_span"), "answer": ans,
                     "metrics": pl.lexical_metrics(ans, it.get("gold_span", ""))})
        if i % 50 == 0:
            print(f"  {i}/{len(items)}  {time.time()-t0:.0f}s  missing={missing}", flush=True)

    out = f"answers_{args.method_tag}{('_' + args.out_tag) if args.out_tag else ''}.jsonl"
    pl.write_jsonl(pl.ANSWERS_DIR / out, rows,
                   meta={"method": args.method_tag, "ckpt": args.checkpoint,
                         "desentinel": cli_desentinel,
                         "n": len(rows), "family": FAMILY,
                         "n_gen": args.n_gen or "min(n_gold*3+40,160)",
                         "missing_art_token": missing})
    print(f"saved {out} ({len(rows)}, missing={missing})")


if __name__ == "__main__":
    main()
