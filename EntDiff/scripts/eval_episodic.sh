#!/bin/bash
# Episodic probe on a trained diffusion checkpoint, then LLM grading.
#
# Usage: bash eval_episodic.sh <checkpoint_dir> <tag> {quality|narrativeqa}
#
# Step 1 generates answers with the same prompt and the same length budget
# n_gen = min(3*n_gold + 40, 160) used for every arm in the paper. For the
# diffusion model the budget is the number of mask tokens placed in the input,
# so it is a length instruction, not a ceiling; the AR arm uses the identical
# formula as max_new_tokens.
#
# Step 2 decomposes nothing new -- it grades the generated answer against the
# pre-computed atomic facts in episodic_probe/items/gold_facts*.jsonl.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CKPT="${1:?usage: $0 <checkpoint_dir> <tag> {quality|narrativeqa}}"
TAG="${2:?}"
case "${3:-quality}" in
  quality)     ITEMS=items_epctx_1000.jsonl;  CORPUS=quality_corpus_clean.jsonl;   FACTS=gold_facts.jsonl ;;
  narrativeqa) ITEMS=items_len50_epctx.jsonl; CORPUS=lit_len50_corpus_clean.jsonl; FACTS=gold_facts_len50.jsonl ;;
  *) echo "dataset must be quality or narrativeqa"; exit 1 ;;
esac
cd "$ROOT/episodic_probe/code"
export TOKENIZERS_PARALLELISM=false

echo ">>> [1/2] generating answers"
python eval_dream.py --checkpoint "$CKPT" --method_tag "$TAG" \
  --items_file "$ITEMS" --corpus_file "$CORPUS"

echo ">>> [2/2] grading with the LLM judge"
python judge_facts3.py \
  --answers "$ROOT/episodic_probe/answers/answers_${TAG}.jsonl" \
  --facts "$FACTS" --method "$TAG" --batch 40 --par 4
