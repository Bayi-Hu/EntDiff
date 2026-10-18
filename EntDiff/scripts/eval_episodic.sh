#!/bin/bash
# Usage: bash eval_episodic.sh <checkpoint_dir> <tag> {quality|narrativeqa}
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

python eval_dream.py --checkpoint "$CKPT" --method_tag "$TAG" \
  --items_file "$ITEMS" --corpus_file "$CORPUS"

python judge_facts3.py \
  --answers "$ROOT/episodic_probe/answers/answers_${TAG}.jsonl" \
  --facts "$FACTS" --method "$TAG" --batch 40 --par 4
