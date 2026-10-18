#!/bin/bash
# EntDiff on NarrativeQA (the 50-book subset): MC peak .6068 @ epoch 35,
# episodic Hit Ratio peak .1459 @ epoch 50.
#
# --entity_vocab MUST be given explicitly. The flag's default points at the
# 83-book vocabulary; training the 50-book corpus against it does not error,
# it just masks the wrong spans.
#SBATCH -J entdiff_nqa
#SBATCH -N 1 --ntasks-per-node=1 --cpus-per-task=8 --mem=96G --gres=gpu:1
#SBATCH -t 16:00:00
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT/src"
export TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python diffusion_train_entity.py \
  --model Dream-org/Dream-v0-Instruct-7B \
  --dataset literaryqa50_deanaph \
  --entity_mode span \
  --entity_vocab len50dna_entity_vocab.json \
  --epochs 80 --lr 2e-5 --batch_size 4 --grad_accum 4 \
  --max_chunk_words 64 --random_seed 42 \
  --max_seq_length 4096 --warmup_steps 500 \
  --n_instruct_mix 2000 \
  --chunkq_per_chunk 0 \
  --mc_decode argmax \
  --disable_recall \
  --qa_eval_every_n_epochs 1 \
  --genqa_every_n_epochs 10 --genqa_skip_pretrain \
  --save_every_n_epochs 5 --save_total_limit 1 \
  --out_root "$ROOT/runs" --run_name narrativeqa_entdiff_span_ep80_seed42 \
  --resume_from auto
