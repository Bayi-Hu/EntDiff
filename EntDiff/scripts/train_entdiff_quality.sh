#!/bin/bash
# EntDiff on QuALITY: de-anaphorised corpus + entity-span masking, Dream-7B.
# Reproduces the run reported as "EntDiff (span)" on QuALITY (MC peak .5376 @ epoch 42).
#
# Paths are resolved relative to this script, so the release directory can be
# moved anywhere. Adjust the SBATCH block for your scheduler, or drop it and run
# the python command directly on a single 80GB GPU.
#SBATCH -J entdiff_qual
#SBATCH -N 1 --ntasks-per-node=1 --cpus-per-task=8 --mem=96G --gres=gpu:1
#SBATCH -t 16:00:00
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT/src"
export TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python diffusion_train_entity.py \
  --model Dream-org/Dream-v0-Instruct-7B \
  --dataset quality_deanaph \
  --entity_mode span \
  --entity_vocab quality_dna_entity_vocab_reviewed.json \
  --epochs 80 --lr 2e-5 --batch_size 4 --grad_accum 4 \
  --max_chunk_words 64 --random_seed 42 \
  --max_seq_length 4096 --warmup_steps 500 \
  --n_instruct_mix 2000 \
  --chunkq_per_chunk 0 \
  --mc_decode argmax \
  --disable_recall \
  --qa_eval_every_n_epochs 1 \
  --save_every_n_epochs 10 --save_total_limit 1 \
  --out_root "$ROOT/runs" --run_name quality_entdiff_span_ep80_seed42 \
  --resume_from auto
