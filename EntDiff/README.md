# EntDiff

Self-contained: every path resolves relative to this directory, so the folder
can be moved or archived as-is.

```
data/              corpora, MC test sets, entity vocabularies
episodic_probe/    probe questions, reference facts, probe + grading code
src/               training and preprocessing
scripts/           entry points
```

## Requirements

```bash
pip install -r requirements.txt
```

Grading additionally needs the `claude` CLI on `PATH` (`judge_facts3.py` shells
out to `claude -p --model opus`). Training and multiple-choice evaluation do
not.

Each run needs one 80GB GPU. The training scripts carry an `#SBATCH` header for
Slurm; delete it, or just `bash` the script, to run locally.

## Train

```bash
bash scripts/train_entdiff_narrativeqa.sh     # NarrativeQA, 50 books
bash scripts/train_entdiff_quality.sh         # QuALITY, 265 articles
```

Checkpoints and per-epoch multiple-choice accuracy are written under
`runs/<run_name>/`. Both scripts pass `--resume_from auto`, so re-running after
an interruption continues from the last checkpoint.

`--entity_mode span` selects entity-span masking; `--entity_mode none` gives
i.i.d. token masking on the same data.

**`--entity_vocab` must be passed explicitly.** Its default points at a
vocabulary that is not shipped here, and a mismatch does not raise an error --
it silently masks the wrong spans. Both scripts pass it.

## Evaluate

Multiple-choice accuracy is produced during training
(`runs/<run_name>/acc_epoch<N>.jsonl`); no separate command is needed.

The episodic probe runs against a saved checkpoint:

```bash
bash scripts/eval_episodic.sh <checkpoint_dir> <tag> {quality|narrativeqa}
```

Step 1 generates answers into `episodic_probe/answers/answers_<tag>.jsonl`.
Step 2 grades them against the reference facts and writes
`episodic_probe/judge/f3_<tag>/grades.jsonl`.

Grading is resumable: re-running the same command skips question ids that
already have a grade, so an interrupted run can be continued with no extra
flags.

The `claude -p --model opus` alias resolves at call time and the resolved model
id is not recorded in the output. Pin an explicit version if grades need to stay
comparable across runs made on different days.
