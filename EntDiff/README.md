# EntDiff

Paths resolve relative to this directory.

```
data/              corpora, MC test sets, entity vocabularies
episodic_probe/    probe questions, reference facts, probe + grading code
src/               training
scripts/           entry points
```

## Setup

```bash
pip install -r requirements.txt
```

Grading needs the `claude` CLI on PATH; training and MC evaluation do not.
One 80GB GPU per run.

## Train

```bash
bash scripts/train_entdiff_narrativeqa.sh
bash scripts/train_entdiff_quality.sh
```

Output goes to `runs/<run_name>/`. MC accuracy is written every epoch as
`acc_epoch<N>.jsonl`. `--resume_from auto` continues from the last checkpoint.

`--entity_mode span` is entity-span masking; `none` is i.i.d. token masking.
`--entity_vocab` must be passed explicitly — its default is a different corpus
and a mismatch does not raise an error.

## Evaluate

```bash
bash scripts/eval_episodic.sh <checkpoint_dir> <tag> {quality|narrativeqa}
```

Writes `episodic_probe/answers/answers_<tag>.jsonl`, then
`episodic_probe/judge/f3_<tag>/grades.jsonl`. Re-running skips already-graded
items.
