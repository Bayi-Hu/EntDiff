"""
judge_facts3.py — label every gold fact hit / distorted / missing.

judge_facts.py counts hits only, which collapses two very different failures into
the same zero: a model that reports the right moment and gets it wrong, and a
model that never touches that moment at all. With 85-92% of items scoring no
hits, almost all of the between-method signal is squeezed out of the one bit
that remains.

Here each fact gets one of three labels:

    H  PRED conveys the fact
    D  PRED reports that same moment but materially wrong -- wrong participant,
       reversed outcome, contradicted detail
    M  PRED does not touch that moment at all

H + D + M = k, and three numbers fall out of it:

    recall    H / k          what judge_facts.py measured
    coverage  (H+D) / k      how much of the passage the answer aimed at
    fidelity  H / (H+D)      of what it aimed at, how much it got right

D vs M is a factual determination about one fact, not a threshold on the answer
as a whole, so it should keep the stability that made counting work: two
gradings of the same file under judge_facts.py differed by 0.0002, against 0.10
for a holistic 0/1/2 rubric.

This does not replace judge_facts.py. Both read the same frozen fact lists from
build_gold_facts.py, and H here is defined identically, so recall is comparable
across the two scripts.

Usage:
  python judge_facts3.py --answers <path|name> --method <name> [--limit 200]
"""
import argparse, json, re, subprocess
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import statistics as st
import probe_lib as pl

PROTOCOL = "v6-facts3"
MODEL = "opus"

RUBRIC = """A model was asked to recall the passage that comes immediately before
or after a given sentence in a story it had been trained on. It answered
closed-book. You are labelling, fact by fact, what it did with the passage.

Each item gives you:
  SENT   the sentence the model was cued with
  FACTS  a numbered list of what the true passage reports
  PRED   what the model wrote

Give every numbered fact exactly one label:

  "H"  PRED conveys the fact. Judge by meaning: a paraphrase counts, and
       wording, names and exact figures need not match. If PRED says "the pilot
       was thrown clear" and the fact is "Rolf is flung out of the ship", that
       is H.

  "D"  PRED reports the same moment but gets it materially wrong. The event is
       recognisably there and something essential about it is different: the
       wrong person does it, the outcome is reversed, a stated detail
       contradicts the fact. "Rolf lands in a river and is swept downstream"
       against "Rolf is flung out through the darkness onto rock" is D -- the
       moment is the same, what happens in it is not.

  "M"  PRED does not touch that moment at all. Use M when PRED is about
       something else entirely, when it only describes what the passage is
       "about" without reporting anything, when it merely repeats SENT, and
       when it loops the same lines over and over.

The distinction that matters is D against M: did the answer aim at this moment
and miss, or did it never aim at it? Aiming and missing is D even when the
content is completely wrong. Never touching it is M even when the answer is
fluent and on-topic for the story as a whole.

Two more rules:
  * Material in PRED that corresponds to no fact is ignored. Length earns
    nothing and costs nothing.
  * Label each fact on its own. A passage often gets all M or all H together,
    but mixtures are common and normal -- report what you find.

Reply with ONE line of JSON per item, in input order, and nothing else -- no
preamble, no commentary, no code fences. "labels" is a string of k characters,
one per fact in order, each H, D or M:
{"uid":"...","labels":"MHDMM"}

--- ITEMS ---
"""

OBJ_RE = re.compile(r'\{[^{}]*"uid"[^{}]*\}')
one_line = lambda s: " ".join((s or "").split())


def render(r, facts):
    numbered = "\n".join(f"  {i}. {f}" for i, f in enumerate(facts, 1))
    return (f'uid: {r["qid"]}\n'
            f'SENT: {one_line(r.get("sentence"))}\n'
            f'FACTS:\n{numbered}\n'
            f'PRED: {one_line(r.get("answer"))}\n')


def grade_batch(batch, retries=2):
    prompt = RUBRIC + "\n\n".join(render(r, f) for r, f in batch)
    want = {r["qid"]: len(f) for r, f in batch}
    best = []
    for _ in range(retries + 1):
        try:
            out = subprocess.run(["claude", "-p", "--model", MODEL,
                                  "--allowedTools", ""], input=prompt,
                                 capture_output=True, text=True, timeout=1800).stdout
        except subprocess.TimeoutExpired:
            continue
        seen, res = set(), []
        for m in OBJ_RE.finditer(out):
            try:
                d = json.loads(m.group(0))
            except Exception:
                continue
            uid, lab = d.get("uid"), d.get("labels")
            if uid not in want or uid in seen or not isinstance(lab, str):
                continue
            lab = "".join(lab.split()).upper()
            if len(lab) != want[uid] or set(lab) - set("HDM"):
                continue
            seen.add(uid)
            c = Counter(lab)
            res.append({"qid": uid, "n_facts": want[uid], "labels": lab,
                        "H": c["H"], "D": c["D"], "M": c["M"]})
        if len(res) > len(best):
            best = res
        if len(best) == len(batch):
            break
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--answers", required=True,
                    help="path to an answers file, or a bare filename under "
                         "episodic_probe/answers/")
    ap.add_argument("--method", required=True, help="names judge/f3_<method>/")
    ap.add_argument("--facts", default="gold_facts.jsonl")
    ap.add_argument("--batch", type=int, default=40)
    ap.add_argument("--par", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0,
                    help="grade only the first N items (pilot runs). The first N "
                         "in file order, so two methods limited the same way "
                         "cover the same items.")
    args = ap.parse_args()

    facts = {json.loads(l)["qid"]: json.loads(l)["facts"]
             for l in open(pl.ITEMS_DIR / args.facts) if l.strip()}
    src = Path(args.answers)
    if not src.exists():
        src = pl.ANSWERS_DIR / args.answers
    rows = [r for r in pl.iter_items(src)
            if r.get("family") == pl.FAMILY and r["qid"] in facts]
    if args.limit:
        rows = rows[:args.limit]

    out_dir = pl.JUDGE_DIR / f"f3_{args.method}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "grades.jsonl"
    done = {}
    if out_file.exists():
        for line in open(out_file):
            if line.strip():
                d = json.loads(line)
                done[d["qid"]] = d
    todo = [(r, facts[r["qid"]]) for r in rows if r["qid"] not in done]
    print(f"{args.method}: {len(rows)} items, {len(done)} graded, {len(todo)} to go")

    batches = [todo[i:i + args.batch] for i in range(0, len(todo), args.batch)]
    with open(out_file, "a") as f, ThreadPoolExecutor(max_workers=args.par) as ex:
        for n, res in enumerate(ex.map(grade_batch, batches), 1):
            for d in res:
                f.write(json.dumps(d) + "\n")
                done[d["qid"]] = d
            f.flush()
            print(f"  batch {n}/{len(batches)}  +{len(res)}  total={len(done)}", flush=True)

    got = [done[r["qid"]] for r in rows if r["qid"] in done]
    n = len(got) or 1
    H = sum(g["H"] for g in got); D = sum(g["D"] for g in got)
    K = sum(g["n_facts"] for g in got)
    recall = st.mean(g["H"] / g["n_facts"] for g in got)
    cover = st.mean((g["H"] + g["D"]) / g["n_facts"] for g in got)
    fid = H / (H + D) if H + D else 0.0
    json.dump({"protocol": PROTOCOL, "model": MODEL, "method": args.method,
               "answers": str(src), "facts_file": args.facts,
               "n": len(got), "n_items": len(rows),
               "H": H, "D": D, "M": K - H - D, "K": K,
               "recall": recall, "coverage": cover, "fidelity": fid},
              open(out_dir / "meta.json", "w"), indent=1)
    print(f"\n=== {args.method}  {len(got)}/{len(rows)} ===")
    print(f"  recall   {recall:.4f}   ({H}/{K} facts)")
    print(f"  coverage {cover:.4f}   ({H+D}/{K} aimed at)")
    print(f"  fidelity {fid:.4f}   (of what it aimed at)")
    print(f"  H={H}  D={D}  M={K-H-D}")
    z = sum(1 for g in got if g["H"] == 0 and g["D"] == 0)
    print(f"  完全没碰 GOLD 的题: {z}/{len(got)} = {z/n:.1%}")
    if len(got) < len(rows):
        print(f"  未评 {len(rows)-len(got)} 题 — 重跑同一命令即可续上")


if __name__ == "__main__":
    main()
