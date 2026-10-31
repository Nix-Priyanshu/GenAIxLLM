"""Score Step 1 (plain LLM) vs Step 2 (RAG) against practice/questions.csv.

A question is counted correct if every `must_include` fact (or one alternative
from each `/`-separated group) appears in the reply, and no `must_not_include`
phrase appears. This is a simple substring/alternate-group check -- good enough
for a quick pass/fail signal; read failures manually too.

Usage: python evaluate.py --step 1   (baseline, no documents)
       python evaluate.py --step 2   (RAG)
"""
import argparse
import csv
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import PRACTICE_DIR
from src.step1_baseline import answer as step1_answer
from src.agent import ZentroAgent


def _facts_present(reply: str, must_include: str) -> bool:
    reply_l = reply.lower()
    for group in [g for g in must_include.split(";") if g.strip()]:
        alts = [a.strip().lower() for a in group.split("/")]
        if not any(a in reply_l for a in alts if a):
            return False
    return True


def _forbidden_present(reply: str, must_not_include: str) -> bool:
    reply_l = reply.lower()
    return any(p.strip().lower() in reply_l for p in must_not_include.split(";") if p.strip())


def run(step: int, limit: int = None, delay: float = 4.0, out_path: str = None):
    """delay: seconds to wait between LLM calls, to stay under free-tier rate
    limits (the Gemini free tier is commonly ~15 requests/minute -> 4s is safe).
    A single question's failure (even after config.py's own retries) is logged
    as an ERROR and the run continues, so one flaky call doesn't lose the rest."""
    rows = list(csv.DictReader(open(os.path.join(PRACTICE_DIR, "questions.csv"), encoding="utf-8")))
    if limit:
        rows = rows[:limit]

    agent = ZentroAgent() if step == 2 else None
    correct = 0
    errors = 0
    results = []
    out_path = out_path or os.path.join(PRACTICE_DIR, f"eval_step{step}_results.json")

    for i, row in enumerate(rows):
        q = row["question"]
        try:
            if step == 1:
                reply = step1_answer(q)
                sources = []
            else:
                reply, sources = agent.answer_once(q)
            ok = _facts_present(reply, row.get("must_include", "")) and \
                 not _forbidden_present(reply, row.get("must_not_include", ""))
            correct += ok
            status = "PASS" if ok else "FAIL"
        except Exception as e:
            reply, sources, ok, status = f"ERROR: {e}", [], False, "ERROR"
            errors += 1

        results.append({"id": row["id"], "status": status, "reply": reply, "sources": sources,
                         "gold_evidence": row.get("gold_evidence", "")})
        print(f"{row['id']}: {status}")

        # save incrementally so a crash doesn't lose progress already made
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(results, fh, ensure_ascii=False, indent=2)

        if i < len(rows) - 1:
            time.sleep(delay)

    print(f"\nStep {step}: {correct}/{len(rows)} correct"
          + (f" ({errors} error(s) after retries -- see {out_path})" if errors else ""))
    print(f"Full results saved to {out_path}")
    return results


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--step", type=int, choices=[1, 2], required=True)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--delay", type=float, default=4.0, help="seconds between calls (rate-limit pacing)")
    args = p.parse_args()
    run(args.step, args.limit, args.delay)
