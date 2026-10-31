"""Trace practice/conversations.csv through the REAL agent (real LLM calls),
grouped by conversation_id in turn order, one fresh agent per conversation.

Unlike evaluate_conversations.py (which mocks the LLM to test the deterministic
verification plumbing for free), this calls the actual model, so it's the only
way to check whether the LLM's own judgment is right -- especially whether it
correctly decides to ESCALATE (and so calls create_ticket) on conversations
like PC002/PC005-PC008/PC010, which the mocked checker can't exercise.

Checks each turn's reply against must_include/must_not_include, and loosely
checks expected_tool_calls against ZentroAgent.last_tool_calls.

Usage: python evaluate_conversations_live.py
       python evaluate_conversations_live.py --conversation PC002
       python evaluate_conversations_live.py --delay 2
"""
import argparse
import csv
import json
import os
import re
import sys
import time
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import PRACTICE_DIR
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


def _tool_calls_match(expected: str, actual: list) -> bool:
    if not expected.strip():
        return True
    actual_joined = " | ".join(actual)
    for call in [c.strip() for c in expected.split(";") if c.strip()]:
        name = call.split("(")[0].strip()
        if name not in actual_joined:
            return False
        for field in ("order_id", "verification_value"):
            m = re.search(rf"{field}=(\S+?)[,)]", call)
            if m and m.group(1) not in actual_joined:
                return False
    return True


def load_conversations():
    rows = list(csv.DictReader(open(os.path.join(PRACTICE_DIR, "conversations.csv"), encoding="utf-8")))
    convs = defaultdict(list)
    for r in rows:
        convs[r["conversation_id"]].append(r)
    for cid in convs:
        convs[cid].sort(key=lambda r: int(r["turn"]))
    return convs


def run(only: str = None, delay: float = 4.0, out_path: str = None):
    convs = load_conversations()
    if only:
        convs = {only: convs[only]} if only in convs else {}
        if not convs:
            print(f"No conversation with id {only}")
            return

    out_path = out_path or os.path.join(PRACTICE_DIR, "eval_conversations_live_results.json")
    all_results = []
    total_turns = passed_turns = errors = 0

    for cid, turns in convs.items():
        print(f"\n=== {cid} ({turns[0]['scenario']}) ===")
        agent = ZentroAgent()
        for row in turns:
            total_turns += 1
            try:
                reply = agent.respond(row["customer_message"])
                facts_ok = _facts_present(reply, row.get("must_include", ""))
                forbidden_ok = not _forbidden_present(reply, row.get("must_not_include", ""))
                tools_ok = _tool_calls_match(row.get("expected_tool_calls", ""), agent.last_tool_calls)
                ok = facts_ok and forbidden_ok and tools_ok
                status = "PASS" if ok else "FAIL"
                passed_turns += ok
                reasons = []
                if not facts_ok:
                    reasons.append("missing must_include fact(s)")
                if not forbidden_ok:
                    reasons.append("contains must_not_include phrase")
                if not tools_ok:
                    reasons.append(f"tool calls mismatch (expected: {row.get('expected_tool_calls')!r}, "
                                    f"got: {agent.last_tool_calls!r})")
            except Exception as e:
                reply, status, reasons = f"ERROR: {e}", "ERROR", [str(e)]
                errors += 1

            print(f"  turn {row['turn']}: {status}" + (f" -- {'; '.join(reasons)}" if status != 'PASS' else ""))
            all_results.append({
                "conversation_id": cid, "turn": row["turn"], "status": status,
                "customer_message": row["customer_message"], "reply": reply,
                "expected_tool_calls": row.get("expected_tool_calls", ""),
                "actual_tool_calls": list(agent.last_tool_calls), "reasons": reasons,
            })
            with open(out_path, "w", encoding="utf-8") as fh:
                json.dump(all_results, fh, ensure_ascii=False, indent=2)
            time.sleep(delay)

    print(f"\nTurns: {passed_turns}/{total_turns} passed" + (f" ({errors} error(s))" if errors else ""))
    print(f"Full results saved to {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--conversation", type=str, default=None, help="run only this conversation_id")
    p.add_argument("--delay", type=float, default=4.0)
    args = p.parse_args()
    run(args.conversation, args.delay)
