"""Validate Step 3's tool-calling / verification state machine against every
real conversation in practice/conversations.csv -- WITHOUT calling the LLM.

The LLM only decides what to say; which tool gets called with which arguments
is decided by ZentroAgent.respond()'s own code (order-ID/proof detection), so
that logic can be checked deterministically and for free. This catches bugs in
the verification flow before you spend API quota testing it end to end.

Usage: python src/evaluate_conversations.py
"""
import csv
import os
import sys
from collections import defaultdict
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.agent import ZentroAgent
from src.config import PRACTICE_DIR


def _parse_expected(call_str: str):
    """'lookup_order(order_id=89782837, verification_value=4533)' -> dict"""
    call_str = call_str.strip()
    if not call_str:
        return None
    name, _, rest = call_str.partition("(")
    rest = rest.rstrip(")")
    kwargs = {}
    if rest:
        for part in rest.split(","):
            k, _, v = part.partition("=")
            kwargs[k.strip()] = v.strip()
    return {"name": name.strip(), "kwargs": kwargs}


def load_conversations():
    rows = list(csv.DictReader(open(os.path.join(PRACTICE_DIR, "conversations.csv"), encoding="utf-8")))
    grouped = defaultdict(list)
    for r in rows:
        grouped[r["conversation_id"]].append(r)
    for cid in grouped:
        grouped[cid].sort(key=lambda r: int(r["turn"]))
    return grouped


def run():
    grouped = load_conversations()
    total_calls = 0
    matched_calls = 0
    mismatches = []

    with patch("src.agent.call_llm_chat", return_value="(mocked reply)"):
        for cid, turns in grouped.items():
            agent = ZentroAgent()
            recorded = []

            def record_lookup(order_id, verification_value=None, _rec=recorded):
                _rec.append({"name": "lookup_order", "kwargs": {
                    "order_id": str(order_id), **({"verification_value": str(verification_value)}
                                                     if verification_value is not None else {})}})
                from tools.tools import lookup_order as real_lookup
                return real_lookup(order_id, verification_value)

            def record_ticket(summary, order_id=None, reason=None, intent=None, _rec=recorded):
                _rec.append({"name": "create_ticket", "kwargs": {}})
                from tools.tools import create_ticket as real_ticket
                return real_ticket(summary, order_id=order_id, reason=reason, intent=intent)

            with patch("src.agent.lookup_order", side_effect=record_lookup), \
                 patch("src.agent.create_ticket", side_effect=record_ticket):
                for turn in turns:
                    before = len(recorded)
                    agent.respond(turn["customer_message"])
                    made_this_turn = recorded[before:]
                    expected = _parse_expected(turn["expected_tool_calls"])

                    if expected is None:
                        continue  # no tool call expected this turn; don't require silence
                    total_calls += 1
                    hit = any(
                        c["name"] == expected["name"] and
                        all(c["kwargs"].get(k) == v for k, v in expected["kwargs"].items())
                        for c in made_this_turn
                    )
                    if hit:
                        matched_calls += 1
                    else:
                        mismatches.append({
                            "conversation_id": cid, "turn": turn["turn"],
                            "customer_message": turn["customer_message"],
                            "expected": turn["expected_tool_calls"],
                            "actually_made": made_this_turn,
                        })

    print(f"Tool-call match: {matched_calls}/{total_calls}\n")
    for m in mismatches:
        print(f"=== {m['conversation_id']} turn {m['turn']} ===")
        print("  customer:", m["customer_message"])
        print("  expected:", m["expected"])
        print("  got:     ", m["actually_made"])
        print()
    return mismatches


if __name__ == "__main__":
    run()
