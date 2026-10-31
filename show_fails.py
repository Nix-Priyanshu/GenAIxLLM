import json
with open("practice/eval_step2_results.json", encoding="utf-8") as f:
    results = json.load(f)
for r in results:
    if r["status"] != "PASS":
        print(f"=== {r['id']} ({r['status']}) ===")
        print("sources:", r.get("sources"))
        print("reply:", r["reply"])
        print()
