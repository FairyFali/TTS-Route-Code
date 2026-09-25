"""Side-by-side report of the task benchmarks: zero-shot (Best-Route) verifier vs in-domain verifier, MATH and MMLU.
Reads results/task_bench{,_indomain}/<domain>/summary.json.  Prints per-budget test acc@cost and APGR."""
import json, os
B = ["10", "15", "20", "25", "30", "40", "50", "75", "100"]
ROWS = [("ours", "ours"), ("ours-mv", "ours, majority-vote read-out"), ("FrugalGPT-style cascade (learned sequence)", "FrugalGPT-style cascade"), ("AutoMix-style cascade 8b->27b->72b", "AutoMix-style cascade"),
        ("BEST-Route-style (query-only, model x best-of-n)", "BEST-Route-style"), ("query-only router 8b/72b (RouteLLM-style)", "RouteLLM-style router"), ("BoN-8b", "BoN-8b"), ("Vanilla", "Vanilla")]
for dom in ("math", "mmlu"):
    for tag, name in (("", "zero-shot Best-Route verifier"), ("_indomain", "in-domain verifier, NOT cross-fitted (contaminated control)"), ("_crossfit", "in-domain verifier, cross-fitted")):
        p = f"experiments_query/results/task_bench{tag}/{dom}/summary.json"
        if not os.path.exists(p): print(f"\n[{dom} / {name}: not available yet]"); continue
        S = json.load(open(p)); T = S["table"]
        print(f"\n== {dom.upper()} / {name} ==  anchors 1b {S['anchors']['1b'][0]:.3f}@{S['anchors']['1b'][1]:.1f}, 8b {S['anchors']['8b'][0]:.3f}@{S['anchors']['8b'][1]:.1f}, 72b {S['anchors']['72b'][0]:.3f}@{S['anchors']['72b'][1]:.1f}; 27b single {S['fixed']['27b-single']['test_acc']:.3f}@{S['fixed']['27b-single']['test_cost']:.1f}")
        if "aucs" in S and "ours" in S["aucs"]: print("   ours head AUC:", S["aucs"]["ours"])
        print(f"   {'method':30s}" + "".join(f"{b:>11s}" for b in B) + "   APGR 1b / 8b")
        for key, lab in ROWS:
            if key not in T: continue
            pb = T[key]["per_budget"]; a1, a8 = T[key]["apgr"]
            print(f"   {lab:30s}" + "".join((f"{pb[b][0]:.3f}@{pb[b][1]:4.0f}" if pb.get(b) else "    --    ").rjust(11) for b in B) + f"   {a1:.3f} / {a8:.3f}")
