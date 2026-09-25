# TTS-Route: Query-Adaptive Test-Time Scaling with a Router and Label-Free Planners

Anonymous code release for the ICLR submission. This repository contains the code needed to reproduce the main
experiment of the paper: the label-free profiling planners, the proxy verifier, the TTS-Router with its Lagrangian
decoder, and the evaluation on Best-Route-Mix, MATH and MMLU (Tables 1 and 2 and the frontier figure). Baselines,
ablations and analysis scripts are not part of this release.

## Layout

```
experiments_query/
  build_bestroute_dynamic.py     width-first planner (tts-planner-w) + label-free profiling on Best-Route-Mix
  depth_planner.py               depth-first planner (tts-planner-d: two samples, then feedback + revise rounds)
  experiment_adaptive_tts.py     generation runtime: cached, cost-accounted LLM calls (NodeRunner), prompts, answer keys
  build_bestroute_tts.py         LLM judge for Best-Route-Mix (oracle_analysis.py: judge client)
  oracle_mathmmlu.py             the same profiling on MATH / MMLU (exact-match grading)
  make_splits.py                 fixed 80/20 split of the Best-Route-Mix pool (seed 42, stratified by source)
  bestroute_rm/                  proxy verifier: judge every cached sample, build correctness pairs, train (DeBERTa-v3
                                 reward model), score (verifier.py, used online by both planners)
  pipeline_probe_router.py       scores the 1B / 8B probe answers with the verifier (router input features)
  train_router_modes.py          TTS-Router: query + probe answers + probe scores -> 11 action-success heads
  eval_modes_lagrange.py         Lagrangian decoder, lambda sweep, selection-side frontier and budget table
  task_bench.py, report_task_bench.py   MATH / MMLU: task-trained router + decoder and the per-budget table
  train_router_searchcost.py, train_router_litep6.py, train_router_probe_chs_lite.py, pretest_labels.py,
  pipeline_planner_as_needed.py, run_tts.py, memory_bank.py, experiment_tts_preference.py   shared loaders / features
  budget/                        analytic FLOPs cost model (prefill + decode) and model specs
swarm/, experiments/evaluator/   LLM client, graph runtime and dataset loaders (derived from GPTSwarm, MIT)
datasets/best_route/             Best-Route-Mix queries and references
datasets/splits/                 split manifests (Best-Route-Mix 4,604 / 1,142) and the MATH / MMLU query list
```

## Setup

```
conda create -n ttsroute python=3.12 && conda activate ttsroute
pip install -r requirements.txt
export OPENROUTER_API_KEY=...        # pool models (Llama-3.2-1B, Llama-3.1-8B, Gemma-3-27B, Qwen-2.5-72B) via OpenRouter
export TTSFLY_JUDGE_MODEL=...        # LLM judge for Best-Route-Mix (default deepseek/deepseek-v4-flash); DEEPSEEK_API_KEY if used
```
Run every script from the repository root as a module (`python -m experiments_query.<script> ...`). MATH and MMLU are
read from `datasets/MATH/{train,test}` and `datasets/MMLU/data/{dev,test}` in the layout of the original releases
(download separately); the profiled queries are the ids in `datasets/splits/mathmmlu_qids.json`. Every LLM call is
cached under `experiments_query/results/<run>/node_cache.json`, so interrupted runs resume and re-runs are free.

## Reproducing the main experiment

Costs are in units of one 1B call on the query; the per-query budget B_q is one 72B call. All stages write under
`experiments_query/results/` with the directory names the later stages expect (defaults of each script).

**1. Profiling library (Best-Route-Mix).** Label-free search of every pool model on every query, train and test side.
```
# width-first planner, global budget (all four models; profiling recipe: width<=16, batch 4, n_min 2)
python -m experiments_query.build_bestroute_dynamic --split train --n 100000 --repeats 1 --tau 0.35 --max-width 16 \
   --max-depth 3 --batch 4 --min-width 2 --stop-signal both_or --readout verifier --patience 1 \
   --verifier-path experiments_query/results/verifier_v2/models/checkpoint-best --verifier-max-length 1024 \
   --outdir experiments_query/results/oracle_v2_full          # test side: --split test --outdir .../oracle_v2_test
# width-first planner, tiered budget for 1B / 8B (deployed recipe: width<=8, n_min 4, batch 1, patience 3)
python -m experiments_query.build_bestroute_dynamic --split train --n 100000 --repeats 1 --tau 0.35 --max-width 8 \
   --max-depth 3 --min-width 4 --batch 1 --depth-frontier 4 --stop-signal both_or --readout verifier --patience 3 \
   --tiered-budget --models 1b,8b --verifier-path experiments_query/results/verifier_v2/models/checkpoint-best \
   --verifier-max-length 1024 --outdir experiments_query/results/oracle_v2_full_b1t   # test: .../oracle_v2_test_b1t
# depth-first planner, global budget
python -m experiments_query.depth_planner --split train --budget global --models 1b,8b,27b --width 2 \
   --outdir experiments_query/results/depth_planner_train     # test: --split test --n 1142 --outdir .../depth_planner_test
```
The first pass is run with the initial verifier (step 2 trained on the judged single-call answers); the profiling
used in the paper is the pass with the final verifier.

**2. Verifier.** Judge every cached sample of the train-side search, build correctness pairs, train the reward model.
```
python -m experiments_query.bestroute_rm.judge_all_samples --search-dir experiments_query/results/oracle_v2_full
python -m experiments_query.bestroute_rm.build_correctness_pairs        # -> results/verifier_v2/reward_modelling_correctness
python -m experiments_query.bestroute_rm.reward_modeling --model_name_or_path microsoft/deberta-v3-large \
   --output_dir experiments_query/results/verifier_v2/models --max_length 1024 ...   # TRL RewardTrainer arguments
```
Test-split prompts are never used for verifier training (the pair builder reads the train-side search only).

**3. Router inputs and training.**
```
python -m experiments_query.pipeline_probe_router      # verifier scores of the 1B / 8B probe answers, train and test
python -m experiments_query.train_router_modes         # 11-head router on query + probe texts + probe scores; val split 10%
python -m experiments_query.eval_modes_lagrange        # Lagrangian decode, lambda swept and selected on the val split;
                                                       #   prints the per-budget test table and writes lagrange_fine.json
```
`eval_modes_lagrange.py` prints, for every target budget, the operating point selected on the validation split and its
realised test accuracy and cost (Table 1, row "ours"); APGR is the mean normalised gain over 100 budgets between the
1B / 8B and 72B single-call anchors, with feasibility judged on the realised test cost.

**4. MATH and MMLU.**
```
python -m experiments_query.oracle_mathmmlu --domain math                                  # global-budget profiling, both planners
python -m experiments_query.oracle_mathmmlu --domain math --tiered --planners w --models 1b,8b   # tiered width search
python -m experiments_query.task_bench --domain math                                       # task-trained router + decoder, table rows
python -m experiments_query.report_task_bench
```
(the same for `--domain mmlu`). The router is fitted on the task's training split with five-fold out-of-fold
predictions for selection and evaluated on the 500 held-out test queries.

## Licence

MIT (`LICENSE`). `swarm/` and `experiments/evaluator/` derive from the MIT-licensed GPTSwarm framework.
