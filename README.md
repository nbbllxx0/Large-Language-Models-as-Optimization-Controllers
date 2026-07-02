# Large Language Models as Optimization Controllers

Pre-release research code and reproduction materials for:

**Large Language Models as Optimization Controllers: Adaptive Continuation for SIMP Topology Optimization**

Authors: Shaoliang Yang, Jun Wang, Yunsheng Wang  
Department of Mechanical Engineering, Santa Clara University

This repository implements an online, state-conditioned continuation controller for three-field SIMP topology optimization. The controller receives compact solver-state observations during an optimization run and selects numeric continuation parameters for penalization, Heaviside projection sharpness, filter radius, and OC move limit. The implementation also includes deterministic baselines, cached-action replay, controlled ablations, and analysis scripts used to inspect the reported results.

## Pre-Release Status

This repository is prepared as a pre-release companion to the manuscript. The code and data layout are intended to support inspection and reproduction of the reported computational results. The archival public release may receive minor packaging, documentation, or path-name refinements before version tagging, but the solver, controller logic, prompts, cached action logs, and analysis outputs needed to inspect the reported runs are included.

No API credential is stored in the repository.

## What Is Included

- `pub_simp_solver.py`: three-field SIMP solver for 2-D and 3-D compliance minimization.
- `pub_baseline_controller.py`: fixed, fixed+tail, three-field continuation, expert heuristic, schedule-only, grayness-rule, and replay-compatible controllers.
- `pub_llm_agent.py`: online LLM controller with Direct Numeric Control output and deterministic safety checks.
- `pub_run_comparison.py`: experiment runner for benchmark problems, ablations, replay, and sensitivity checks.
- `pub_meta_optimizer.py`: exploratory outer-configuration utility.
- `PROMPTS.md`: index of online-controller and outer-configuration prompt definitions in the executable source.
- `aggregate_revision_batches.py`: aggregation utility for controller-specific result folders.
- `analyze_revision_results.py` and `analyze_call_logs.py`: result and controller-log inspection tools.
- `results_pub_*`: archived run outputs and controller logs used by the manuscript.

Merged evidence tables used in the paper are included under `analysis/`.

## Setup

Use Python 3.10 or newer. A minimal environment can be installed with:

```bash
pip install -r requirements.txt
```

The listed dependencies are:

```bash
numpy scipy matplotlib pyamg scikit-image plotly
```

Live LLM runs use the Google Gemini API. Set the API key in the environment:

```bash
export GEMINI_API_KEY=your_key_here
```

On Windows PowerShell:

```powershell
$env:GEMINI_API_KEY = "your_key_here"
```

Cached replay and deterministic-controller runs do not require API access.

## Quick Start

Run deterministic baselines without API access:

```bash
python pub_run_comparison.py --no-llm --preset fast
```

Run one 2-D cantilever comparison with the LLM controller:

```bash
python pub_run_comparison.py --problem cantilever --mode 2d --preset long --n-runs 1
```

Run a cached-action replay from an archived call log:

```bash
python pub_run_comparison.py --no-llm --controllers llm_replay --mode 2d --problem cantilever --nelx 120 --nely 60 --max-iter 300 --min-iter 60 --n-runs 1 --replay-log results_pub_cantilever_2d_review_r0_llm_agent/llm_agent_call_log_run0.json --tag replay_example --no-plots
```

Aggregate controller-specific result folders:

```bash
python aggregate_revision_batches.py results_pub_cantilever_2d_review_r0_* --reference fixed_tail --out analysis_example
```

## Reproducibility Notes

The live LLM controller uses temperature 0 and structured JSON output. Because hosted model endpoints can evolve, archived controller logs are included to support replay of the action sequence used in the reported runs. Deterministic baselines and cached replay are the preferred first checks when validating a local installation.

Repeated run indices should be interpreted as run-indexed executions rather than independent stochastic density initializations. The solver initialization and OC update are deterministic; any nonzero variation in live LLM runs reflects controller/API trajectory variation.

## Citation

If you use this code, please cite the accompanying manuscript:

```bibtex
@article{yang2026llmcontrollers,
  title  = {Large Language Models as Optimization Controllers: Adaptive Continuation for SIMP Topology Optimization},
  author = {Yang, Shaoliang and Wang, Jun and Wang, Yunsheng},
  year   = {2026},
  note   = {Manuscript under review}
}
```

## License

This project is released under the BSD 3-Clause License. See `LICENSE` for details.
