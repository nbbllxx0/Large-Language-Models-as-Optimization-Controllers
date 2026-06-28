# Large Language Models as Optimization Controllers

Pre-release repository for:

**Large Language Models as Optimization Controllers: Adaptive Continuation for SIMP Topology Optimization**

Authors: Shaoliang Yang, Jun Wang, Yunsheng Wang  
Department of Mechanical Engineering, Santa Clara University

This repository is the project home for the source code and reproduction
materials associated with the manuscript. The project studies online,
state-conditioned continuation control for three-field SIMP topology
optimization. The controller receives compact solver-state observations during
an optimization run and selects numeric continuation parameters for
penalization, Heaviside projection sharpness, filter radius, and OC move limit.

## Pre-Release Status

This repository has been initialized as the public release location. The code
and reproduction archive are being organized for release after journal
acceptance. The release is intended to include the SIMP solver, LLM controller,
deterministic baselines, cached-action replay support, benchmark definitions,
prompts, controller logs, and analysis scripts needed to inspect the reported
computational results.

No API credential will be stored in this repository. Live LLM runs require users
to provide their own API key through an environment variable. Cached replay and
deterministic-controller runs are designed to support inspection without live
API access.

## Planned Repository Contents

The public release is planned to include:

- Three-field SIMP solver for 2-D and 3-D compliance minimization.
- Online LLM controller with Direct Numeric Control output.
- Deterministic baselines, including fixed, fixed+tail, three-field
  continuation, expert heuristic, schedule-only, and grayness-rule controllers.
- Cached-action replay tools for reproducing archived LLM action sequences.
- Controlled-ablation, gate-sensitivity, budget-sensitivity,
  model-sensitivity, and replay/freeze analysis scripts.
- Benchmark definitions, controller logs, prompts, and merged evidence tables
  used to inspect the reported results.

## Reproducibility Notes

The live LLM experiments use temperature 0 with structured JSON output. Because
hosted model endpoints can evolve, archived controller logs are used to support
replay of the reported action sequences. Deterministic baselines and cached
replay are the recommended first checks when validating a local installation.

Repeated run indices should be interpreted as run-indexed executions rather
than independent stochastic density initializations. The solver initialization
and OC update are deterministic; any nonzero variation in live LLM runs reflects
controller/API trajectory variation.

## Citation

If you use this project, please cite the accompanying manuscript:

```bibtex
@article{yang2026llmcontrollers,
  title  = {Large Language Models as Optimization Controllers: Adaptive Continuation for SIMP Topology Optimization},
  author = {Yang, Shaoliang and Wang, Jun and Wang, Yunsheng},
  year   = {2026},
  note   = {Manuscript under review}
}
```

## License

This project is released under the BSD 3-Clause License. See `LICENSE` for
details.
