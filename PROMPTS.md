# Prompt Materials

This file indexes the prompt material used by the LLM-controlled SIMP runs.
The executable source files are the authoritative definitions.

## Online Controller Prompt

- Source file: `pub_llm_agent.py`
- System prompt definitions: `SYSTEM_PROMPTS`
- User prompt builder: `_build_prompt(state, max_iter)`
- API call wrapper: `_call_gemini(model_name, system, prompt, temperature)`

At each LLM call, the controller sends a compact solver-state observation that
includes the current iteration, compliance, volume fraction, grayness,
stagnation/checkerboard indicators, objective trend, current continuation
parameters, allowed parameter ranges, and grayness-gate status. The response is
required to be structured JSON containing numeric continuation controls and a
short rationale. The deterministic safety layer then clips and validates the
returned values before they are applied.

## Phase-Label Prompt

- Source file: `pub_llm_agent.py`
- System prompt key: `phase`

This prompt is retained for the phase-label controller variant. It asks for a
phase label and short note rather than direct numeric controls.

## Outer-Configuration Prompt

- Source file: `pub_meta_optimizer.py`
- System prompt definition: `META_SYSTEM_PROMPT`
- User prompt builder: `build_meta_prompt(summary, current_config, run_history)`

The outer-configuration prompt summarizes completed run results and asks for
bounded updates to controller configuration parameters. The revised manuscript
treats this outer loop as exploratory, and the no-meta ablation is included to
separate this utility from the online controller behavior.

## Archived Call Logs

The archived `results_pub_*` folders include controller call logs recording the
model identifier, prompt character counts, output character counts, accepted
controller actions, fallback/error entries when present, and final solver
summaries. Cached-action replay uses these logs to reproduce the reported LLM
action sequence without a live API call.
