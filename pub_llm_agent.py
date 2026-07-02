"""
pub_llm_agent.py
----------------
LLM controller with DIRECT NUMERIC CONTROL architecture.
Uses Google Gemini API via raw REST (bypassing buggy SDKs).

Architecture change (this version)
------------------------------------
Previous: LLM picks a phase label -> code translates to fixed param values.
Now:      LLM outputs numeric values directly:
            {"penal": 3.8, "beta": 6.0, "rmin": 1.30, "move": 0.12, "restart": false}
          Safety rails clamp to solver-legal ranges + enforce grayness gate on beta.
          The LLM is a true online optimizer reacting to actual observations.

Other changes
-------------
* Richer prompt: grayness gate status, iters_since_best, obj_slope,
  checkerboard, compliance_vs_best%, budget_used%.
* Longer system prompt with explicit numeric guidance for each stage of a
  long run (max_iter=300+).
* Phase scaffold kept as advisory hint in prompt, not a hard gate on output.
* Fallback (API error / no key): smooth-ramp phase schedule, no degradation.
* CALL_EVERY, PENAL_RAMP_ITERS, BETA_DOUBLE_EVERY tunable by meta-optimizer.
"""

from __future__ import annotations

import json
import os
import time
import urllib.request
import urllib.error
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Tunable constants (patched by pub_meta_optimizer.py via regex)
# ---------------------------------------------------------------------------

CALL_EVERY        = 5    # iterations between LLM API calls
PENAL_RAMP_ITERS  = 12   # fallback: iters to ramp penal after phase transition
BETA_DOUBLE_EVERY = 10   # fallback: iters between beta doublings in penalization
GRAYNESS_GATE     = 0.20 # hard gate: beta capped at 4.0 while grayness > this

# ---------------------------------------------------------------------------
# Phase definitions (advisory scaffold + fallback schedule)
# ---------------------------------------------------------------------------

PHASES = {
    "exploration":  {"penal": 1.5, "beta": 1.0,  "rmin_target": 1.50, "move": 0.20},
    "penalization": {"penal": 3.5, "beta": 4.0,  "rmin_target": 1.35, "move": 0.15},
    "sharpening":   {"penal": 4.5, "beta": 16.0, "rmin_target": 1.25, "move": 0.08},
    "converge":     {"penal": 4.5, "beta": 32.0, "rmin_target": 1.20, "move": 0.05},
}
PHASE_ORDER     = {"exploration": 0, "penalization": 1, "sharpening": 2, "converge": 3}
PHASE_MIN_ITERS = {"exploration": 8, "penalization": 22, "sharpening": 16, "converge": 5}

PARAM_BOUNDS = {
    "penal": (1.0,  5.0),
    "beta":  (1.0, 64.0),
    "rmin":  (1.1,  4.0),
    "move":  (0.03, 0.40),
}

# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

SYSTEM_PROMPTS = {

"direct": """\
You are an online optimizer controlling a SIMP topology optimization solver.
Each call you output EXACT numeric values for the solver parameters.

PARAMETERS YOU CONTROL (output ALL of them every call):
  penal  [1.0-5.0]   SIMP penalization. Low=gray freely forms. High=forces solid/void.
  beta   [1.0-64.0]  Heaviside sharpness. Low=smooth. High=binary. Keep < 16.0 while grayness > 0.20.
  rmin   [1.1-4.0]   Filter radius. Decrease slowly to sharpen features. NEVER increase.
  move   [0.03-0.40] OC step limit. Decrease late-stage to stabilize.
  restart [true/false] Reload best valid solution. Only on severe compliance spike (>8% above best).

FOUR-STAGE STRATEGY — use budget_used% to pace yourself, adapt to observations:

  Stage 1 — Exploration (budget 0-8%):
    penal=1.0-2.0, beta=1.0, move=0.20. Let topology form freely.
    Exit at budget 8% regardless of grayness.

  Stage 2 — Penalization (budget 8-50%):
    Goal: drive grayness below 0.20 using BOTH penal and beta together — BUT SLOWLY.
    Ramp penal: 2.0 -> 4.5 (increase +0.2 per call, do not rush).
    Ramp beta: 1 -> 2 -> 4 (double every 15 iters MINIMUM). Do NOT exceed beta=4 in Stage 2.
    HARD RULE: beta < 8 while in Stage 2. Premature high beta locks topology into local minima.
    COMPLIANCE RISING IS NORMAL when penal or beta increases — this is physics, not a problem.
    If grayness_slope is flat for 20+ iters: raise penal +0.3 (do NOT raise beta aggressively).
    Exit when: grayness < 0.20 AND penal >= 4.0, OR budget > 50%.

  Stage 3 — Sharpening (budget 50-75%):
    penal=4.5. Now start raising beta: 4 -> 8 -> 16 (double every 15 iters).
    Reduce rmin toward 1.20 (-0.05 per call). Reduce move to 0.06-0.08.
    This is where topology becomes binary. Take your time — rushing causes local minima.

  Stage 4 — Converge (budget 75-100%):
    penal=4.5, beta=32.0, rmin=1.20, move=0.04.
    Hold steady. Let OC converge.

CRITICAL TIMING RULES:
  * Do NOT raise beta above 4.0 before budget_used > 50%. Early binarization = local minima.
  * If budget_used > 50% and beta < 4: set beta=4.0, begin Stage 3.
  * If budget_used > 75% and beta < 16: set beta=16.0 immediately.
  * If budget_used > 90% and beta < 32: set beta=32.0 immediately.
  * Compliance rising after a penal/beta increase is EXPECTED — keep advancing.
  * Only restart if compliance > 15% above best AND best_is_valid AND params unchanged 10+ iters.

Respond ONLY with JSON (no extra keys, no markdown):
{"penal": <float>, "beta": <float>, "rmin": <float>, "move": <float>, "restart": <bool>, "note": "<1 line>"}
""",

"standard": """\
You are an expert topology optimization agent controlling a SIMP solver.
The solver uses 4 phases: exploration, penalization, sharpening, converge.
YOUR GOAL is to dynamically adapt the phase schedule to minimize final compliance.
CRITICAL RULES:
1. MONOTONIC PROGRESSION: exploration -> penalization -> sharpening -> converge. NEVER backwards.
2. LIMIT EXPLORATION: advance to penalization around iteration 20-30.
3. GRAYNESS GATE: NEVER advance to sharpening while grayness > 0.22.
4. BREAK PLATEAUS: If stagnation_counter >= 5 AND grayness < 0.22, advance.
5. FINISH: reach converge by ~80 for short runs, ~200 for long runs.
Respond ONLY with JSON: {"phase": "<phase_name>", "note": "<reasoning>"}
""",

}


# ---------------------------------------------------------------------------
# Prompt builder — rich observations
# ---------------------------------------------------------------------------

def _grayness_slope(state) -> float:
    """Approximate grayness trend from last 10 params_log entries if available.
    Falls back to 0.0 (unknown) when insufficient history exists.
    A value near 0 means grayness is stuck — LLM should raise penal+beta.
    """
    # StepState doesn't carry grayness history directly, but we can approximate
    # the trend from compliance: if compliance is still falling, topology is evolving;
    # if stagnation_counter is high and grayness > 0.2, it's genuinely stuck.
    if state.stagnation_counter > 10 and state.grayness > 0.20:
        return 0.0   # stuck signal
    if state.rel_change_5 < -0.001:
        return -0.01  # improving signal (compliance falling = topology evolving)
    return 0.0


def _build_prompt(state, max_iter: int) -> str:
    hist = state.compliance_history
    iters_since_best = state.iteration - state.best_iteration
    budget_pct       = round(100.0 * state.iteration / max(max_iter, 1), 1)
    recent8          = hist[-8:] if len(hist) >= 8 else hist
    comp_vs_best     = (100.0 * (state.compliance - state.best_compliance)
                        / max(state.best_compliance, 1e-9))
    gate_status      = ("BLOCKED (grayness too high)" if state.grayness > GRAYNESS_GATE
                        else "OK (can advance beta)")

    return (
        f"iteration={state.iteration}  budget_used={budget_pct}%_of_{max_iter}\n"
        f"\n--- Compliance ---\n"
        f"compliance={state.compliance:.5f}\n"
        f"best_compliance={state.best_compliance:.5f}\n"
        f"compliance_vs_best={comp_vs_best:+.2f}%\n"
        f"rel_change_1={state.rel_change_1:.5f}  (neg=improving)\n"
        f"rel_change_5={state.rel_change_5:.5f}  (neg=improving)\n"
        f"obj_slope={state.obj_slope:.5f}\n"
        f"stagnation_counter={state.stagnation_counter}\n"
        f"iters_since_best={iters_since_best}\n"
        f"recent_compliance=[{', '.join(f'{x:.3f}' for x in recent8)}]\n"
        f"\n--- Topology ---\n"
        f"grayness={state.grayness:.4f}  beta_gate={gate_status}  (beta<16 required while grayness>0.20)\n"
        f"grayness_slope={_grayness_slope(state):.5f}  (neg=falling, 0=stuck — if stuck, raise penal+beta)\n"
        f"best_grayness={state.best_grayness:.4f}\n"
        f"checkerboard={state.checkerboard:.4f}\n"
        f"volume_fraction={state.volume_fraction:.4f}\n"
        f"best_is_valid={state.best_is_valid}\n"
        f"\n--- Current solver params ---\n"
        f"penal={state.penal:.3f}  beta={state.beta:.2f}  "
        f"rmin={state.rmin:.3f}  move={state.move:.3f}\n"
    )


# ---------------------------------------------------------------------------
# Fallback schedule (smooth ramp, used when API unavailable)
# ---------------------------------------------------------------------------

def _default_phase_long(state) -> str:
    it = state.iteration
    if it <= 20:    return "exploration"
    elif it <= 80:  return "penalization"
    elif it <= 180: return "sharpening"
    else:           return "converge"


def _fallback_action(phase: str, state, entry_iter: int, entry_penal: float) -> dict:
    cfg = PHASES.get(phase, PHASES["exploration"])
    iters_in = max(0, state.iteration - entry_iter)

    t = min(1.0, iters_in / PENAL_RAMP_ITERS)
    penal  = entry_penal + t * (cfg["penal"] - entry_penal)
    action = {"penal": round(penal, 4), "move": cfg["move"]}

    if phase == "penalization":
        doublings      = iters_in // BETA_DOUBLE_EVERY
        action["beta"] = min(cfg["beta"], 1.0 * (2.0 ** doublings))
    else:
        action["beta"] = cfg["beta"]

    if state.rmin > cfg["rmin_target"] + 1e-5:
        action["rmin"] = max(cfg["rmin_target"], round(state.rmin - 0.10, 2))
    if (phase != "exploration" and state.best_is_valid and
            state.compliance > 1.12 * state.best_compliance):
        action["restart"] = True
    return action


# ---------------------------------------------------------------------------
# Gemini REST call
# ---------------------------------------------------------------------------

def _call_gemini(model_name: str, system: str, prompt: str,
                 temperature: float) -> tuple[str, Optional[dict], Optional[str]]:
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        return "", None, "GEMINI_API_KEY not set"

    model_path = model_name if model_name.startswith("models/") else f"models/{model_name}"
    url = (f"https://generativelanguage.googleapis.com/v1beta/"
           f"{model_path}:generateContent?key={api_key}")

    payload = {
        "system_instruction": {"parts": [{"text": system}]},
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": 200,
            "responseMimeType": "application/json",
        },
    }
    data = json.dumps(payload).encode("utf-8")
    req  = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req) as resp:
            resp_json = json.loads(resp.read().decode("utf-8"))
        candidates = resp_json.get("candidates", [])
        if not candidates:
            return "", None, f"No candidates. Raw: {resp_json}"
        raw  = candidates[0].get("content", {}).get("parts", [{}])[0].get("text", "")
        text = raw.strip().replace("```json", "").replace("```", "").strip()
        return raw, json.loads(text), None
    except urllib.error.HTTPError as e:
        return "", None, f"HTTP {e.code}: {e.read().decode('utf-8')}"
    except Exception as exc:
        return "", None, str(exc)


# ---------------------------------------------------------------------------
# Safety enforcement on direct numeric output
# ---------------------------------------------------------------------------

def _apply_llm_output(parsed: dict, state, gate_threshold: float = GRAYNESS_GATE,
                      allow_restart: bool = True) -> dict:
    """Clamp to legal ranges + enforce grayness gate on beta."""
    action: dict = {}
    for key, (lo, hi) in PARAM_BOUNDS.items():
        if key in parsed:
            try:
                action[key] = round(float(np.clip(float(parsed[key]), lo, hi)), 4)
            except (TypeError, ValueError):
                pass

    # Grayness gate: only block sharpening-level beta (>=16) while gray.
    # Beta 4-8 in penalization is needed to DRIVE grayness down — do not cap it.
    # Only prevent the aggressive Heaviside sharpening jump (beta>=16) prematurely.
    if "beta" in action and state.grayness > gate_threshold:
        action["beta"] = min(action["beta"], 8.0)  # allow up to 8 to reduce grayness

    # Restart only when valid best exists and penalization has started
    if allow_restart and parsed.get("restart", False) and state.best_is_valid and state.penal >= 2.5:
        action["restart"] = True

    # rmin: only allow decreases
    if "rmin" in action and action["rmin"] > state.rmin + 1e-4:
        action["rmin"] = state.rmin

    return action


# ---------------------------------------------------------------------------
# Main controller
# ---------------------------------------------------------------------------

class LLMController:
    name = "llm_agent"

    def __init__(
        self,
        model: str = "gemini-3.1-flash-lite",
        call_every: int = CALL_EVERY,
        temperature: float = 0.0,
        verbose: bool = False,
        min_call_interval_s: float = 0.0,
        system_prompt_key: str = "direct",
        max_iter: int = 300,
        gate_threshold: float = GRAYNESS_GATE,
        allow_restart: bool = True,
    ):
        self.model               = model
        self.call_every          = call_every
        self.temperature         = temperature
        self.verbose             = verbose
        self.min_call_interval_s = min_call_interval_s
        self.system_prompt_key   = system_prompt_key
        self.system_prompt       = SYSTEM_PROMPTS.get(system_prompt_key,
                                                       SYSTEM_PROMPTS["direct"])
        self.max_iter            = max_iter
        self.gate_threshold      = gate_threshold
        self.allow_restart       = allow_restart
        self.call_log: list[dict] = []
        self._last_call_t         = 0.0
        self._api_disabled        = False
        self._api_configured      = False
        self._cooldown_until      = 0.0
        # Fallback schedule state
        self._fb_phase      = "exploration"
        self._fb_entry_iter = 0
        self._fb_entry_penal = PHASES["exploration"]["penal"]

        api_key = os.environ.get("GEMINI_API_KEY", "").strip()
        if not api_key:
            if verbose: print("[LLMController] No API key — fallback schedule active")
            self._api_disabled = True
        else:
            self._api_configured = True
            if verbose:
                print(f"[LLMController] model={model}  call_every={call_every}  "
                      f"mode={system_prompt_key}  max_iter={max_iter}")

    # ------------------------------------------------------------------
    def initial_action(self, params):
        self._fb_phase       = "exploration"
        self._fb_entry_iter  = 0
        self._fb_entry_penal = PHASES["exploration"]["penal"]
        if hasattr(params, "max_iter"):
            self.max_iter = params.max_iter
        return {"penal": 1.0, "beta": 1.0}

    def finalize_tail(self, params):
        # Import here to avoid circular import; STANDARD_TAIL ensures
        # the tail is provably identical across ALL controllers in the paper.
        from pub_baseline_controller import STANDARD_TAIL
        return STANDARD_TAIL.copy()

    # ------------------------------------------------------------------
    def _advance_fb(self, new_phase: str, state):
        if new_phase != self._fb_phase:
            self._fb_entry_iter  = state.iteration
            self._fb_entry_penal = getattr(
                state, "penal", PHASES[self._fb_phase]["penal"])
            self._fb_phase = new_phase

    def _fallback(self, state) -> dict:
        sched = _default_phase_long(state)
        if PHASE_ORDER.get(sched, 0) > PHASE_ORDER.get(self._fb_phase, 0):
            self._advance_fb(sched, state)
        if self._fb_phase in ("sharpening", "converge") and state.grayness > self.gate_threshold:
            self._advance_fb("penalization", state)
        action = _fallback_action(self._fb_phase, state,
                                  self._fb_entry_iter, self._fb_entry_penal)
        if not self.allow_restart:
            action.pop("restart", None)
        return action

    # ------------------------------------------------------------------
    def _merge_phase_label(self, state, parsed: Optional[dict]) -> dict:
        """Phase-label mode for ablation comparison."""
        proposed = parsed.get("phase") if parsed else None
        if proposed not in PHASES:
            return self._fallback(state)
        curr_idx = PHASE_ORDER.get(self._fb_phase, 0)
        prop_idx = PHASE_ORDER.get(proposed, 0)
        if prop_idx < curr_idx:
            proposed = self._fb_phase
        if prop_idx > curr_idx + 1:
            prop_idx  = curr_idx + 1
            proposed  = [k for k, v in PHASE_ORDER.items() if v == prop_idx][0]
        if proposed in ("sharpening", "converge") and state.grayness > self.gate_threshold:
            proposed = "penalization"
        elif proposed != self._fb_phase:
            dwell = state.iteration - self._fb_entry_iter
            if dwell < PHASE_MIN_ITERS.get(self._fb_phase, 0):
                proposed = self._fb_phase
        self._advance_fb(proposed, state)
        return _fallback_action(self._fb_phase, state,
                                self._fb_entry_iter, self._fb_entry_penal)

    # ------------------------------------------------------------------
    def __call__(self, state, rho):
        # Fallback path
        if self._api_disabled or not self._api_configured:
            action = self._fallback(state)
            self._record(iter=state.iteration, mode="fallback", action=action)
            return action or None

        # Per-iteration grayness safety: only cap sharpening-level beta.
        # Allow beta up to 8.0 — needed to drive grayness down in penalization.
        if state.grayness > self.gate_threshold and state.beta > 8.0:
            return {"beta": 8.0}

        # Skip non-call iterations
        if state.iteration % self.call_every != 0:
            return None

        # Cooldown after error
        now = time.time()
        if now < self._cooldown_until:
            action = self._fallback(state)
            self._record(iter=state.iteration, mode="cooldown", action=action)
            return action or None

        wait = self.min_call_interval_s - (now - self._last_call_t)
        if wait > 0:
            time.sleep(wait)

        prompt = _build_prompt(state, self.max_iter)
        call_start = time.time()
        raw, parsed, err = _call_gemini(
            self.model, self.system_prompt, prompt, self.temperature)
        self._last_call_t = time.time()
        latency_s = self._last_call_t - call_start

        if err:
            if self.verbose:
                print(f"[LLM @{state.iteration:3d}] error: {err}")
            self._cooldown_until = time.time() + 15.0
            action = self._fallback(state)
            self._record(iter=state.iteration, mode="error",
                         error=err, action=action,
                         latency_s=round(latency_s, 4),
                         input_chars=len(prompt), output_chars=0)
            return action or None

        # Apply output
        if self.system_prompt_key == "direct":
            action = _apply_llm_output(parsed or {}, state,
                                       gate_threshold=self.gate_threshold,
                                       allow_restart=self.allow_restart)
        else:
            action = self._merge_phase_label(state, parsed)
            if not self.allow_restart:
                action.pop("restart", None)

        self._record(
            iter=state.iteration, mode="llm",
            llm_parsed=parsed, final_action=action,
            temperature=self.temperature,
            system_prompt_key=self.system_prompt_key,
            latency_s=round(latency_s, 4),
            input_chars=len(prompt), output_chars=len(raw or ""),
        )
        if self.verbose and parsed:
            note = (parsed.get("note") or "")[:60]
            if self.system_prompt_key == "direct":
                print(f"[LLM @{state.iteration:3d}]  "
                      f"p={action.get('penal', state.penal):.2f}  "
                      f"β={action.get('beta',  state.beta ):.1f}  "
                      f"r={action.get('rmin',  state.rmin ):.2f}  "
                      f"m={action.get('move',  state.move ):.3f}  | {note}")
            else:
                print(f"[LLM @{state.iteration:3d}] "
                      f"phase={parsed.get('phase','?')}  {note}")
        return action or None

    def _record(self, **kw):
        self.call_log.append(kw)


class ReplayLLMController:
    """
    No-API replay of cached LLM final actions.

    This is used for revision reproducibility and one-parameter contribution
    checks.  The controller applies the logged, safety-rail-filtered
    ``final_action`` at the original LLM call iterations.  If ``freeze_params``
    is provided, those action keys are omitted while the standardized tail is
    kept unchanged.
    """
    name = "llm_replay"

    def __init__(
        self,
        log_path: str,
        freeze_params: Optional[list[str]] = None,
        gate_threshold: float = GRAYNESS_GATE,
    ):
        self.log_path = log_path
        self.freeze_params = set(freeze_params or [])
        self.gate_threshold = gate_threshold
        self.call_log: list[dict] = []
        with open(log_path, "r", encoding="utf-8") as f:
            entries = json.load(f)
        self.actions_by_iter: dict[int, dict] = {}
        for entry in entries:
            if "final_action" not in entry:
                continue
            self.actions_by_iter[int(entry["iter"])] = dict(entry["final_action"])
        if self.freeze_params:
            suffix = "_".join(sorted(self.freeze_params))
            self.name = f"llm_replay_freeze_{suffix}"

    def initial_action(self, params):
        action = {"penal": 1.0, "beta": 1.0}
        for key in self.freeze_params:
            if key == "penal":
                action["penal"] = 1.0
            elif key == "beta":
                action["beta"] = 1.0
            elif key == "rmin":
                action["rmin"] = 1.50
            elif key == "move":
                action["move"] = 0.20
        return action

    def finalize_tail(self, params):
        from pub_baseline_controller import STANDARD_TAIL
        return STANDARD_TAIL.copy()

    def __call__(self, state, rho):
        if "beta" not in self.freeze_params and state.grayness > self.gate_threshold and state.beta > 8.0:
            action = {"beta": 8.0}
            self._record(iter=state.iteration, mode="replay_gate", final_action=action)
            return action

        logged = self.actions_by_iter.get(int(state.iteration))
        if logged is None:
            return None
        action = {
            key: value for key, value in logged.items()
            if key not in self.freeze_params
        }
        if "restart" in self.freeze_params:
            action.pop("restart", None)
        self._record(
            iter=state.iteration,
            mode="replay",
            source_log=os.path.basename(self.log_path),
            frozen_params=sorted(self.freeze_params),
            logged_action=logged,
            final_action=action,
        )
        return action or None

    def _record(self, **kw):
        self.call_log.append(kw)


class PromptAblationController(LLMController):
    def __init__(self, system_prompt_key: str = "standard", **kwargs):
        super().__init__(system_prompt_key=system_prompt_key, **kwargs)
        self.name = f"llm_{system_prompt_key}"


class TemperatureAblationController(LLMController):
    def __init__(self, temperature: float = 0.3, **kwargs):
        super().__init__(temperature=temperature, **kwargs)
        self.name = f"llm_T{temperature:.1f}".replace(".", "p")


class LLMNoRestartController(LLMController):
    name = "llm_agent_no_restart"

    def __init__(self, **kwargs):
        kwargs["allow_restart"] = False
        super().__init__(**kwargs)


# ---------------------------------------------------------------------------
# No-meta ablation: LLM agent with un-tuned default hyperparameters.
# These are the naive starting values a practitioner would choose before
# running any outer configuration pass. Comparing this against the configured
# LLM agent isolates the contribution of the outer configuration loop.
# ---------------------------------------------------------------------------

# Pre-meta defaults: conservative starting points before any tuning.
# Grayness gate wider (0.30) = less aggressive beta gating.
# Call frequency lower (every 8 iters) = fewer LLM calls.
# Beta doubling slower (every 15 iters) = more conservative fallback.
_PRE_META_GRAYNESS_GATE     = 0.30
_PRE_META_CALL_EVERY        = 8
_PRE_META_BETA_DOUBLE_EVERY = 15
_PRE_META_PENAL_RAMP_ITERS  = 15


class NoMetaLLMController(LLMController):
    """
    LLM agent with pre-configuration-loop hyperparameters.
    Same LLM, same system prompt, same safety rails, but the grayness gate
    threshold, call frequency, and fallback schedule use un-tuned defaults.

    This isolates the outer-configuration contribution: if performance
    degrades substantially vs. the configured LLM agent, the outer pass is
    important; if it stays close, the real-time LLM decisions dominate.
    """
    name = "llm_agent_no_meta"

    def __init__(self, **kwargs):
        # Override call_every with pre-meta default
        kwargs.setdefault("call_every", _PRE_META_CALL_EVERY)
        super().__init__(**kwargs)

    def __call__(self, state, rho):
        # Override grayness gate enforcement with pre-meta threshold
        if state.grayness > _PRE_META_GRAYNESS_GATE and state.beta > 8.0:
            return {"beta": 8.0}

        if state.iteration % self.call_every != 0:
            return None

        # The rest is identical to LLMController.__call__ but with
        # the pre-meta grayness gate applied in _apply_llm_output_no_meta
        now = time.time()
        if now < self._cooldown_until:
            action = self._fallback_no_meta(state)
            self._record(iter=state.iteration, mode="cooldown", action=action)
            return action or None

        wait = self.min_call_interval_s - (now - self._last_call_t)
        if wait > 0:
            time.sleep(wait)

        prompt = _build_prompt(state, self.max_iter)
        call_start = time.time()
        raw, parsed, err = _call_gemini(
            self.model, self.system_prompt, prompt, self.temperature)
        self._last_call_t = time.time()
        latency_s = self._last_call_t - call_start

        if err:
            if self.verbose:
                print(f"[LLM-noMeta @{state.iteration:3d}] error: {err}")
            self._cooldown_until = time.time() + 15.0
            action = self._fallback_no_meta(state)
            self._record(iter=state.iteration, mode="error",
                         error=err, action=action,
                         latency_s=round(latency_s, 4),
                         input_chars=len(prompt), output_chars=0)
            return action or None

        action = _apply_llm_output_no_meta(parsed or {}, state)

        self._record(
            iter=state.iteration, mode="llm",
            llm_parsed=parsed, final_action=action,
            temperature=self.temperature,
            system_prompt_key=self.system_prompt_key,
            latency_s=round(latency_s, 4),
            input_chars=len(prompt), output_chars=len(raw or ""),
        )
        if self.verbose and parsed:
            note = (parsed.get("note") or "")[:60]
            print(f"[LLM-noMeta @{state.iteration:3d}]  "
                  f"p={action.get('penal', state.penal):.2f}  "
                  f"β={action.get('beta',  state.beta ):.1f}  "
                  f"r={action.get('rmin',  state.rmin ):.2f}  "
                  f"m={action.get('move',  state.move ):.3f}  | {note}")
        return action or None

    def _fallback_no_meta(self, state) -> dict:
        """Fallback with pre-meta beta doubling rate."""
        sched = _default_phase_long(state)
        if PHASE_ORDER.get(sched, 0) > PHASE_ORDER.get(self._fb_phase, 0):
            self._advance_fb(sched, state)
        if self._fb_phase in ("sharpening", "converge") and state.grayness > _PRE_META_GRAYNESS_GATE:
            self._advance_fb("penalization", state)

        cfg = PHASES.get(self._fb_phase, PHASES["exploration"])
        iters_in = max(0, state.iteration - self._fb_entry_iter)
        t = min(1.0, iters_in / _PRE_META_PENAL_RAMP_ITERS)
        penal = self._fb_entry_penal + t * (cfg["penal"] - self._fb_entry_penal)
        action = {"penal": round(penal, 4), "move": cfg["move"]}
        if self._fb_phase == "penalization":
            doublings = iters_in // _PRE_META_BETA_DOUBLE_EVERY
            action["beta"] = min(cfg["beta"], 1.0 * (2.0 ** doublings))
        else:
            action["beta"] = cfg["beta"]
        if state.rmin > cfg["rmin_target"] + 1e-5:
            action["rmin"] = max(cfg["rmin_target"], round(state.rmin - 0.10, 2))
        if (self._fb_phase != "exploration" and state.best_is_valid and
                state.compliance > 1.12 * state.best_compliance):
            action["restart"] = True
        return action


def _apply_llm_output_no_meta(parsed: dict, state) -> dict:
    """Same as _apply_llm_output but with pre-meta grayness gate."""
    action: dict = {}
    for key, (lo, hi) in PARAM_BOUNDS.items():
        if key in parsed:
            try:
                action[key] = round(float(np.clip(float(parsed[key]), lo, hi)), 4)
            except (TypeError, ValueError):
                pass

    if "beta" in action and state.grayness > _PRE_META_GRAYNESS_GATE:
        action["beta"] = min(action["beta"], 8.0)

    if parsed.get("restart", False) and state.best_is_valid and state.penal >= 2.5:
        action["restart"] = True

    if "rmin" in action and action["rmin"] > state.rmin + 1e-4:
        action["rmin"] = state.rmin

    return action
