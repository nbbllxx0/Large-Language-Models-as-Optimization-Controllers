"""
pub_baseline_controller.py
--------------------------
Controllers compatible with pub_simp_solver.py (three-field formulation).

All controllers:
- Return beta actions for Heaviside projection control
- Never restart when best_is_valid=False
- Use phase-based logic rather than iteration micro-hacks
"""

from __future__ import annotations
import numpy as np


# ---------------------------------------------------------------------------
# Ablation: schedule-only (same phase logic as LLMController, no API calls)
# Use this to isolate the LLM contribution: LLMController vs ScheduleOnlyController
# ---------------------------------------------------------------------------

class ScheduleOnlyController:
    """
    Deterministic phase schedule identical to LLMController's _default_phase.
    Comparing this vs LLMController isolates whether LLM phase overrides help.
    """
    name = "schedule_only"

    def initial_action(self, params):
        return {"penal": 1.0, "beta": 1.0}

    def finalize_tail(self, params):
        return {
            "enabled": True, "tail_iters": 20, "restart_from_best": True,
            "penal": 4.5, "rmin": 1.20, "move": 0.05, "beta": 32.0,
        }

    def __call__(self, state, rho):
        it = state.iteration
        if it <= 15:
            action = {"penal": 1.5, "beta": 1.0, "move": 0.20}
        elif it <= 40:
            action = {"penal": 3.5, "beta": 4.0, "move": 0.15}
            if state.rmin > 1.35:
                action["rmin"] = max(1.35, round(state.rmin - 0.10, 2))
        elif it <= 65:
            action = {"penal": 4.5, "beta": 16.0, "move": 0.08}
            if state.rmin > 1.25:
                action["rmin"] = max(1.25, round(state.rmin - 0.10, 2))
        else:
            action = {"penal": 4.5, "beta": 32.0, "move": 0.05}
            if state.rmin > 1.20:
                action["rmin"] = max(1.20, round(state.rmin - 0.05, 2))
        if state.best_is_valid and state.compliance > 1.12 * state.best_compliance:
            action["restart"] = True
        return action or None


# ---------------------------------------------------------------------------
# Fixed (no continuation whatsoever — true baseline)
# ---------------------------------------------------------------------------

class FixedController:
    name = "fixed"

    def initial_action(self, params):
        return None

    def finalize_tail(self, params):
        return {"enabled": False}

    def __call__(self, state, rho):
        return None


class FixedTailController(FixedController):
    """
    Fixed controller with the standardized tail.  This addresses the review
    concern that the no-intervention fixed baseline is otherwise asymmetric
    with the continuation controllers.
    """
    name = "fixed_tail"

    def finalize_tail(self, params):
        return STANDARD_TAIL.copy()


# ---------------------------------------------------------------------------
# Linear penalization + Heaviside ramp (standard academic baseline)
# ---------------------------------------------------------------------------

class ThreeFieldContinuation:
    """
    Standard three-field continuation:
    - Linear penal ramp from p_start → p_end over ramp_iters
    - Heaviside beta doubles every beta_double_every iters after ramp
    - Filter radius tightened in late stage
    """
    name = "three_field_continuation"

    def __init__(
        self,
        p_start: float = 1.0,
        p_end: float = 4.5,
        ramp_iters: int = 30,
        beta_start: float = 1.0,
        beta_max: float = 32.0,
        beta_double_every: int = 10,
        rmin_late: float = 1.25,
    ):
        self.p_start = p_start
        self.p_end = p_end
        self.ramp_iters = ramp_iters
        self.beta_start = beta_start
        self.beta_max = beta_max
        self.beta_double_every = beta_double_every
        self.rmin_late = rmin_late
        self._beta = beta_start

    def initial_action(self, params):
        self._beta = self.beta_start
        return {"penal": self.p_start, "beta": self.beta_start}

    def finalize_tail(self, params):
        return {"enabled": False}

    def __call__(self, state, rho):
        action: dict = {}
        it = state.iteration

        # Penalty ramp
        if it <= self.ramp_iters:
            t = it / self.ramp_iters
            action["penal"] = round(
                self.p_start + t * (self.p_end - self.p_start), 3)
        else:
            action["penal"] = self.p_end

        # Beta doubling
        if it > self.ramp_iters:
            steps_after = it - self.ramp_iters
            if steps_after % self.beta_double_every == 0:
                self._beta = min(self._beta * 2.0, self.beta_max)
            action["beta"] = self._beta

        # Filter tightening
        if it > 45 and state.rmin > self.rmin_late and it % 10 == 0:
            action["rmin"] = round(max(self.rmin_late, state.rmin - 0.10), 2)

        # Move reduction
        if it > 50 and state.move > 0.08:
            action["move"] = max(0.08, round(state.move - 0.01, 2))

        return action or None


# ---------------------------------------------------------------------------
# Expert heuristic (improved: no blurry restarts, uses beta)
# ---------------------------------------------------------------------------

class ExpertHeuristic:
    """
    Heuristic that mimics what a careful human practitioner would do:
    - Slow early penal ramp
    - Restart only on valid best
    - Beta ramp once p is high enough
    - Late-stage filter tightening
    """
    name = "expert_heuristic"

    def __init__(self):
        self._beta = 1.0

    def initial_action(self, params):
        self._beta = 1.0
        return {"penal": 1.5, "beta": 1.0}

    def finalize_tail(self, params):
        return {"enabled": False}

    def __call__(self, state, rho):
        action: dict = {}
        it = state.iteration

        # Penalty ramp (step-wise)
        if it <= 40 and it % 10 == 0:
            action["penal"] = min(round(state.penal + 0.75, 2), 4.5)

        # Beta ramp: only start when p is high enough
        if state.penal >= 3.0 and it % 15 == 0:
            self._beta = min(self._beta * 2.0, 32.0)
            action["beta"] = self._beta

        # Filter tightening (late)
        if it > 45 and state.rmin > 1.25 and it % 12 == 0:
            action["rmin"] = round(max(1.25, state.rmin - 0.10), 2)

        # Move reduction
        if it > 50:
            action["move"] = max(0.06, round(state.move - 0.015, 3))

        # Safe restart: only if best_is_valid AND severe deterioration
        if (state.best_is_valid and
                state.compliance > 1.12 * state.best_compliance and
                state.penal >= 3.0):
            action["restart"] = True
            action["penal"] = max(state.penal, 3.5)
            action["move"] = min(state.move, 0.10)

        return action or None


class GraynessRuleController:
    """
    Deterministic closed-loop continuation controller driven by grayness and
    simple convergence indicators.  It uses the same grayness threshold as the
    LLM safety gate but makes no LLM/API calls.
    """
    name = "grayness_rule"

    def __init__(self, gate: float = 0.20, allow_restart: bool = False):
        self.gate = gate
        self.allow_restart = allow_restart
        self._beta = 1.0

    def initial_action(self, params):
        self._beta = 1.0
        return {"penal": 1.0, "beta": 1.0, "rmin": 1.50, "move": 0.20}

    def finalize_tail(self, params):
        return STANDARD_TAIL.copy()

    def __call__(self, state, rho):
        action: dict = {}
        gray = state.grayness
        it = state.iteration

        if gray > self.gate:
            # Patient penalization: increase p first, allow only mild beta.
            action["penal"] = min(4.5, round(max(state.penal, state.penal + 0.08), 3))
            action["beta"] = min(8.0, max(state.beta, 2.0 if state.penal >= 2.5 else 1.0))
            action["move"] = max(0.10, min(state.move, 0.18))
            if state.rmin > 1.35 and it % 10 == 0:
                action["rmin"] = max(1.35, round(state.rmin - 0.05, 2))
        elif gray > 0.02:
            # Topology has mostly matured; sharpen gradually and tighten.
            if it % 10 == 0 or state.stagnation_counter >= 5:
                self._beta = min(32.0, max(state.beta * 2.0, 8.0))
            action["penal"] = 4.5
            action["beta"] = max(state.beta, self._beta)
            action["move"] = max(0.06, min(state.move, 0.08))
            if state.rmin > 1.20 and it % 8 == 0:
                action["rmin"] = max(1.20, round(state.rmin - 0.05, 2))
        else:
            # Fully binary tail-ready regime.
            action.update({"penal": 4.5, "beta": 32.0, "rmin": min(state.rmin, 1.20), "move": 0.05})

        if (self.allow_restart and state.best_is_valid and state.penal >= 3.0 and
                state.compliance > 1.12 * state.best_compliance):
            action["restart"] = True
            action["move"] = min(action.get("move", state.move), 0.08)

        return action or None


class GraynessRuleRestartController(GraynessRuleController):
    name = "grayness_rule_restart"

    def __init__(self, gate: float = 0.20):
        super().__init__(gate=gate, allow_restart=True)


# ---------------------------------------------------------------------------
# MBB beam problem variant (for multi-problem experiments)
# ---------------------------------------------------------------------------

class MBBHeuristic:
    """
    Heuristic tuned for the MBB beam (simply supported, center load).
    Uses same three-field logic but different beta schedule.
    """
    name = "mbb_heuristic"

    def __init__(self):
        self._beta = 1.0

    def initial_action(self, params):
        self._beta = 1.0
        return {"penal": 1.0, "beta": 1.0}

    def finalize_tail(self, params):
        return {
            "enabled": True,
            "tail_iters": 15,
            "restart_from_best": True,
            "penal": 4.5,
            "rmin": 1.2,
            "move": 0.05,
            "beta": 32.0,
        }

    def __call__(self, state, rho):
        action: dict = {}
        it = state.iteration
        t = min(1.0, it / 50.0)
        action["penal"] = round(1.0 + t * 3.5, 3)

        if it > 20 and it % 12 == 0:
            self._beta = min(self._beta * 2.0, 32.0)
            action["beta"] = self._beta

        if it > 40 and state.rmin > 1.2:
            action["rmin"] = max(1.2, round(state.rmin - 0.08, 2))

        return action or None


# ---------------------------------------------------------------------------
# Shared tail config — ALL controllers that use a tail must reference this
# so the tail is provably identical across conditions in the paper.
# ---------------------------------------------------------------------------

STANDARD_TAIL = {
    "enabled":           True,
    "tail_iters":        40,   # increased from 20: LLM topologies are structurally
                               # richer and need more sharpening iterations to fully
                               # exploit the lower-compliance intermediate topology.
    "restart_from_best": True,
    "penal":             4.5,
    "rmin":              1.20,
    "move":              0.05,
    "beta":              32.0,
}


# ---------------------------------------------------------------------------
# TailOnlyController — null baseline for ablation
#
# Does nothing during the main loop (returns None every iteration).
# Then runs the standard tail. This answers the ablation question:
# "how much does ANY exploration contribute vs tail alone?"
#
# If tail_only ≈ llm_agent+tail, the LLM's exploration contributed nothing.
# If llm_agent+tail << tail_only, the LLM's exploration is the real contribution.
# ---------------------------------------------------------------------------

class TailOnlyController:
    """
    Zero-exploration baseline.  Main loop runs with penal=1.0 (below the
    validity gate min_penal_for_best=3.0), so no valid best_rho is ever
    recorded.  The tail then starts from uniform density — a true test of
    whether the 40-iteration sharpening tail alone can produce a competitive
    topology without any exploration-phase investment.

    Expected result: much worse than any continuation controller, confirming
    that the main-loop exploration phase genuinely matters.
    """
    name = "tail_only"

    def initial_action(self, params):
        # Set penal=1.0 so the validity gate (penal >= 3.0) is never satisfied.
        # This ensures best_is_valid=False throughout the main loop,
        # and the tail starts from uniform density rather than a warmed-up snapshot.
        return {"penal": 1.0, "beta": 1.0}

    def finalize_tail(self, params):
        return STANDARD_TAIL.copy()

    def __call__(self, state, rho):
        # Keep penal=1.0 throughout — no exploration, no penalization.
        return {"penal": 1.0, "beta": 1.0}


# ---------------------------------------------------------------------------
# Patch existing controllers to use STANDARD_TAIL where tail is enabled,
# so all comparisons use an identical tail process.
# ---------------------------------------------------------------------------

# ScheduleOnlyController already uses identical values — keep in sync via constant.
ScheduleOnlyController.finalize_tail = lambda self, params: STANDARD_TAIL.copy()

# ThreeFieldContinuation and ExpertHeuristic had tail disabled.
# For the "exploration + tail" experiment, enable it so the comparison is fair:
# every controller gets the same sharpening tail, and we compare exploration quality.
ThreeFieldContinuation.finalize_tail = lambda self, params: STANDARD_TAIL.copy()
ExpertHeuristic.finalize_tail        = lambda self, params: STANDARD_TAIL.copy()

# FixedController intentionally has no tail (it's the true no-intervention baseline).
# MBBHeuristic keeps its own tail (problem-specific, not used in main comparison).
