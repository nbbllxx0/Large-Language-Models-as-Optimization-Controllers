"""
analyze_call_logs.py
--------------------
Analyzes LLM call logs produced by pub_run_comparison.py to answer:

1. How often did LLM deviate from schedule? (LLM contribution rate)
2. When LLM deviated, did compliance improve or worsen? (LLM quality)
3. Which phases were chosen most often? (phase preference)
4. Failure modes: iterations where LLM chose wrong phase (hindsight analysis)
5. Prompt key effect: does system_prompt_key matter?
6. Temperature effect: does higher T increase variance?

Usage:
    python analyze_call_logs.py results/results_pub_cantilever_2d/llm_agent_call_log_run0.json
    python analyze_call_logs.py results/results_pub_cantilever_2d/  # analyze all logs in dir
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def load_log(path: str) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def find_logs(path: str) -> list[str]:
    if os.path.isfile(path):
        return [path]
    return sorted(glob.glob(os.path.join(path, "*call_log*.json")))


# ---------------------------------------------------------------------------
# Per-log analysis
# ---------------------------------------------------------------------------

def analyze_log(log: list[dict], run_name: str = "") -> dict:
    total = len(log)
    if total == 0:
        return {}

    llm_calls = [e for e in log if e.get("mode") == "llm"]
    sched_only = [e for e in log if e.get("mode") in ("schedule_only", "cooldown",
                                                        "schedule_only_disabled",
                                                        "schedule_only_cooldown")]
    errors = [e for e in llm_calls if e.get("error") not in (None, "")]

    # LLM deviation rate: iterations where LLM chose different phase than schedule
    deviations = [e for e in llm_calls
                  if e.get("llm_parsed") and
                  e.get("sched_phase") != e["llm_parsed"].get("phase")]

    # Phase histogram
    phase_counts = {}
    for e in llm_calls:
        p = e.get("chosen_phase", "unknown")
        phase_counts[p] = phase_counts.get(p, 0) + 1

    # Sched phase histogram
    sched_counts = {}
    for e in log:
        p = e.get("sched_phase") or e.get("phase", "unknown")
        sched_counts[p] = sched_counts.get(p, 0) + 1

    # Restart requests
    restarts_requested = sum(1 for e in llm_calls
                             if e.get("llm_parsed", {}) and
                             e["llm_parsed"].get("restart", False))
    restarts_executed = sum(1 for e in llm_calls
                            if e.get("final_action", {}) and
                            e["final_action"].get("restart", False))

    return {
        "run_name":          run_name,
        "total_entries":     total,
        "llm_calls":         len(llm_calls),
        "schedule_only":     len(sched_only),
        "errors":            len(errors),
        "error_rate":        len(errors) / max(len(llm_calls), 1),
        "deviations":        len(deviations),
        "deviation_rate":    len(deviations) / max(len(llm_calls), 1),
        "phase_counts":      phase_counts,
        "sched_phase_counts": sched_counts,
        "restarts_requested": restarts_requested,
        "restarts_executed":  restarts_executed,
        "temperatures":      list({e.get("temperature", 0.0) for e in llm_calls}),
        "prompt_keys":       list({e.get("system_prompt_key", "standard") for e in llm_calls}),
        "error_messages":    [e.get("error") for e in errors[:5]],
    }


# ---------------------------------------------------------------------------
# Failure mode: hindsight analysis
# ---------------------------------------------------------------------------

def find_failure_modes(log: list[dict], compliance_hist: list[float]) -> list[dict]:
    """
    Find LLM calls where the chosen phase led to a compliance increase
    in the next 3 iterations (hindsight bad decisions).
    """
    failures = []
    llm_iters = {e["iter"]: e for e in log if e.get("mode") == "llm"}

    for it, entry in llm_iters.items():
        if it + 3 >= len(compliance_hist):
            continue
        c_now  = compliance_hist[it - 1] if it > 0 else compliance_hist[0]
        c_plus3 = compliance_hist[min(it + 2, len(compliance_hist) - 1)]
        rel_change = (c_plus3 - c_now) / max(abs(c_now), 1e-10)
        if rel_change > 0.03:   # compliance went up >3% — bad
            failures.append({
                "iter":         it,
                "chosen_phase": entry.get("chosen_phase"),
                "sched_phase":  entry.get("sched_phase"),
                "deviated":     entry.get("chosen_phase") != entry.get("sched_phase"),
                "rel_change":   round(rel_change, 4),
                "c_before":     round(c_now, 4),
                "c_after":      round(c_plus3, 4),
                "llm_note":     (entry.get("llm_parsed") or {}).get("note", ""),
            })
    return failures


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_phase_distribution(analyses: list[dict], out_path: str):
    all_phases = ["exploration", "penalization", "sharpening", "converge", "unknown"]
    colors = {"exploration": "#4ECDC4", "penalization": "#378ADD",
              "sharpening": "#1D9E75", "converge": "#D85A30", "unknown": "#888"}

    n = len(analyses)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # LLM chosen phase
    ax = axes[0]
    x = np.arange(n)
    bottoms = np.zeros(n)
    for phase in all_phases:
        vals = np.array([a["phase_counts"].get(phase, 0) for a in analyses], dtype=float)
        totals = np.array([max(a["llm_calls"], 1) for a in analyses], dtype=float)
        pcts = vals / totals * 100
        ax.bar(x, pcts, bottom=bottoms, label=phase, color=colors.get(phase, "#888"))
        bottoms += pcts
    ax.set_xticks(x)
    ax.set_xticklabels([a["run_name"][:15] for a in analyses], rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("% of LLM calls")
    ax.set_title("LLM chosen phase distribution")
    ax.legend(fontsize=7, loc="upper right")

    # Deviation rate
    ax = axes[1]
    devrates = [a["deviation_rate"] * 100 for a in analyses]
    errrates  = [a["error_rate"]     * 100 for a in analyses]
    ax.bar(x - 0.2, devrates, 0.35, label="Deviation from schedule", color="#378ADD")
    ax.bar(x + 0.2, errrates,  0.35, label="API error rate",          color="#D85A30")
    ax.set_xticks(x)
    ax.set_xticklabels([a["run_name"][:15] for a in analyses], rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("%")
    ax.set_title("LLM deviation & error rates")
    ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("path", help="JSON log file or directory containing logs")
    p.add_argument("--compliance", help="compliance_history JSON (optional, for failure analysis)")
    p.add_argument("--out", default="log_analysis", help="output directory")
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    log_paths = find_logs(args.path)
    if not log_paths:
        print(f"No log files found at {args.path}")
        sys.exit(1)

    print(f"Found {len(log_paths)} log file(s)")
    analyses = []
    all_failures = []

    for lp in log_paths:
        run_name = os.path.basename(lp).replace("_call_log", "").replace(".json", "")
        log = load_log(lp)
        a = analyze_log(log, run_name=run_name)
        analyses.append(a)

        # Failure mode analysis
        if args.compliance:
            with open(args.compliance) as f:
                c_hist = json.load(f)
            failures = find_failure_modes(log, c_hist)
            all_failures.extend(failures)
            a["failure_modes"] = len(failures)
            a["failure_mode_examples"] = failures[:3]

        print(f"\n  {run_name}")
        print(f"    LLM calls:       {a['llm_calls']}")
        print(f"    Deviation rate:  {a['deviation_rate']*100:.1f}%  "
              f"({a['deviations']} deviations from schedule)")
        print(f"    Error rate:      {a['error_rate']*100:.1f}%")
        print(f"    Phase counts:    {a['phase_counts']}")
        print(f"    Restarts req:    {a['restarts_requested']}  executed: {a['restarts_executed']}")
        if a.get("error_messages"):
            print(f"    Errors:          {a['error_messages'][:2]}")
        if failures if args.compliance else []:
            print(f"    Failure modes:   {len(failures)} (compliance rose >3% after LLM decision)")

    # Save JSON summary
    out_json = os.path.join(args.out, "call_log_analysis.json")
    with open(out_json, "w") as f:
        json.dump(analyses, f, indent=2, default=str)
    print(f"\n  Saved {out_json}")

    # Plot
    if len(analyses) > 0:
        plot_phase_distribution(analyses, os.path.join(args.out, "phase_distribution.png"))

    # Failure mode table
    if all_failures:
        print(f"\n  Failure modes ({len(all_failures)} total):")
        print(f"  {'iter':>4} {'chosen':>14} {'sched':>14} {'deviated':>9} {'Δ%':>7} {'note'}")
        for f in all_failures[:10]:
            print(f"  {f['iter']:>4} {f['chosen_phase']:>14} {f['sched_phase']:>14}"
                  f" {str(f['deviated']):>9} {f['rel_change']*100:>+6.1f}%  {f['llm_note'][:40]}")


if __name__ == "__main__":
    main()
