"""
pub_meta_optimizer.py
---------------------
Optional outer configuration loop: after each comparison run, a Gemini LLM
reflects on the results and proposes bounded updates to the LLM controller's
own config (GRAYNESS_GATE, PHASE_MIN_ITERS, beta schedule) in
pub_llm_agent.py.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import subprocess
import sys
from typing import Optional

# ---------------------------------------------------------------------------
# Parameter bounds — hard safety limits the LLM cannot exceed
# ---------------------------------------------------------------------------

PARAM_BOUNDS = {
    # Gate / schedule
    "GRAYNESS_GATE":                    (0.10, 0.35),
    "PHASE_MIN_ITERS.exploration":      (2,    20),
    "PHASE_MIN_ITERS.penalization":     (4,    25),
    "PHASE_MIN_ITERS.sharpening":       (4,    25),
    "PHASE_MIN_ITERS.converge":         (0,    10),
    # Call frequency and ramp shape
    "CALL_EVERY":                       (3,    15),
    "PENAL_RAMP_ITERS":                 (4,    20),
    "BETA_DOUBLE_EVERY":                (5,    20),
    # Direct numeric control targets (for direct mode)
    "beta_sharpening_cap":              (4.0,  32.0),
}

# ---------------------------------------------------------------------------
# Read current config from pub_llm_agent.py
# ---------------------------------------------------------------------------

def read_current_config(agent_path: str) -> dict:
    with open(agent_path, encoding="utf-8") as f:
        src = f.read()

    config = {}

    m = re.search(r"^GRAYNESS_GATE\s*=\s*([\d.]+)", src, re.MULTILINE)
    if m: config["GRAYNESS_GATE"] = float(m.group(1))

    m = re.search(r"PHASE_MIN_ITERS\s*=\s*(\{[^}]+\})", src, re.DOTALL)
    if m:
        raw = re.sub(r"#[^\n]*", "", m.group(1))
        pmi = ast.literal_eval(raw.strip())
        for k, v in pmi.items():
            config[f"PHASE_MIN_ITERS.{k}"] = v

    for const in ("CALL_EVERY", "PENAL_RAMP_ITERS", "BETA_DOUBLE_EVERY"):
        m = re.search(rf"^{const}\s*=\s*(\d+)", src, re.MULTILINE)
        if m: config[const] = int(m.group(1))

    return config

# ---------------------------------------------------------------------------
# Apply a validated delta to pub_llm_agent.py
# ---------------------------------------------------------------------------

def apply_delta(agent_path: str, delta: dict, dry_run: bool = False) -> str:
    with open(agent_path, encoding="utf-8") as f:
        src = f.read()

    changes = []

    if "GRAYNESS_GATE" in delta:
        v = delta["GRAYNESS_GATE"]
        src, n = re.subn(r"^(GRAYNESS_GATE\s*=\s*)[\d.]+", f"\\g<1>{v}", src, flags=re.MULTILINE)
        if n: changes.append(f"  GRAYNESS_GATE -> {v}")

    pmi_keys = [k for k in delta if k.startswith("PHASE_MIN_ITERS.")]
    if pmi_keys:
        m = re.search(r"(PHASE_MIN_ITERS\s*=\s*)(\{[^}]+\})", src, re.DOTALL)
        if m:
            raw = re.sub(r"#[^\n]*", "", m.group(2))
            pmi = ast.literal_eval(raw.strip())
            for k in pmi_keys:
                phase = k.split(".", 1)[1]
                pmi[phase] = delta[k]
                changes.append(f"  PHASE_MIN_ITERS[{phase!r}] -> {delta[k]}")
            new_pmi = "{\n" + "".join(f'    "{ph}": {v},\n' for ph, v in pmi.items()) + "}"
            src = src[:m.start(2)] + new_pmi + src[m.end(2):]

    for const in ("CALL_EVERY", "PENAL_RAMP_ITERS", "BETA_DOUBLE_EVERY"):
        if const in delta:
            v = int(delta[const])
            src, n = re.subn(rf"^({const}\s*=\s*)\d+", f"\\g<1>{v}", src, flags=re.MULTILINE)
            if n: changes.append(f"  {const} -> {v}")

    if not changes: return "  (no changes)"
    if not dry_run:
        with open(agent_path, "w", encoding="utf-8") as f: f.write(src)
    return "\n".join(changes)

def validate_delta(delta: dict) -> tuple[dict, list[str]]:
    accepted, rejected = {}, []
    for key, val in delta.items():
        if key not in PARAM_BOUNDS:
            rejected.append(f"  {key}: unknown parameter, skipped")
            continue
        lo, hi = PARAM_BOUNDS[key]
        if not (lo <= val <= hi):
            rejected.append(f"  {key}={val} out of bounds [{lo}, {hi}], clipped")
            val = max(lo, min(hi, val))
        accepted[key] = val
    return accepted, rejected

META_SYSTEM_PROMPT = """You are a meta-optimizer for a topology optimization LLM controller.
Your job: analyze run results and propose config changes.

IMPORTANT ARCHITECTURE NOTE:
The LLM runs in 'direct' mode — it outputs numeric params directly each call.
PHASE_MIN_ITERS and PENAL_RAMP_ITERS only affect the FALLBACK schedule (API errors).
Do NOT change PHASE_MIN_ITERS — it has no effect on live LLM runs.

WHAT ACTUALLY MATTERS (only tune these):
- "CALL_EVERY" (int) [3 to 15] — frequency of LLM decisions. Lower = more adaptive.
- "GRAYNESS_GATE" (float) [0.10 to 0.35] — beta cap threshold. Lower = LLM can raise beta sooner.
- "BETA_DOUBLE_EVERY" (int) [5 to 20] — fallback beta ramp (affects error recovery).

COMPARING RESULTS:
- Compare llm_agent ONLY against three_field_continuation and expert_heuristic.
- tail_only is the NULL BASELINE — ignore it for gap calculation.
- Two metrics matter:
    FinalC: final compliance after tail (lower is better).
    TailEntryC: best_rho quality handed to tail (lower = better LLM exploration).
- If llm TailEntryC < heuristic TailEntryC but FinalC > heuristic FinalC:
    The LLM finds a better topology but the tail can't exploit it.
    Recommend: do NOT change LLM params. Note this in analysis. The fix is more tail iters.
- If llm TailEntryC > heuristic TailEntryC:
    LLM is not finding a good topology. Lower CALL_EVERY or GRAYNESS_GATE.

Rules:
1. NEVER propose PHASE_MIN_ITERS changes — they do nothing in direct mode.
2. If llm TailEntryC is already lower than best heuristic, output empty delta {}.
3. Make incremental changes only.
4. Never propose values outside bounds.

Respond ONLY with JSON matching the provided schema.
"""
def _call_gemini_meta(prompt: str, model: str, api_key: str) -> tuple[Optional[dict], Optional[str]]:
    import urllib.request
    import urllib.error

    model_path = model if model.startswith("models/") else f"models/{model}"
    url = f"https://generativelanguage.googleapis.com/v1beta/{model_path}:generateContent?key={api_key}"

    payload = {
        "system_instruction": {"parts": [{"text": META_SYSTEM_PROMPT}]},
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.2,
            "maxOutputTokens": 1024,
            "responseMimeType": "application/json",
            "responseSchema": {
                "type": "OBJECT",
                "properties": {
                    "llm_agent_FinalC": {"type": "NUMBER"},
                    "best_heuristic_FinalC": {"type": "NUMBER"},
                    "analysis": {"type": "STRING"},
                    "delta": {
                        "type": "OBJECT",
                        "properties": {
                            "GRAYNESS_GATE":   {"type": "NUMBER"},
                            "CALL_EVERY":      {"type": "INTEGER"},
                            "BETA_DOUBLE_EVERY": {"type": "INTEGER"}
                        }
                    }
                },
                "required": ["llm_agent_FinalC", "best_heuristic_FinalC", "analysis", "delta"]
            }
        }
    }

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})

    try:
        with urllib.request.urlopen(req) as response:
            resp_json = json.loads(response.read().decode("utf-8"))

        candidates = resp_json.get("candidates", [])
        if not candidates:
            return None, f"API returned no candidates. Raw response: {resp_json}"

        raw_text = candidates[0].get("content", {}).get("parts", [{}])[0].get("text", "")
        clean_text = raw_text.strip().replace("```json", "").replace("```", "").strip()
        return json.loads(clean_text), None

    except urllib.error.HTTPError as e:
        err_body = e.read().decode("utf-8")
        return None, f"HTTP Error {e.code}: {err_body}"
    except Exception as e:
        return None, f"REST API error: {e}"

def build_meta_prompt(summary: dict, current_config: dict, run_history: list[dict]) -> str:
    lines = ["=== Current run results ==="]
    lines.append("CRITICAL: FinalC = final compliance (LOWER IS BETTER).")
    lines.append("TailEntryC = compliance of best_rho handed to tail (lower = better exploration).")
    lines.append("tail_only is the NULL BASELINE (zero exploration). Ignore it when comparing LLM.")
    lines.append("Compare llm_agent only against: three_field_continuation, expert_heuristic.")
    lines.append("")

    for name, s in summary.items():
        fc  = s.get("final_C_mean", 0.0)
        fg  = s.get("final_G_mean", 0.0)
        tec = s.get("tail_entry_best_C", float("nan"))
        teg = s.get("tail_entry_best_G", float("nan"))
        tei = s.get("tail_entry_best_iter", -1)
        tv  = "valid" if s.get("tail_entry_valid", False) else "NO-VALID"
        lines.append(f"  {name:30s} | FinalC={fc:.4f} | TailEntryC={tec:.4f} | TailEntryG={teg:.4f}"
                     f" | TailIter={tei:.0f} | {tv}")

    lines.append("")
    lines.append("=== WHAT THE META-OPTIMIZER CAN ACTUALLY CHANGE ===")
    lines.append("The LLM runs in 'direct' mode — it outputs numeric params each call.")
    lines.append("PHASE_MIN_ITERS only affects the FALLBACK schedule (API errors). Do NOT tune it.")
    lines.append("WHAT ACTUALLY MATTERS for the live LLM:")
    lines.append("  CALL_EVERY: how often LLM is called. Lower = more decisions = more adaptive.")
    lines.append("  GRAYNESS_GATE: beta cap threshold. Lower = LLM can raise beta sooner.")
    lines.append("  BETA_DOUBLE_EVERY: fallback beta ramp speed (affects error recovery only).")
    lines.append("")
    lines.append("KEY INSIGHT: If llm TailEntryC is LOWER than heuristics but FinalC is HIGHER,")
    lines.append("  the tail (20 iters fixed) is not enough to exploit the LLM topology.")
    lines.append("  In this case, recommend increasing tail_iters in STANDARD_TAIL (not tunable here).")
    lines.append("")
    lines.append("=== Current LLM controller config ===")
    for k, v in current_config.items():
        lines.append(f"  {k} = {v}")

    if run_history:
        lines.append("")
        lines.append("=== Previous run history (last 3) ===")
        for i, h in enumerate(run_history[-3:]):
            llm_c  = h.get("llm_agent_final_c", "?")
            best_c = h.get("best_competitor_final_c", "?")
            delta  = h.get("accepted_delta", {})
            gap    = h.get("gap_pct", "?")
            lines.append(f"  Run {h.get('run', i)}: llm={llm_c}  best_heuristic={best_c}"
                         f"  gap={gap}%  delta={delta}")

    return "\n".join(lines)

def meta_step(results_path: str, agent_path: str, model: str, api_key: str, history_path: str, dry_run: bool = False, verbose: bool = True) -> dict:
    if not os.path.exists(results_path):
        raise FileNotFoundError(f"Results file not found: {results_path!r}")
    with open(results_path, encoding="utf-8") as f: summary = json.load(f)

    run_history = []
    if os.path.exists(history_path):
        with open(history_path, encoding="utf-8") as f: run_history = json.load(f)

    current_config = read_current_config(agent_path)
    prompt = build_meta_prompt(summary, current_config, run_history)

    if verbose: print("\n[MetaOptimizer] Calling Gemini meta-reflector (REST API)...")
    parsed, err = _call_gemini_meta(prompt, model, api_key)

    if err or not parsed:
        print(f"[MetaOptimizer] API error: {err}")
        return {"error": err}

    llm_c_extracted = parsed.get("llm_agent_FinalC")
    best_h_extracted = parsed.get("best_heuristic_FinalC")
    analysis = parsed.get("analysis", "")
    raw_delta = parsed.get("delta", {})

    if verbose:
        print(f"\n[MetaOptimizer] Extracted LLM FinalC: {llm_c_extracted} | Best Heuristic FinalC: {best_h_extracted}")
        print(f"[MetaOptimizer] Analysis:\n  {analysis}")
        print(f"\n[MetaOptimizer] Proposed delta: {raw_delta}")

    accepted, rejected = validate_delta(raw_delta)

    if verbose and rejected: print(f"[MetaOptimizer] Rejected/clipped:\n" + "\n".join(rejected))

    if not accepted:
        if verbose: print("[MetaOptimizer] No valid changes to apply.")
        change_log = "(no changes)"
    else:
        change_log = apply_delta(agent_path, accepted, dry_run=dry_run)
        if verbose: print(f"\n[MetaOptimizer] {'(DRY RUN) ' if dry_run else ''}Applied changes:\n{change_log}")

    def get_c(run_dict):
        c = run_dict.get("final_C_mean")
        if c is None: c = run_dict.get("final_compliance_mean")
        return c if c is not None else float("inf")

    competitors = {k: v for k, v in summary.items() if k not in ("fixed", "llm_agent")}
    best_c = min(get_c(v) for v in competitors.values()) if competitors else float("inf")
    llm_c = get_c(summary.get("llm_agent", {}))
    gap_pct = round((llm_c - best_c) / best_c * 100, 3) if best_c not in (0, float("inf")) else None

    record = {
        "run": len(run_history), "llm_agent_final_c": llm_c, "best_competitor_final_c": best_c,
        "gap_pct": gap_pct, "analysis": analysis, "proposed_delta": raw_delta,
        "accepted_delta": accepted, "config_before": current_config,
        "config_after": read_current_config(agent_path) if not dry_run else current_config,
    }
    run_history.append(record)

    if not dry_run:
        with open(history_path, "w", encoding="utf-8") as f: json.dump(run_history, f, indent=2)

    return record

def auto_loop(n_iters: int, run_cmd: str, results_path: str, agent_path: str, model: str, api_key: str, history_path: str, verbose: bool = True, converge_threshold_pct: float = 0.5):
    # FIX: Ensure subprocess resolves to the EXACT SAME python executable running this meta optimizer.
    if run_cmd.startswith("python "):
        run_cmd = f'"{sys.executable}"' + run_cmd[6:]
        
    # FIX: Ensure the API key actually passes to the spawned process so the LLM isn't skipped.
    env = os.environ.copy()
    env["GEMINI_API_KEY"] = api_key
    
    print(f"[MetaOptimizer] Starting auto-loop: {n_iters} iterations\n[MetaOptimizer] Run command: {run_cmd}\n[MetaOptimizer] Convergence threshold: {converge_threshold_pct}%\n")
    no_change_streak = 0
    
    for i in range(n_iters):
        print(f"\n{'='*60}\n[MetaOptimizer] === Meta-iteration {i+1}/{n_iters} ===\n{'='*60}")
        print(f"[MetaOptimizer] Running solver: {run_cmd}")
        
        result = subprocess.run(run_cmd, shell=True, env=env)
        if result.returncode != 0:
            print(f"[MetaOptimizer] Solver exited with code {result.returncode}. Stopping.")
            break

        if not os.path.exists(results_path):
            candidates = [os.path.join(os.path.dirname(results_path) or ".", n) for n in ("summary.json", "summary_pub.json")]
            found = next((c for c in candidates if os.path.exists(c)), None)
            if not found:
                import glob as _glob
                matches = sorted(_glob.glob("results/results_pub_*/summary.json"))
                if matches: found = matches[-1]
            if found:
                print(f"[MetaOptimizer] Note: using {found!r} (requested path not found).")
                results_path = found
            else:
                print(f"[MetaOptimizer] ERROR: no results file found. Tried: {candidates}")
                break

        record = meta_step(results_path, agent_path, model, api_key, history_path, False, verbose)
        gap = record.get("gap_pct")
        gap_str = f"{gap:+.3f}%" if isinstance(gap, float) else "n/a"
        print(f"\n[MetaOptimizer] Gap to best competitor: {gap_str}")

        if isinstance(gap, float) and gap <= converge_threshold_pct and gap > -999: # Also break if negative
            print(f"[MetaOptimizer] Converged (gap {gap:+.3f}% <= {converge_threshold_pct}%). Stopping.")
            break

        if not record.get("accepted_delta"):
            no_change_streak += 1
            if no_change_streak >= 2:
                print("[MetaOptimizer] No changes for 2 consecutive runs. Stopping.")
                break
            print("[MetaOptimizer] No changes this iteration -- continuing.")
        else:
            no_change_streak = 0

    print("\n[MetaOptimizer] Loop complete.")

def main():
    p = argparse.ArgumentParser(description="Meta-optimizer for pub_llm_agent.py")
    p.add_argument("--results", default="results/results_pub_cantilever_2d/summary.json", help="Path to summary JSON")
    p.add_argument("--agent", default="pub_llm_agent.py", help="Path to pub_llm_agent.py")
    p.add_argument("--history", default="meta_optimizer_history.json", help="Path to store run history JSON")
    p.add_argument("--model", default="gemini-3.1-flash-lite", help="Gemini model name")
    p.add_argument("--dry_run", action="store_true", help="Print proposed changes without writing")
    p.add_argument("--loop", action="store_true", help="Run auto-loop")
    p.add_argument("--n_iters", type=int, default=5, help="Number of meta-iterations")
    p.add_argument("--run_cmd", default="python pub_run_comparison.py", help="Shell command")
    p.add_argument("--converge", type=float, default=0.5, help="Stop loop when gap <= this %%")
    p.add_argument("--quiet", action="store_true", default=False, help="Suppress verbose output")
    args = p.parse_args()

    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key: sys.exit("ERROR: GEMINI_API_KEY not set")

    if args.loop:
        auto_loop(args.n_iters, args.run_cmd, args.results, args.agent, args.model, api_key, args.history, not args.quiet, args.converge)
    else:
        meta_step(args.results, args.agent, args.model, api_key, args.history, args.dry_run, not args.quiet)

if __name__ == "__main__":
    main()
