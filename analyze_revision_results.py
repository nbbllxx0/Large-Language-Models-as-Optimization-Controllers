from __future__ import annotations

import argparse
import csv
import itertools
import json
from pathlib import Path
from statistics import mean


def _load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _bootstrap_ci(values: list[float], n_boot: int = 10000) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    if len(values) == 1:
        return values[0], values[0]
    # Deterministic bootstrap for reproducible manuscript tables.
    samples = []
    n = len(values)
    state = 123456789
    for _ in range(n_boot):
        draw = []
        for _ in range(n):
            state = (1103515245 * state + 12345) & 0x7FFFFFFF
            draw.append(values[state % n])
        samples.append(mean(draw))
    samples.sort()
    return samples[int(0.025 * n_boot)], samples[int(0.975 * n_boot)]


def _paired_signflip_p(diffs: list[float]) -> float:
    diffs = [d for d in diffs if abs(d) > 1e-12]
    if not diffs:
        return 1.0
    observed = abs(mean(diffs))
    count = 0
    extreme = 0
    for signs in itertools.product((-1.0, 1.0), repeat=len(diffs)):
        val = abs(mean([s * d for s, d in zip(signs, diffs)]))
        count += 1
        if val >= observed - 1e-12:
            extreme += 1
    return extreme / count


def _call_log_stats(result_dir: Path) -> list[dict]:
    rows = []
    for path in sorted(result_dir.glob("*_call_log_run*.json")):
        controller = path.name.split("_call_log_run", 1)[0]
        calls = _load_json(path)
        modes = {}
        latency = []
        input_chars = 0
        output_chars = 0
        for entry in calls:
            mode = entry.get("mode", "unknown")
            modes[mode] = modes.get(mode, 0) + 1
            if "latency_s" in entry:
                latency.append(float(entry["latency_s"]))
            input_chars += int(entry.get("input_chars", 0) or 0)
            output_chars += int(entry.get("output_chars", 0) or 0)
        rows.append({
            "result_dir": result_dir.name,
            "controller": controller,
            "run_file": path.name,
            "total_log_entries": len(calls),
            "llm_calls": modes.get("llm", 0),
            "error_calls": modes.get("error", 0),
            "fallback_entries": modes.get("fallback", 0),
            "cooldown_entries": modes.get("cooldown", 0),
            "mean_latency_s": round(mean(latency), 4) if latency else 0.0,
            "total_latency_s": round(sum(latency), 4),
            "input_chars": input_chars,
            "output_chars": output_chars,
            "input_tokens_est": round(input_chars / 4.0),
            "output_tokens_est": round(output_chars / 4.0),
        })
    return rows


def analyze_result_dir(result_dir: Path, reference: str) -> tuple[list[dict], list[dict], list[dict]]:
    per_run = _load_json(result_dir / "per_run_metrics.json")
    by_controller: dict[str, dict[int, float]] = {}
    summary_rows = []
    for row in per_run:
        by_controller.setdefault(row["controller"], {})[int(row["run_idx"])] = float(row["final_compliance"])

    ref = by_controller.get(reference)
    if not ref:
        raise ValueError(f"{result_dir}: reference controller not found: {reference}")

    for controller, values_by_run in sorted(by_controller.items()):
        values = [values_by_run[i] for i in sorted(values_by_run)]
        ci_lo, ci_hi = _bootstrap_ci(values)
        common = sorted(set(ref) & set(values_by_run))
        diffs = [values_by_run[i] - ref[i] for i in common]
        pct_diffs = [100.0 * (values_by_run[i] - ref[i]) / ref[i] for i in common]
        summary_rows.append({
            "result_dir": result_dir.name,
            "controller": controller,
            "n": len(values),
            "mean_final_C": round(mean(values), 6),
            "ci95_low": round(ci_lo, 6),
            "ci95_high": round(ci_hi, 6),
            f"mean_delta_vs_{reference}": round(mean(diffs), 6) if diffs else "",
            f"mean_pct_vs_{reference}": round(mean(pct_diffs), 4) if pct_diffs else "",
            f"signflip_p_vs_{reference}": round(_paired_signflip_p(diffs), 6) if diffs else "",
        })

    pairwise_rows = []
    llm = by_controller.get("llm_agent")
    if llm:
        for controller, values_by_run in sorted(by_controller.items()):
            if controller == "llm_agent":
                continue
            common = sorted(set(llm) & set(values_by_run))
            if not common:
                continue
            diffs = [llm[i] - values_by_run[i] for i in common]
            pct = [100.0 * (llm[i] - values_by_run[i]) / values_by_run[i] for i in common]
            pairwise_rows.append({
                "result_dir": result_dir.name,
                "comparison": f"llm_agent_minus_{controller}",
                "n_pairs": len(common),
                "mean_delta": round(mean(diffs), 6),
                "mean_pct": round(mean(pct), 4),
                "signflip_p": round(_paired_signflip_p(diffs), 6),
            })

    return summary_rows, pairwise_rows, _call_log_stats(result_dir)


def _write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dirs", nargs="+")
    parser.add_argument("--reference", default="fixed")
    parser.add_argument("--out", default="revision_analysis")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict] = []
    pairwise_rows: list[dict] = []
    overhead_rows: list[dict] = []
    for raw in args.result_dirs:
        result_dir = Path(raw)
        s, p, o = analyze_result_dir(result_dir, args.reference)
        summary_rows.extend(s)
        pairwise_rows.extend(p)
        overhead_rows.extend(o)

    _write_csv(out / "controller_summary_stats.csv", summary_rows)
    _write_csv(out / "llm_pairwise_tests.csv", pairwise_rows)
    _write_csv(out / "llm_overhead_call_stats.csv", overhead_rows)
    print(f"Wrote analysis CSVs to {out}")


if __name__ == "__main__":
    main()
