from __future__ import annotations

import argparse
import csv
import itertools
import json
from pathlib import Path
from statistics import mean, pstdev


GROUP_FIELDS = ("mode", "problem", "nelx", "nely", "nelz", "volfrac", "max_iter", "grayness_gate")


def _load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _metrics_path(result_dir: Path) -> Path:
    final_path = result_dir / "per_run_metrics.json"
    if final_path.exists():
        return final_path
    partial_path = result_dir / "per_run_metrics.partial.json"
    if partial_path.exists():
        return partial_path
    raise FileNotFoundError(f"No per-run metrics found in {result_dir}")


def _read_metric_rows(result_dir: Path) -> list[dict]:
    path = _metrics_path(result_dir)
    rows = _load_json(path)
    summary_path = result_dir / "summary.json"
    metadata = {}
    if summary_path.exists():
        metadata = _load_json(summary_path).get("_metadata", {})
    out = []
    for row in rows:
        if "error" in row:
            continue
        item = dict(row)
        item["result_dir"] = result_dir.name
        item["metrics_file"] = path.name
        for key in GROUP_FIELDS:
            if key in metadata and key not in item:
                item[key] = metadata[key]
        out.append(item)
    return out


def _bootstrap_ci(values: list[float], n_boot: int = 10000) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    if len(values) == 1:
        return values[0], values[0]
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
    total = 0
    extreme = 0
    for signs in itertools.product((-1.0, 1.0), repeat=len(diffs)):
        val = abs(mean([s * d for s, d in zip(signs, diffs)]))
        total += 1
        if val >= observed - 1e-12:
            extreme += 1
    return extreme / total


def _summarize(rows: list[dict], reference: str) -> tuple[list[dict], list[dict]]:
    by_group: dict[tuple[str, ...], list[dict]] = {}
    by_group_run: dict[tuple[str, ...], dict[int, dict]] = {}
    for row in rows:
        group_key = tuple(str(row.get(field, "")) for field in GROUP_FIELDS)
        key = group_key + (str(row["controller"]),)
        by_group.setdefault(key, []).append(row)
        by_group_run.setdefault(key, {})[int(row["run_idx"])] = row

    summary_rows = []
    for key, ctrl_rows in sorted(by_group.items()):
        group_key = key[:-1]
        controller = key[-1]
        ref = by_group_run.get(group_key + (reference,), {})
        ctrl_by_run = by_group_run[key]
        final_c = [float(r["final_compliance"]) for r in ctrl_rows]
        final_g = [float(r["final_grayness"]) for r in ctrl_rows]
        wall = [float(r.get("wall_time", 0.0)) for r in ctrl_rows]
        ci_lo, ci_hi = _bootstrap_ci(final_c)
        common = sorted(set(ref) & set(ctrl_by_run))
        diffs = [
            float(ctrl_by_run[i]["final_compliance"])
            - float(ref[i]["final_compliance"])
            for i in common
        ]
        pct = [
            100.0
            * (
                float(ctrl_by_run[i]["final_compliance"])
                - float(ref[i]["final_compliance"])
            )
            / float(ref[i]["final_compliance"])
            for i in common
        ]
        summary = {field: group_key[i] for i, field in enumerate(GROUP_FIELDS)}
        summary.update({
            "controller": controller,
            "n": len(final_c),
            "mean_final_C": round(mean(final_c), 6),
            "std_final_C": round(pstdev(final_c), 6) if len(final_c) > 1 else 0.0,
            "ci95_low": round(ci_lo, 6),
            "ci95_high": round(ci_hi, 6),
            "mean_final_G": round(mean(final_g), 8),
            "mean_wall_time_s": round(mean(wall), 3),
            f"mean_delta_vs_{reference}": round(mean(diffs), 6) if diffs else "",
            f"mean_pct_vs_{reference}": round(mean(pct), 4) if pct else "",
            f"signflip_p_vs_{reference}": round(_paired_signflip_p(diffs), 6) if diffs else "",
            "run_indices": " ".join(str(int(r["run_idx"])) for r in sorted(ctrl_rows, key=lambda x: int(x["run_idx"]))),
            "seeds": " ".join(str(r.get("seed", "")) for r in sorted(ctrl_rows, key=lambda x: int(x["run_idx"]))),
        })
        summary_rows.append(summary)

    pairwise_rows = []
    for group_key in sorted({k[:-1] for k in by_group_run}):
        llm = by_group_run.get(group_key + ("llm_agent",))
        if not llm:
            continue
        for key in sorted(by_group_run):
            controller = key[-1]
            if key[:-1] != group_key or controller == "llm_agent":
                continue
            ctrl_by_run = by_group_run[key]
            common = sorted(set(llm) & set(ctrl_by_run))
            if not common:
                continue
            diffs = [
                float(llm[i]["final_compliance"])
                - float(ctrl_by_run[i]["final_compliance"])
                for i in common
            ]
            pct = [
                100.0
                * (
                    float(llm[i]["final_compliance"])
                    - float(ctrl_by_run[i]["final_compliance"])
                )
                / float(ctrl_by_run[i]["final_compliance"])
                for i in common
            ]
            pairwise = {field: group_key[i] for i, field in enumerate(GROUP_FIELDS)}
            pairwise.update({
                "comparison": f"llm_agent_minus_{controller}",
                "n_pairs": len(common),
                "mean_delta": round(mean(diffs), 6),
                "mean_pct": round(mean(pct), 4),
                "signflip_p": round(_paired_signflip_p(diffs), 6),
                "run_indices": " ".join(str(i) for i in common),
            })
            pairwise_rows.append(pairwise)

    return summary_rows, pairwise_rows


def _call_log_stats(result_dir: Path) -> list[dict]:
    rows = []
    metadata = {}
    summary_path = result_dir / "summary.json"
    if summary_path.exists():
        metadata = _load_json(summary_path).get("_metadata", {})
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
        row = {field: metadata.get(field, "") for field in GROUP_FIELDS}
        row.update({
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
        rows.append(row)
    return rows


def _write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dirs", nargs="+")
    parser.add_argument("--reference", default="fixed_tail")
    parser.add_argument("--out", default="../analysis/controlled_batches")
    args = parser.parse_args()

    result_dirs = [Path(p) for p in args.result_dirs]
    rows = []
    overhead_rows = []
    seen = set()
    for result_dir in result_dirs:
        for row in _read_metric_rows(result_dir):
            key = (
                row.get("mode", ""),
                row.get("problem", ""),
                row.get("nelx", ""),
                row.get("nely", ""),
                row.get("nelz", ""),
                row.get("volfrac", ""),
                row.get("max_iter", ""),
                row.get("grayness_gate", ""),
                row["controller"],
                int(row["run_idx"]),
                row.get("seed"),
            )
            if key in seen:
                continue
            seen.add(key)
            rows.append(row)
        overhead_rows.extend(_call_log_stats(result_dir))

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    summary_rows, pairwise_rows = _summarize(rows, args.reference)
    _write_csv(out / "combined_per_run_metrics.csv", rows)
    _write_csv(out / "combined_controller_summary.csv", summary_rows)
    _write_csv(out / "combined_llm_pairwise_tests.csv", pairwise_rows)
    _write_csv(out / "combined_llm_overhead_call_stats.csv", overhead_rows)
    with (out / "combined_per_run_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    print(f"Wrote merged analysis to {out}")


if __name__ == "__main__":
    main()
