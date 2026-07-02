"""
generate_latex_table.py
-----------------------
Reads results_pub/summary_pub.json and generates a LaTeX table
ready to paste into a paper.

Usage:
    python generate_latex_table.py
    python generate_latex_table.py --input path/to/summary.json
"""

from __future__ import annotations
import argparse
import json
import os


DISPLAY_NAMES = {
    "fixed":                    "Fixed (no continuation)",
    "three_field_continuation": "Three-field continuation",
    "expert_heuristic":         "Expert heuristic",
    "mbb_heuristic":            "MBB heuristic",
    "llm_agent":                "LLM phase-decision agent",
}


def _fmt(val, std=None, bold=False) -> str:
    if val is None:
        return "--"
    s = f"{val:.4f}"
    if std is not None and std > 0:
        s = f"{val:.4f} $\\pm$ {std:.4f}"
    if bold:
        s = f"\\textbf{{{s}}}"
    return s


def _pct(val) -> str:
    if val is None:
        return "--"
    sign = "+" if val > 0 else ""
    return f"{sign}{val:.2f}\\%"


def generate_table(summary: dict, caption: str = "Compliance comparison") -> str:
    names = list(summary.keys())

    # Find best (lowest) best_compliance_mean
    best_best = min(s["best_compliance_mean"] for s in summary.values())
    best_final = min(s["final_compliance_mean"] for s in summary.values())

    n_runs = list(summary.values())[0].get("n_runs", 1)
    multi = n_runs > 1

    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"  \centering")
    lines.append(f"  \\caption{{{caption} ({n_runs} run{'s' if n_runs > 1 else ''})}}")
    lines.append(r"  \label{tab:compliance_comparison}")

    if multi:
        col_spec = "l" + "c" * 6
        lines.append(f"  \\begin{{tabular}}{{{col_spec}}}")
        lines.append(r"    \toprule")
        lines.append(
            r"    Controller & Best $C$ (mean$\pm$std) & Final $C$ (mean$\pm$std) & "
            r"Best $G$ & Final $G$ & vs.\ fixed (best) & vs.\ fixed (final) \\"
        )
    else:
        col_spec = "l" + "c" * 6
        lines.append(f"  \\begin{{tabular}}{{{col_spec}}}")
        lines.append(r"    \toprule")
        lines.append(
            r"    Controller & Best $C$ & Final $C$ & "
            r"Best $G$ & Final $G$ & vs.\ fixed (best) & vs.\ fixed (final) \\"
        )

    lines.append(r"    \midrule")

    for name in names:
        s = summary[name]
        display = DISPLAY_NAMES.get(name, name.replace("_", " "))

        is_best_best  = abs(s["best_compliance_mean"]  - best_best)  < 1e-6
        is_best_final = abs(s["final_compliance_mean"] - best_final) < 1e-6

        if multi:
            best_str  = _fmt(s["best_compliance_mean"],  s.get("best_compliance_std"),  is_best_best)
            final_str = _fmt(s["final_compliance_mean"], s.get("final_compliance_std"), is_best_final)
        else:
            best_str  = _fmt(s["best_compliance_mean"],  bold=is_best_best)
            final_str = _fmt(s["final_compliance_mean"], bold=is_best_final)

        row = (
            f"    {display} & {best_str} & {final_str} & "
            f"{s['best_grayness_mean']:.4f} & {s['final_grayness_mean']:.4f} & "
            f"{_pct(s['best_vs_fixed_pct'])} & {_pct(s['final_vs_fixed_pct'])} \\\\"
        )
        lines.append(row)

    lines.append(r"    \bottomrule")
    lines.append(r"  \end{tabular}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="results_pub/summary_pub.json")
    p.add_argument("--caption", default="Compliance comparison across controllers")
    args = p.parse_args()

    if not os.path.exists(args.input):
        print(f"ERROR: {args.input} not found. Run pub_run_comparison.py first.")
        return

    with open(args.input) as f:
        summary = json.load(f)

    table = generate_table(summary, caption=args.caption)
    print(table)

    out = args.input.replace(".json", "_table.tex")
    with open(out, "w") as f:
        f.write(table)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
