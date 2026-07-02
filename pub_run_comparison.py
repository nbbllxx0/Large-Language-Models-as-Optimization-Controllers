"""
pub_run_comparison.py
---------------------
Experiment runner for controlled SIMP continuation studies.
"""

from __future__ import annotations

import argparse
import json
import os
import time
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

try:
    import skimage.measure as measure
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    _SKIMAGE_AVAILABLE = True
except ImportError:
    _SKIMAGE_AVAILABLE = False

try:
    import plotly.graph_objects as go
    import plotly.io as pio
    _PLOTLY_AVAILABLE = True
except ImportError:
    _PLOTLY_AVAILABLE = False

from pub_simp_solver import SIMPParams, run_simp

warnings.filterwarnings("ignore")

PALETTE = {
    "fixed":                    "#888780",
    "tail_only":                "#B8860B",
    "reference_no_heaviside":   "#BBBBBB",
    "three_field_continuation":  "#378ADD",
    "expert_heuristic":         "#1D9E75",
    "schedule_only":            "#8E44AD",
    "llm_agent":                "#D85A30",
    "llm_agent_no_meta":        "#F4A460",
    "llm_minimal":              "#E67E22",
    "llm_verbose":              "#C0392B",
    "llm_compliance_focused":   "#922B21",
    "llm_T0p0":                 "#D85A30",
    "llm_T0p3":                 "#E8854A",
    "llm_T0p7":                 "#F0A06A",
    "llm_T1p0":                 "#F5BC8A",
}
MARKERS = {k: m for k, m in zip(PALETTE,["o","s","^","D","*","P","h","H","X","v","<",">","p"])}


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def _args():
    p = argparse.ArgumentParser()
    p.add_argument("--no-llm",      action="store_true")
    p.add_argument("--ablation",    action="store_true",
                   help="Also run prompt ablation variants (minimal, verbose, compliance_focused)")
    p.add_argument("--no-meta-ablation", action="store_true",
                   help="Also run LLM agent with pre-configuration-loop hyperparameters")
    p.add_argument("--review-ablation", action="store_true",
                   help="Also run controlled deterministic scaffold baselines")
    p.add_argument("--llm-no-restart", action="store_true",
                   help="Also run LLM agent with restart disabled")
    p.add_argument("--temperatures", type=str, default="",
                   help="Comma-separated temperatures, e.g. 0.0,0.3,0.7")
    p.add_argument("--reference",   action="store_true",
                   help="Include fixed-penal no-Heaviside reference (matches Sigmund 88-line)")
    p.add_argument("--verbose",     action="store_true")
    p.add_argument("--mode",        choices=["2d","3d"], default="2d")
    p.add_argument("--problem",     choices=["cantilever","mbb","lbracket"], default="cantilever")
    
    # --preset shortcuts: fast=60x30/100it, long=120x60/300it, hard=180x90/300it
    p.add_argument("--preset",      type=str, default=None, choices=["fast","long","hard"])
    p.add_argument("--nelx",        type=int, default=60)
    p.add_argument("--nely",        type=int, default=30)
    p.add_argument("--nelz",        type=int, default=0)
    p.add_argument("--volfrac",     type=float, default=0.4)
    p.add_argument("--max-iter",    type=int, default=300)   # 300 for meaningful LLM advantage
    p.add_argument("--min-iter",    type=int, default=60)
    p.add_argument("--tail-iters",  type=int, default=20)
    p.add_argument("--n-runs",      type=int, default=1)     # use 5 for final pub table
    p.add_argument("--run-offset",  type=int, default=0,
                   help="Offset applied to deterministic seed/run numbering for resumable batches")
    p.add_argument("--call-every",  type=int, default=5)     # matches CALL_EVERY in pub_llm_agent.py
    p.add_argument("--grayness-gate", type=float, default=0.20)
    p.add_argument("--model",       type=str, default="gemini-3.1-flash-lite")
    p.add_argument("--tag",         type=str, default="",
                   help="Append a safe suffix to the output directory")
    p.add_argument("--controllers", type=str, default="",
                   help=("Comma-separated controller names to run after the controller "
                         "set is built, e.g. fixed,grayness_rule,llm_agent. "
                         "Use this for low-memory resumable batches."))
    p.add_argument("--replay-log", type=str, default="",
                   help="Cached LLM call-log JSON to replay without API calls")
    p.add_argument("--replay-freeze-params", type=str, default="",
                   help=("Comma-separated LLM action keys for replay freeze ablations; "
                         "valid examples: penal,beta,rmin,move,restart"))
    p.add_argument("--no-plots",    action="store_true",
                   help="Skip figure generation for low-memory batch runs")
    p.add_argument("--checkpoint-dir", type=str, default=None)
    p.add_argument("--amg-threshold", type=int, default=3000,
                   help="DOF count above which AMG-preconditioned CG is used (default 3000; "
                        "lower = AMG used more aggressively on 3D meshes)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Params factory
# ---------------------------------------------------------------------------

def _make_params(args, seed=None) -> SIMPParams:
    nelz = 0
    if args.mode == "3d":
        nelz = args.nelz if args.nelz > 0 else max(5, args.nely // 3)
    return SIMPParams(
        nelx=args.nelx, nely=args.nely, nelz=nelz,
        volfrac=args.volfrac, penal=3.0, rmin=1.5, move=0.2,
        max_iter=args.max_iter, min_iter=args.min_iter,
        tail_default_iters=args.tail_iters,
        use_heaviside=True, beta_init=1.0, beta_max=32.0,
        min_penal_for_best=3.0, max_gray_for_best=0.25,
        seed=seed,
        checkpoint_dir=args.checkpoint_dir,
        amg_ndof_threshold=getattr(args, "amg_threshold", 3000),
    )

def _make_reference_params(args, seed=None) -> SIMPParams:
    nelz = 0
    if args.mode == "3d":
        nelz = args.nelz if args.nelz > 0 else max(5, args.nely // 3)
    return SIMPParams(
        nelx=args.nelx, nely=args.nely, nelz=nelz,
        volfrac=args.volfrac, penal=3.0, rmin=1.5, move=0.2,
        max_iter=args.max_iter, min_iter=args.min_iter,
        tail_default_iters=0,
        use_heaviside=False,
        min_penal_for_best=0.5, max_gray_for_best=1.0,
        seed=seed,
    )


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _plot_compliance(agg: dict, args, out_dir: str):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    # Exclude tail_only from compliance curves — its extreme values
    # (starting from uniform density) blow up the y-axis scale and make
    # all other controllers invisible.  tail_only results are reported
    # in the summary table instead.
    plot_agg = {k: v for k, v in agg.items() if k != "tail_only"}
    for ax_i, ax in enumerate(axes):
        for name, runs in plot_agg.items():
            c = PALETTE.get(name, "#444")
            hists = [r["compliance_history"] for r in runs]
            L = max(len(h) for h in hists)
            mat = np.array([h + [h[-1]]*(L-len(h)) for h in hists])
            mean, std = mat.mean(0), mat.std(0)
            xs = np.arange(L) if ax_i == 0 else np.arange(max(0, L//3), L)
            ys_m = mean if ax_i == 0 else mean[max(0, L//3):]
            ys_s = std  if ax_i == 0 else std[max(0, L//3):]
            ax.plot(xs, ys_m, color=c, lw=1.8, label=name.replace("_"," "))
            if len(runs) > 1:
                ax.fill_between(xs, ys_m-ys_s, ys_m+ys_s, color=c, alpha=0.15)
            if ax_i == 0:
                ax.scatter(L-1, ys_m[-1], color=c, marker=MARKERS.get(name,"o"), s=120, zorder=5)
            else:
                ax.axhline(np.mean([r["final_compliance"] for r in runs]),
                           color=c, lw=0.9, ls="--", alpha=0.5)
        ax.set_xlabel("Iteration", fontsize=10)
        ax.set_ylabel("Compliance", fontsize=10)
        ax.set_title(["Full convergence","Late-stage zoom"][ax_i], fontsize=11)
        ax.legend(fontsize=7, ncol=2)
        ax.grid(alpha=0.3)
    dim = args.mode.upper()
    nruns = f"  ({args.n_runs} runs)" if args.n_runs > 1 else ""
    plt.suptitle(f"{dim} {args.problem.upper()}  {args.nelx}×{args.nely}  "
                 f"vf={args.volfrac}{nruns}", fontsize=11)
    plt.tight_layout()
    path = os.path.join(out_dir, "compliance_curves.png")
    plt.savefig(path, dpi=160, bbox_inches="tight"); plt.close(); print(f"  Saved {path}")


def _plot_designs(agg: dict, args, out_dir: str):
    if args.mode == "3d":
        _plot_designs_3d(agg, args, out_dir); return

    names = list(agg.keys())
    n = len(names)
    fig = plt.figure(figsize=(4.2*n, 4.8))
    gs = gridspec.GridSpec(2, n, figure=fig, hspace=0.3, wspace=0.08)

    for col, name in enumerate(names):
        best_run = min(agg[name], key=lambda r: r["final_compliance"]) 
        
        ax = fig.add_subplot(gs[0, col])
        rho = best_run["rho_final"].reshape(args.nelx, args.nely).T
        comp = best_run["final_compliance"]
        gray = best_run["final_grayness"]
        vol  = np.mean(best_run["rho_final"])
        
        ax.imshow(1-rho, cmap="gray", vmin=0, vmax=1, origin="lower", aspect="equal")
        ax.set_title(f"{name.replace('_',' ')}\nFinal: C={comp:.3f}  G={gray:.4f}  V={vol:.3f}", fontsize=8)
        ax.axis("off")
        
        ax_h = fig.add_subplot(gs[1, col])
        ax_h.hist(best_run["rho_final"].ravel(), bins=50, color=PALETTE.get(name,"#888"), alpha=0.75, density=True)
        ax_h.axvline(args.volfrac, color="k", lw=0.8, ls="--")
        ax_h.set_xlim(0,1); ax_h.set_xlabel("ρ̄", fontsize=7)
        ax_h.set_title(f"G={gray:.4f}", fontsize=7)
        ax_h.tick_params(labelsize=6)
        
    plt.suptitle("Final density fields + histograms", fontsize=10)
    path = os.path.join(out_dir, "designs.png")
    plt.savefig(path, dpi=160, bbox_inches="tight"); plt.close(); print(f"  Saved {path}")


def _plot_designs_3d(agg: dict, args, out_dir: str):
    names = list(agg.keys())
    n = len(names)
    nelz = args.nelz if args.nelz > 0 else max(5, args.nely//3)
    fig, axes = plt.subplots(3, n, figsize=(4.0*n, 7.0))
    if n == 1: axes = axes.reshape(3,1)
    
    for col, name in enumerate(names):
        best_run = min(agg[name], key=lambda r: r["final_compliance"])
        rho = best_run["rho_final"].reshape(args.nelx, args.nely, nelz)
        vol = np.mean(best_run["rho_final"])
        
        # We use mean projections (X-rays) rather than a single slice.
        # This shows the entire integrated density, making hollow structures easily understandable.
        for row, (sl, lbl) in enumerate([
            (np.mean(rho, axis=2).T, "XY Projection"),
            (np.mean(rho, axis=1).T, "XZ Projection"),
            (np.mean(rho, axis=0).T, "YZ Projection"),
        ]):
            ax = axes[row, col]
            ax.imshow(1-sl, cmap="gray", vmin=0, vmax=1, origin="lower", aspect="equal")
            if row == 0:
                ax.set_title(f"{name.replace('_',' ')}\nFinal: C={best_run['final_compliance']:.3f}  V={vol:.3f}", fontsize=8)
            ax.set_ylabel(lbl, fontsize=7); ax.axis("off")
            
    plt.suptitle("3D Density Projections (X-Ray)", fontsize=10)
    path = os.path.join(out_dir, "designs_3d.png")
    plt.savefig(path, dpi=160, bbox_inches="tight"); plt.close(); print(f"  Saved {path}")


def _generate_3d_visualizations(agg: dict, args, out_dir: str):
    if not _SKIMAGE_AVAILABLE:
        print("  scikit-image not installed. Skipping marching cubes 3D visualizations.")
        return

    for name, runs in agg.items():
        if name == "reference_no_heaviside": continue 
        
        best_run = min(runs, key=lambda r: r["final_compliance"])
        nelx, nely, nelz = best_run["nelx"], best_run["nely"], best_run["nelz"]
        
        if nelz == 0: continue # Only for 3D

        # Reshape to 3D grid
        rho_3d = best_run["rho_final"].reshape(nelx, nely, nelz)

        # Pad array with zeros so marching cubes closes the outer boundaries cleanly
        padded_rho = np.zeros((nelx + 2, nely + 2, nelz + 2))
        padded_rho[1:-1, 1:-1, 1:-1] = rho_3d

        try:
            verts, faces, normals, values = measure.marching_cubes(padded_rho, level=0.5)
            
            # Shift vertices back due to padding
            verts_x = verts[:, 0] - 1
            verts_y = verts[:, 1] - 1
            verts_z = verts[:, 2] - 1
            
            # Map to physical aspect ratio
            aspect_x = 1.0
            aspect_y = nely / nelx
            aspect_z = nelz / nelx
            
            verts_mapped_x = (verts_x / (max(nelx - 1, 1))) * aspect_x - 0.5 * aspect_x
            verts_mapped_y = (verts_y / (max(nely - 1, 1))) * aspect_y - 0.5 * aspect_y
            verts_mapped_z = (verts_z / (max(nelz - 1, 1))) * aspect_z - 0.5 * aspect_z
            
            verts_mapped = np.stack([verts_mapped_x, verts_mapped_y, verts_mapped_z], axis=1)

            # --- Matplotlib Static Plot ---
            fig = plt.figure(figsize=(8, 6))
            ax = fig.add_subplot(111, projection='3d')
            
            mesh = Poly3DCollection(verts_mapped[faces])
            mesh.set_edgecolor('k')
            mesh.set_linewidth(0.1)
            mesh.set_facecolor(PALETTE.get(name, '#378ADD'))
            mesh.set_alpha(1.0)
            ax.add_collection3d(mesh)
            
            ax.set_xlim(-0.5 * aspect_x, 0.5 * aspect_x)
            ax.set_ylim(-0.5 * aspect_y, 0.5 * aspect_y)
            ax.set_zlim(-0.5 * aspect_z, 0.5 * aspect_z)
            
            ax.set_box_aspect([aspect_x, aspect_y, aspect_z])
            ax.set_axis_off()
            
            title = f"{name.replace('_', ' ')} (Final C={best_run['final_compliance']:.3f})"
            plt.title(title, fontsize=10)
            ax.view_init(elev=30, azim=-60)
            
            screenshot_path = os.path.join(out_dir, f"{name}_final_3d.png")
            plt.savefig(screenshot_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"  Saved {screenshot_path}")

            # --- Plotly Interactive HTML (if available) ---
            if _PLOTLY_AVAILABLE:
                hex_color = PALETTE.get(name, '#378ADD')
                plotly_mesh = go.Mesh3d(
                    x=verts_mapped[:, 0], 
                    y=verts_mapped[:, 1], 
                    z=verts_mapped[:, 2],
                    i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
                    color=hex_color, opacity=1.0, flatshading=True,
                    lighting=dict(ambient=0.4, diffuse=0.5, specular=0.1, roughness=0.1),
                    name=name
                )
                
                fig_html = go.Figure(data=[plotly_mesh])
                fig_html.update_layout(
                    title_text=title,
                    scene=dict(
                        xaxis=dict(visible=False),
                        yaxis=dict(visible=False),
                        zaxis=dict(visible=False),
                        aspectmode='manual',
                        aspectratio=dict(x=aspect_x, y=aspect_y, z=aspect_z),
                        camera=dict(eye=dict(x=1.5, y=1.2, z=1.2))
                    ),
                    margin=dict(l=0, r=0, b=0, t=40)
                )
                html_path = os.path.join(out_dir, f"{name}_final_3d_interactive.html")
                pio.write_html(fig_html, file=html_path, include_plotlyjs="cdn")
                print(f"  Saved {html_path}")
                
        except Exception as e:
            print(f"  [WARNING] Marching cubes failed for {name}: {e}")


def _plot_params(agg: dict, out_dir: str):
    fig, axes = plt.subplots(4, 1, figsize=(11, 9), sharex=True)
    keys = ["penal","rmin","move","beta"]
    labels =["Penalty p","Filter r_min","Move limit","Heaviside β"]
    plot_agg = {k: v for k, v in agg.items() if k != "tail_only"}
    for idx, (key, ylabel) in enumerate(zip(keys, labels)):
        ax = axes[idx]
        for name, runs in plot_agg.items():
            log = runs[0]["params_log"]
            ax.plot([d["iter"] for d in log],[d.get(key, float("nan")) for d in log],
                    color=PALETTE.get(name,"#333"), lw=1.6,
                    label=name.replace("_"," "))
        ax.set_ylabel(ylabel, fontsize=9); ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7, ncol=3)
    axes[-1].set_xlabel("Iteration", fontsize=10)
    plt.suptitle("Hyperparameter trajectories", fontsize=11)
    plt.tight_layout()
    path = os.path.join(out_dir, "param_traces.png")
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close(); print(f"  Saved {path}")


def _plot_grayness(agg: dict, out_dir: str):
    fig, ax = plt.subplots(figsize=(10, 4))
    plot_agg = {k: v for k, v in agg.items() if k != "tail_only"}
    for name, runs in plot_agg.items():
        c = PALETTE.get(name,"#333")
        log = runs[0]["params_log"]
        ax.plot([d["iter"] for d in log],[d.get("grayness", float("nan")) for d in log],
                color=c, lw=1.6, label=name.replace("_"," "))
    ax.axhline(0.20, color="k", lw=0.8, ls="--", alpha=0.6, label="gate (0.20)")
    ax.set_xlabel("Iteration", fontsize=10); ax.set_ylabel("Grayness", fontsize=10)
    ax.set_title("Grayness convergence — when designs become binary", fontsize=11)
    ax.legend(fontsize=8, ncol=2); ax.grid(alpha=0.3)
    plt.tight_layout()
    path = os.path.join(out_dir, "grayness.png")
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close(); print(f"  Saved {path}")


def _plot_boxplot(agg: dict, out_dir: str):
    names = list(agg.keys())
    fig, ax = plt.subplots(figsize=(7, 5))
    vals = [[r["final_compliance"] for r in agg[n]] for n in names]
    bp = ax.boxplot(vals, patch_artist=True, medianprops=dict(color="k", lw=2))
    for patch, name in zip(bp["boxes"], names):
        patch.set_facecolor(PALETTE.get(name,"#aaa")); patch.set_alpha(0.7)
    ax.set_xticks(range(1, len(names)+1))
    ax.set_xticklabels([n.replace("_","\n") for n in names], fontsize=7)
    ax.set_title("Final compliance distribution", fontsize=11); ax.set_ylabel("Compliance", fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    plt.suptitle(f"Statistical distribution ({len(list(agg.values())[0])} runs)", fontsize=11)
    plt.tight_layout()
    path = os.path.join(out_dir, "statistical_box.png")
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close(); print(f"  Saved {path}")


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def _tail_entry_stats(run: dict) -> dict:
    """
    Extract the state of best_rho at the moment the tail begins.
    ONLY reports the valid best (gate_ok=True snapshots) — these are what
    the tail restart actually uses.  Pre-valid fallback snapshots are
    exposed separately for transparency but never drive the tail.

    Key diagnostic for exploration quality:
      lower best_C + lower best_G at later best_iter = better exploration
      (the controller found a lower-compliance, more-binary topology before
      handing it to the identical tail process).
    """
    valid = run.get("best_is_valid", False)
    return {
        # What the tail actually restarted from (valid gate only)
        "best_C":        round(float(run.get("best_compliance", float("nan"))), 5),
        "best_iter":     int(run.get("best_iteration", -1)),
        "best_G":        round(float(run.get("best_grayness",   float("nan"))), 5),
        "best_is_valid": valid,
        # Pre-valid tracking (diagnostic only, never used by tail)
        "pre_valid_C":   round(float(run.get("pre_valid_best_compliance", float("nan"))), 5),
        "pre_valid_iter":int(run.get("pre_valid_best_iteration", -1)),
        "pre_valid_G":   round(float(run.get("pre_valid_best_grayness",   float("nan"))), 5),
        "tail_enabled":  run.get("tail_config", {}).get("enabled", False),
    }


def _save_summary(agg: dict, args, out_dir: str) -> dict:
    ref_key = "reference_no_heaviside" if "reference_no_heaviside" in agg else "fixed"
    ref_runs = agg.get(ref_key, [])
    ref_final = None
    if ref_runs:
        ref_final = float(np.mean([r["final_compliance"] for r in ref_runs]))

    summary = {}
    for name, runs in agg.items():
        fc  = [r["final_compliance"] for r in runs]
        fg  = [r["final_grayness"]   for r in runs]
        fv  = [np.mean(r["rho_final"]) for r in runs]
        wt  = [r.get("wall_time", 0) for r in runs]
        tes = [_tail_entry_stats(r)   for r in runs]
        summary[name] = {
            "final_C_mean":        round(float(np.mean(fc)), 5),
            "final_C_std":         round(float(np.std(fc)),  5),
            "final_G_mean":        round(float(np.mean(fg)), 5),
            "final_V_mean":        round(float(np.mean(fv)), 4),
            "final_vs_ref%":       None if ref_final is None or ref_final == 0.0
                                   else round(100*(np.mean(fc)-ref_final)/ref_final, 2),
            "wall_time_mean":      round(float(np.mean(wt)), 1),
            "n_runs":              len(runs),
            # --- Tail-entry diagnostics (what tail actually restarted from) ---
            "tail_entry_best_C":    round(float(np.mean([t["best_C"]        for t in tes])), 5),
            "tail_entry_best_iter": round(float(np.mean([t["best_iter"]     for t in tes])), 1),
            "tail_entry_best_G":    round(float(np.mean([t["best_G"]        for t in tes])), 5),
            "tail_entry_valid":     all(t["best_is_valid"]                  for t in tes),
            # Pre-valid diagnostics (never drives tail — transparency only)
            "pre_valid_C":          round(float(np.mean([t["pre_valid_C"]   for t in tes])), 5),
            "pre_valid_iter":       round(float(np.mean([t["pre_valid_iter"]for t in tes])), 1),
            "pre_valid_G":          round(float(np.mean([t["pre_valid_G"]   for t in tes])), 5),
            "tail_enabled":         tes[0]["tail_enabled"] if tes else False,
        }
    summary["_metadata"] = {
        "problem": args.problem,
        "mode": args.mode,
        "nelx": args.nelx,
        "nely": args.nely,
        "nelz": args.nelz,
        "volfrac": args.volfrac,
        "max_iter": args.max_iter,
        "n_runs": args.n_runs,
        "call_every": args.call_every,
        "model": args.model,
        "grayness_gate": args.grayness_gate,
        "review_ablation": bool(args.review_ablation),
        "llm_no_restart": bool(args.llm_no_restart),
        "no_meta_ablation": bool(args.no_meta_ablation),
        "controllers": args.controllers,
        "replay_log": args.replay_log,
        "replay_freeze_params": args.replay_freeze_params,
    }
    path = os.path.join(out_dir, "summary.json")
    with open(path, "w") as f: json.dump(summary, f, indent=2)
    print(f"  Saved {path}")
    return summary


def _save_per_run_metrics(agg: dict, out_dir: str) -> list[dict]:
    rows = []
    for name, runs in agg.items():
        for idx, run in enumerate(runs):
            tail = _tail_entry_stats(run)
            rows.append({
                "controller": name,
                "run_idx": int(run.get("run_idx", idx)),
                "seed": run.get("seed"),
                "final_compliance": float(run["final_compliance"]),
                "final_grayness": float(run["final_grayness"]),
                "final_volume": float(np.mean(run["rho_final"])),
                "wall_time": float(run.get("wall_time", 0.0)),
                "tail_entry_best_C": float(tail["best_C"]),
                "tail_entry_best_iter": float(tail["best_iter"]),
                "tail_entry_best_G": float(tail["best_G"]),
                "tail_entry_valid": bool(tail["best_is_valid"]),
                "tail_enabled": bool(tail["tail_enabled"]),
            })
    path = os.path.join(out_dir, "per_run_metrics.json")
    with open(path, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"  Saved {path}")
    return rows


def _save_incremental_metrics(agg: dict, out_dir: str):
    rows = []
    for name, runs in agg.items():
        for idx, run in enumerate(runs):
            try:
                tail = _tail_entry_stats(run)
                rows.append({
                    "controller": name,
                    "run_idx": int(run.get("run_idx", idx)),
                    "seed": run.get("seed"),
                    "final_compliance": float(run["final_compliance"]),
                    "final_grayness": float(run["final_grayness"]),
                    "final_volume": float(np.mean(run["rho_final"])),
                    "wall_time": float(run.get("wall_time", 0.0)),
                    "tail_entry_best_C": float(tail["best_C"]),
                    "tail_entry_best_iter": float(tail["best_iter"]),
                    "tail_entry_best_G": float(tail["best_G"]),
                    "tail_entry_valid": bool(tail["best_is_valid"]),
                    "tail_enabled": bool(tail["tail_enabled"]),
                })
            except Exception as exc:
                rows.append({"controller": name, "run_idx": idx, "error": str(exc)})
    path = os.path.join(out_dir, "per_run_metrics.partial.json")
    with open(path, "w") as f:
        json.dump(rows, f, indent=2)


def _select_controllers(controllers: list, selected: str) -> list:
    wanted = [name.strip() for name in selected.split(",") if name.strip()]
    if not wanted:
        return controllers

    by_name = {ctrl.name: ctrl for ctrl in controllers}
    missing = [name for name in wanted if name not in by_name]
    if missing:
        available = ", ".join(sorted(by_name))
        raise ValueError(
            "Unknown controller(s): "
            + ", ".join(missing)
            + f". Available controllers: {available}"
        )
    return [by_name[name] for name in wanted]


# ---------------------------------------------------------------------------
# Console table
# ---------------------------------------------------------------------------

def _print_table(summary: dict):
    w = 138
    print(f"\n{'='*w}")
    print(f"{'Controller':<28} {'FinalC mean±std':>18} {'FinalG':>8} {'vs ref%':>9}"
          f" {'TailEntryC':>11} {'TailEntryG':>11} {'TailIter':>9} {'Valid?':>7} {'t(s)':>7}")
    print(f"{'-'*w}")
    for name, s in summary.items():
        if name.startswith("_"):
            continue
        fs    = f"{s['final_C_mean']:.4f}±{s['final_C_std']:.4f}"
        vr    = f"{s['final_vs_ref%']}%" if s['final_vs_ref%'] is not None else "—"
        ten   = "(no tail)"   if not s.get("tail_enabled", True)  else ""
        valid = s.get("tail_entry_valid", False)
        if not s.get("tail_enabled", True):
            tec, teg, tei, vf = "—", "—", "—", "—"
        else:
            tec = f"{s.get('tail_entry_best_C', float('nan')):.4f}"
            teg = f"{s.get('tail_entry_best_G', float('nan')):.4f}"
            tei = f"{s.get('tail_entry_best_iter', -1):.0f}"
            vf  = "YES" if valid else "NO(*)"
        print(f"{name:<28} {fs:>18} {s['final_G_mean']:>8.4f} {vr:>9}"
              f" {tec:>11} {teg:>11} {tei:>9} {vf:>7} {s['wall_time_mean']:>7.1f}  {ten}")
    print(f"{'='*w}")
    print(f"  TailEntry* = best_rho handed to the tail (valid gate: penal>=3.0, gray<0.25).")
    print(f"  Valid?=NO(*) means no gate-valid snapshot was found; tail ran from final rho.")
    print(f"  Exploration quality: lower TailEntryC + lower TailEntryG = better best_rho.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = _args()

    # Apply preset shortcuts — override nelx/nely/max_iter
    if args.preset == "fast":
        args.nelx, args.nely, args.max_iter = 60, 30, 100
    elif args.preset == "long":
        args.nelx, args.nely, args.max_iter = 120, 60, 300
    elif args.preset == "hard":
        args.nelx, args.nely, args.max_iter = 180, 90, 300

    tag = ("_" + "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in args.tag)) if args.tag else ""
    out_dir = os.path.join("results", f"results_pub_{args.problem}_{args.mode}{tag}")
    os.makedirs(out_dir, exist_ok=True)

    # ---- Build controller list ----
    from pub_baseline_controller import (
        FixedController, ThreeFieldContinuation, ExpertHeuristic,
        ScheduleOnlyController, TailOnlyController, FixedTailController,
        GraynessRuleController, GraynessRuleRestartController)

    controllers = [
        FixedController(),           # true no-intervention baseline (no tail)
        TailOnlyController(),        # zero-exploration + tail (null ablation)
        ThreeFieldContinuation(),    # best academic heuristic
        ExpertHeuristic(),           # expert heuristic
        ScheduleOnlyController(),    # schedule-only ablation
    ]
    if args.review_ablation:
        controllers.extend([
            FixedTailController(),
            GraynessRuleController(gate=args.grayness_gate),
            GraynessRuleRestartController(gate=args.grayness_gate),
        ])

    if args.replay_log:
        from pub_llm_agent import ReplayLLMController
        controllers.append(ReplayLLMController(
            args.replay_log, gate_threshold=args.grayness_gate))
        for key in [k.strip() for k in args.replay_freeze_params.split(",") if k.strip()]:
            controllers.append(ReplayLLMController(
                args.replay_log, freeze_params=[key],
                gate_threshold=args.grayness_gate))

    llm_available = False
    if not args.no_llm:
        try:
            from pub_llm_agent import LLMController, PromptAblationController, TemperatureAblationController, NoMetaLLMController, LLMNoRestartController
            controllers.append(LLMController(
                model=args.model, call_every=args.call_every,
                verbose=args.verbose, max_iter=args.max_iter,
                gate_threshold=args.grayness_gate))
            llm_available = True

            if getattr(args, 'llm_no_restart', False):
                controllers.append(LLMNoRestartController(
                    model=args.model, call_every=args.call_every,
                    verbose=args.verbose, max_iter=args.max_iter,
                    gate_threshold=args.grayness_gate))

            if getattr(args, 'no_meta_ablation', False):
                controllers.append(NoMetaLLMController(
                    model=args.model, verbose=args.verbose,
                    max_iter=args.max_iter))

            if args.ablation:
                for key in ["minimal", "verbose", "compliance_focused"]:
                    controllers.append(PromptAblationController(
                        system_prompt_key=key, model=args.model,
                        call_every=args.call_every, verbose=args.verbose,
                        max_iter=args.max_iter))

            if args.temperatures:
                temps =[float(t) for t in args.temperatures.split(",")]
                for T in temps:
                    if abs(T - 0.0) > 1e-6:
                        controllers.append(TemperatureAblationController(
                            temperature=T, model=args.model,
                            call_every=args.call_every, verbose=args.verbose))
        except Exception as e:
            print(f"  [WARNING] LLM controllers unavailable: {e}")

    controllers = _select_controllers(controllers, args.controllers)
    if args.controllers:
        print("  Selected controllers: " + ", ".join(c.name for c in controllers))

    agg: dict[str, list[dict]] = {c.name:[] for c in controllers}
    if args.reference:
        agg["reference_no_heaviside"] =[]

    # ---- Run experiments ----
    for run_idx in range(args.n_runs):
        actual_run_idx = args.run_offset + run_idx
        seed = 42 + actual_run_idx * 7
        params = _make_params(args, seed=seed)
        print(f"\n{'='*65}")
        print(f"  RUN {run_idx+1}/{args.n_runs}  seed={seed}"
              f"  {args.mode.upper()} {args.problem}  {args.nelx}×{args.nely}"
              + (f"×{params.nelz}" if params.nelz else ""))
        print(f"{'='*65}")

        if args.reference:
            rp = _make_reference_params(args, seed=seed)
            t0 = time.time()
            rr = run_simp(rp, callback=FixedController(), problem=args.problem)
            rr["wall_time"] = time.time()-t0
            rr["run_idx"] = actual_run_idx
            rr["seed"] = seed
            agg["reference_no_heaviside"].append(rr)
            vf_ref = np.mean(rr["rho_final"])
            print(f"  reference_no_heaviside  Final C={rr['final_compliance']:.4f}  "
                  f"gray={rr['final_grayness']:.4f}  vf={vf_ref:.4f}  t={rr['wall_time']:.1f}s")

        for ctrl in controllers:
            print(f"\n  --- {ctrl.name} ---")
            if hasattr(ctrl, "call_log"):
                ctrl.call_log = []
            t0 = time.time()
            res = run_simp(params, callback=ctrl, verbose=args.verbose,
                           problem=args.problem)
            res["wall_time"] = time.time()-t0
            res["run_idx"] = actual_run_idx
            res["seed"] = seed
            agg[ctrl.name].append(res)
            
            vf_ctrl = np.mean(res["rho_final"])
            print(f"  final_C={res['final_compliance']:.4f}  "
                  f"final_G={res['final_grayness']:.4f}  "
                  f"final_V={vf_ctrl:.4f}  "
                  f"t={res['wall_time']:.1f}s")
            
            if hasattr(ctrl, "call_log") and ctrl.call_log:
                lp = os.path.join(out_dir, f"{ctrl.name}_call_log_run{actual_run_idx}.json")
                with open(lp, "w") as f:
                    json.dump(ctrl.call_log, f, indent=2)
                ctrl.call_log = []
            _save_incremental_metrics(agg, out_dir)

    # ---- Plots ----
    if args.no_plots:
        print("\n  Skipping plots (--no-plots).")
    else:
        print("\n  Generating plots...")
        _plot_compliance(agg, args, out_dir)
        _plot_designs(agg, args, out_dir)
        _plot_params(agg, out_dir)
        _plot_grayness(agg, out_dir)
        if args.n_runs > 1:
            _plot_boxplot(agg, out_dir)
        if args.mode == "3d":
            _generate_3d_visualizations(agg, args, out_dir) 

    summary = _save_summary(agg, args, out_dir)
    _save_per_run_metrics(agg, out_dir)
    _print_table(summary)
    print(f"\nAll outputs → ./{out_dir}/")

if __name__ == "__main__":
    main()
