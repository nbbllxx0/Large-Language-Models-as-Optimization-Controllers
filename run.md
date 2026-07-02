# Run Commands for Reproduction
# ========================================
#
# $pyExec = path to your Python interpreter
# All commands assume you are in the directory containing the pub_*.py files.
# Set GEMINI_API_KEY environment variable before running LLM experiments.


# ===================================================================
# OUTER CONFIGURATION LOOP â€?optional fast iteration
# ===================================================================

# Configuration pass for one problem â€?cantilever, fast mesh
& $pyExec pub_meta_optimizer.py --loop `
    --results "results/results_pub_cantilever_2d/summary.json" `
    --run_cmd "$pyExec -u pub_run_comparison.py --problem cantilever --mode 2d --preset fast" `
    --n_iters 8

# 60Ã—30, 100 iters per loop. Use this to tune GRAYNESS_GATE / CALL_EVERY / BETA_DOUBLE_EVERY quickly.

# Configuration pass for all three 2-D problems
foreach ($prob in @("cantilever","mbb","lbracket")) {
    & $pyExec pub_meta_optimizer.py --loop `
        --results "results/results_pub_${prob}_2d/summary.json" `
        --run_cmd "$pyExec -u pub_run_comparison.py --problem $prob --mode 2d --preset fast" `
        --n_iters 5
}

# Optimizes across all three geometries â€?prevents overfitting to cantilever.


# ===================================================================
# SINGLE PROBLEM â€?2D
# ===================================================================

# 2D fast â€?60Ã—30, 100 iters (quick check)
& $pyExec pub_run_comparison.py --problem cantilever --mode 2d --preset fast --verbose

# 2D long â€?120Ã—60, 300 iters (primary result)
& $pyExec pub_run_comparison.py --problem cantilever --mode 2d --preset long --verbose

# 2D hard â€?180Ã—90, 300 iters (high-res check)
& $pyExec pub_run_comparison.py --problem cantilever --mode 2d --preset hard --verbose


# ===================================================================
# SINGLE PROBLEM â€?3D
# ===================================================================

# 3D cantilever â€?standard 3D mesh
& $pyExec pub_run_comparison.py --problem cantilever --mode 3d --verbose

# nelz auto-set to nelyÃ·3. Requires pyamg for reasonable speed.

# 3D cantilever â€?explicit mesh size
& $pyExec pub_run_comparison.py --problem cantilever --mode 3d `
    --nelx 60 --nely 30 --nelz 10 --max-iter 300 --verbose

# 3D MBB beam â€?standard mesh
& $pyExec pub_run_comparison.py --problem mbb --mode 3d --verbose

# 3D MBB beam â€?explicit mesh size
& $pyExec pub_run_comparison.py --problem mbb --mode 3d `
    --nelx 40 --nely 20 --nelz 10 --max-iter 300 --verbose


# ===================================================================
# ALL 2D PROBLEMS â€?reproduction sweep
# ===================================================================

# 2D all problems, long preset
foreach ($prob in @("cantilever","mbb","lbracket")) {
    & $pyExec pub_run_comparison.py --problem $prob --mode 2d --preset long --verbose
}

# All problems, five run-indexed executions
foreach ($prob in @("cantilever","mbb","lbracket")) {
    & $pyExec pub_run_comparison.py --problem $prob --mode 2d `
        --preset long --n-runs 5
}

# Run after the optional configuration pass if using configured constants. Each problem ~5-8 min.

# Primary 2-D controlled runs
foreach ($prob in @("cantilever","mbb","lbracket")) {
    & $pyExec pub_run_comparison.py --problem $prob --mode 2d `
        --preset hard --n-runs 5
}

# 180Ã—90 mesh. Each problem ~15-20 min. Run overnight.


# ===================================================================
# ITEM 1: FIVE-RUN-INDEX 3D CANTILEVER
# ===================================================================
# This is the most critical missing experiment. The paper reports
# Supporting 3-D runs for the five-controller comparison.
# Repeated run indices should be interpreted as deterministic trajectory checks,
# not independent random initializations.

# 3D cantilever, 40Ã—20Ã—10, five run-indexed executions (~2 hours total)
& $pyExec pub_run_comparison.py --problem cantilever --mode 3d `
    --nelx 40 --nely 20 --nelz 10 --max-iter 300 --n-runs 5

# 3D cantilever, larger mesh 60Ã—30Ã—10 (~8 hours, run overnight)
& $pyExec pub_run_comparison.py --problem cantilever --mode 3d `
    --nelx 60 --nely 30 --nelz 10 --max-iter 300 --n-runs 1


# ===================================================================
# ITEM 2: TAIL-ONLY CONTROLLER (already in comparison)
# ===================================================================
# TailOnlyController is now included automatically in every run.
# No separate command needed â€?it runs alongside the other controllers.
# Check "tail_only" row in the console output table.
#
# Expected: tail_only >> fixed (tail helps even with no exploration)
#           tail_only >> llm_agent (exploration quality matters)
#           If llm_agent â‰?tail_only, the exploration adds nothing.


# ===================================================================
# ITEM 3: NO-META / PRE-CONFIGURATION-LOOP ABLATION
# ===================================================================
# Runs a second LLM agent instance with un-tuned hyperparameters
# (GRAYNESS_GATE=0.30, CALL_EVERY=8, BETA_DOUBLE_EVERY=15)
# to isolate the outer configuration loop contribution.

# Quick check â€?cantilever 2D, single run
& $pyExec pub_run_comparison.py --problem cantilever --mode 2d `
    --preset long --no-meta-ablation --verbose

# All 2-D problems, five run-indexed executions, with no-meta ablation
foreach ($prob in @("cantilever","mbb","lbracket")) {
    & $pyExec pub_run_comparison.py --problem $prob --mode 2d `
        --preset long --n-runs 5 --no-meta-ablation
}

# 3D cantilever with no-meta ablation, five run-indexed executions
& $pyExec pub_run_comparison.py --problem cantilever --mode 3d `
    --nelx 40 --nely 20 --nelz 10 --max-iter 300 --n-runs 5 --no-meta-ablation


# ===================================================================
# ITEM 4: 3D MBB BEAM
# ===================================================================
# Adds a second 3-D geometry beyond the cantilever case.

# 3D MBB, single seed (quick validation)
& $pyExec pub_run_comparison.py --problem mbb --mode 3d `
    --nelx 40 --nely 20 --nelz 10 --max-iter 300 --verbose

# 3D MBB, five run-indexed executions
& $pyExec pub_run_comparison.py --problem mbb --mode 3d `
    --nelx 40 --nely 20 --nelz 10 --max-iter 300 --n-runs 5

# 3D MBB with no-meta ablation, five run-indexed executions
& $pyExec pub_run_comparison.py --problem mbb --mode 3d `
    --nelx 40 --nely 20 --nelz 10 --max-iter 300 --n-runs 5 --no-meta-ablation


# ===================================================================
# RECOMMENDED EXECUTION ORDER
# ===================================================================
#
# 1. Outer configuration loop (tune agent hyperparameters):
#      Run the configuration commands above if using configured constants.
#
# 2. Primary 2D results (Table 5 in paper):
#      foreach prob in cantilever, mbb, lbracket:
#        pub_run_comparison.py --problem $prob --mode 2d --preset long --n-runs 5
#
# 3. Five-run-index 3D cantilever:
#      pub_run_comparison.py --problem cantilever --mode 3d
#        --nelx 40 --nely 20 --nelz 10 --max-iter 300 --n-runs 5
#
# 4. Five-run-index 3D MBB beam:
#      pub_run_comparison.py --problem mbb --mode 3d
#        --nelx 40 --nely 20 --nelz 10 --max-iter 300 --n-runs 5
#
# 5. No-meta / pre-configuration-loop ablation (all problems):
#      foreach prob in cantilever, mbb, lbracket:
#        pub_run_comparison.py --problem $prob --mode 2d
#          --preset long --n-runs 5 --no-meta-ablation
#      pub_run_comparison.py --problem cantilever --mode 3d
#        --nelx 40 --nely 20 --nelz 10 --max-iter 300 --n-runs 5 --no-meta-ablation
#
# Total estimated compute time depends on API latency and mesh size.
# Total estimated API cost: < $1 USD additional.


# ===================================================================
# USEFUL FLAGS FOR ANY COMMAND
# ===================================================================
# --verbose          # show per-iter LLM decisions (p= Î²= r= m=)
# --no-llm           # heuristics only, skip LLM (fast baseline check)
# --no-meta-ablation # also run pre-configuration-loop LLM agent
# --n-runs 1         # default â€?single run for iteration
# --n-runs 5         # five run-indexed executions
# --call-every 5     # LLM API call frequency (default 5)
# --model gemini-3.1-flash-lite   # current live endpoint example; archived logs retain their recorded identifiers
