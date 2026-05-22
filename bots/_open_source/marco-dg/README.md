# Strategy Notes

The notebook has no markdown strategy section. The explicit strategy text present in the notebook appears in inline code comments, copied exactly below.

```text
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  ORBIT WARS — Agent v3.3 (public release)                                ║
# ║  Top leaderboard score achieved: 1060.5                                   ║
# ║  Author: Marco DG (marcodg)                                               ║
# ║                                                                           ║
# ║  Architecture: Beam Search + Evolutionary/Data-Driven Agent (EDA).       ║
# ║  Phase A: precomputes 1-hop graph, 2-hop chains, sync windows,           ║
# ║           chain_val potential function.                                   ║
# ║  Phase B: per-step beam search over candidate launch sequences,          ║
# ║           scored by production-based horizon value + chain bonus.        ║
# ║                                                                           ║
# ║  Key fix in v3.3 vs v3.2:                                                ║
# ║    _plan_best_launch uses the orbiting source position at time t         ║
# ║    (not ref_t), with 8 intercept iterations for accurate ballistics.     ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
```

```text
# ── Trajectory planner — opening strategy via beam search ────────────────────
#
# Replaces the greedy target scorer with a trajectory-value evaluator. For a
# candidate plan [(src₁,tgt₁), (src₂,tgt₂), ...] we simulate forward:
#
#   V(plan) = Σᵢ P_tgtᵢ × (R − t_captureᵢ)
#
# Each capture becomes a new source for later commits. Beam search explores
# the space of plans up to fixed depth, keeping the top-K by V.
# First moves of the best plan are launched; W>0 commits are deferred.
```
