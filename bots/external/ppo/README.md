# Architecture Decision

The [pilkwang](https://www.kaggle.com/code/pilkwang/orbit-wars-structured-baseline) Structured Baseline v11, used as the core because it has a far more sophisticated architecture:

* World Model with planet timelines, arrival forecasting, same-turn combat resolution
* Mission families: reinforce-to-hold, rescue, recapture, capture, snipe, swarm, crash-exploit, follow-up, live-doomed salvage, rear funneling
* Proactive defense with stacked enemy detection
* Commitment-aware planning that updates the future after each launch

6 Modifications Made
| # | Modification | Source | Purpose |
|---|---|---|---|
| 1 | **Parameter tuning** | Both | `HOSTILE_TARGET_VALUE_MULT` 1.85→2.0, `ELIMINATION_BONUS` 18→25, `FINISHING_DOMINATION` 0.35→0.30, `REINFORCE_MAX_SOURCE_FRACTION` 0.75→0.85, +2 new constants |
| 2 | **Comet heuristic fallback** | Hyperion | When `comet_planet_ids` is empty, detect comets by radius<1.5 AND production==1 |
| 3 | **Elimination bonus extended** | Hyperion | Apply `ELIMINATION_BONUS` against weak enemies in ALL phases, not just late-game |
| 4 | **Quick reinforcement pass** | Hyperion | Before main missions, transfer ships from surplus planets to nearby underdefended allies (within distance 25) |
| 5 | **Domination consolidation** | Hyperion | When dominating (domination>0.25), send 55% of remaining attack budget as aggressive "total war" fleets at enemy planets |
| 6 | **`LATE_AGGRESSION_FLEET_RATIO`** | Hyperion | New constant (0.55) controlling how much of the budget goes to consolidation attacks |
