# Orbit Wars LB 1000+ : Search + Learned Value Function

**A different approach: replace heuristic move scoring with simulated outcomes ranked by a learned model that knows what winning looks like.**

---

```
For each candidate move:
    1. Simulate the game state +20 turns forward in a byte-equivalent local sim
    2. Score the simulated terminal state with a Gradient-Boosted Classifier
       trained on top-agent replays to predict P(I win | this state)
Pick the move with the highest P(win).
```

### Three components

#### 1. Fast local simulator
Byte-equivalent to the Kaggle orbit_wars env, but **~7,000 turns/sec** (the Kaggle env runs ~1 game/min). This makes per-move simulation feasible inside the 1-second turn budget.

#### 2. Learned value function `V(state) → P(I win)`
- Trained on **26,784** (state, eventual_winner) pairs from **196** top-agent winning replays plus **240,982** pairs from **1,000 self-play games**
- **Gradient-boosted classifier**, 500 trees × depth 6
- 16 features: ship_lead, production_lead, planet/ship percentages, centrality, in-flight fraction, game phase, 2P/4P format
- Validation **AUC 0.976**, Brier score 0.054
- Deployed as plain-Python tree walks — zero sklearn/numpy dependency at inference

#### 3. 1-ply minimax search
- Heuristic generates top-K candidates per source planet
- Forward-sim each candidate (with one heuristic opponent counter-move at sim tick 1)
- Multi-source coordination: source N's sim sees sources 1..N-1's committed fleets
- Pick by simulated `V(terminal_state)`

---

## The agent — `main.py`

Layered structure:
1. Imports + namedtuples (Planet, Fleet)
2. Value-function loader (reads tree dump from attached Kaggle Dataset)
3. Heuristic candidate generation
4. Forward simulator (`simulate_outcome`)
5. Value-function scoring (`_value_score`)
6. Search-based action selection + multi-source coordination
7. Safety wrapper (`agent(obs, config)`)

## Packaging Notes

The notebook-generated `main.py` looks for `value_gbc_trees_big.py` or `value_gbc_trees.py` from an attached Kaggle dataset. That dataset file was not present in this repo, so only the emitted `main.py` could be harvested here. The packaged agent still imports and exposes `agent()`, but it will fall back to heuristic scoring when the external tree dump is absent.
