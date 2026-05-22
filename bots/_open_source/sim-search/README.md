# Orbit Wars: Sim + Value Search Agent

## Agent Architecture

```
agent(obs)
  ├── Step 0-2: Detect moving planets
  ├── Defense phase (production-weighted)
  │     └── For each threatened planet, sorted by production DESC:
  │           Find fastest reinforcement fleet that arrives in time
  ├── Attack phase (search-ranked)
  │     └── For each owned planet, sorted by ships DESC:
  │           Generate top-K candidates by heuristic score
  │           Simulate each candidate 20 ticks forward
  │           Score terminal state with value function
  │           Pick best; record commitment for multi-source coordination
  └── Cooperative attack fallback
```

Reference Notebooks Used:

https://www.kaggle.com/code/aidensong123/lb-highest-1000-search-learned-value-function/notebook

https://www.kaggle.com/code/kashiwaba/orbit-wars-reinforcement-learning-tutorial

https://www.kaggle.com/code/djenkivanov/orbit-wars-agent-ow-proto-passed-1-000
