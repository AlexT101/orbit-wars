# Orbit Wars Lab

Local tournament runner + visualizer for the
[Orbit Wars Kaggle competition](https://www.kaggle.com/competitions/orbit-wars).

## Quick start

### Option 1: Docker (recommended)

```bash
docker compose up
```

Open <http://localhost:6001>. Done.

First run builds the image (~3-5 min, pulls pytorch CPU). Subsequent `up`
is instant.

**Port conflict?** Set `PORT` to anything free:
```bash
PORT=7001 docker compose up
```

**macOS / non-standard UID:** to have files written by the container owned
by your host user (not `1000`), create a `.env` once:

```bash
cp .env.example .env          # shows available overrides
echo "UID=$(id -u)" > .env    # or just do this one-liner
echo "GID=$(id -g)" >> .env
```

---

## Architecture

```
viewer/              Vite + TypeScript SPA (vanilla DOM, no framework)
orbit_wars_app/      FastAPI backend + tournament runner (Python 3.12)
web/core/            Vendored @kaggle-environments/core (React replay player)
agents/
  baselines/         Reference agents (tracked in git)
  external/          Curated public notebooks (tracked in git)
  mine/              Your agents go here
runs/
  trueskill.json     Persistent TrueSkill state (seeded snapshot)
```

`docker-compose.yml` runs a single multi-stage image:

1. Node builder → `viewer/dist`
2. Python runtime → serves both API and the static viewer on port 8000
   (published as 6001)

---

## Match scheduler

Tournaments are executed by a process-wide scheduler
([`orbit_wars_app/scheduler.py`](orbit_wars_app/scheduler.py)) rather than run
one-at-a-time:

- **Queue any number of tournaments.** `POST /api/tournaments` expands a config
  into per-match jobs and returns immediately with `{run_id, status:"queued"}`.
  Multiple tournaments (and Quick Matches) can be in flight at once — there is no
  longer a single-tournament lock / 409.
- **Fair round-robin.** Each tournament keeps its own FIFO sub-queue; the
  scheduler interleaves one match per tournament in turn, so a large round-robin
  can't starve a Quick Match queued behind it.
- **One global concurrency setting.** A system-wide worker count (Match Settings
  on the Settings tab, or `PUT /api/scheduler/concurrency`) controls how many
  matches run at once, replacing the old per-tournament 1/2/4/8 picker. Backed by
  a killable [`pebble`](https://pypi.org/project/pebble/) process pool, so workers
  stay warm across matches.
- **Graceful failures.** Per-match crashes are recorded as `crashed`; a match
  that blows its deadline is killed and recorded as `timeout` — neither aborts
  the rest of the run. The deadline isn't user-configurable: it's derived from
  the player count via `match_timeout_for` (`(500 + 60) * players + 20` s, i.e.
  ~500 turns at 1s + 60s overage per player + slack).
- **Stop.** `POST /api/tournaments/{run_id}/stop` (alias `/cancel`) drops that
  tournament's queued matches and kills its in-flight ones.
- **Restart workers.** `POST /api/scheduler/restart-pool` (Match Settings →
  "Restart workers") recycles the pool so a freshly-rebuilt native bot binary
  (`.so`/`.pyd`, cached in warm workers) gets picked up; in-flight matches are
  re-queued, not lost.
- **Introspection.** `GET /api/scheduler` (concurrency, queue depth, active
  tournaments) and `GET /api/scheduler/running` (live matches) back the
  "Active now" panel on the Tournaments tab.

Scheduler/queue state is **in-memory**: a backend restart clears the queue (any
disk run still marked `running` is reconciled to `aborted` on startup). The
`runs/` tree remains the durable archive. The CLI
(`python -m orbit_wars_app.tournament run …`) runs synchronously through a
transient scheduler; its `--parallel` flag sets that run's local worker count.

---

## Credits

Rule-based external agents are redistributed from their authors' public
Kaggle notebooks (links + versions in each agent's `agent.yaml`). 

Orbit Wars lab from: https://github.com/automatylicza/orbit-wars-lab

Replay graphs from: https://github.com/MatthewWHuang/orbit-wars

---

## License

MIT for everything except graphs. See [`LICENSE`](LICENSE).

PolyForm Noncommercial License 1.0.0 for graphs. See [`LICENSE_2`](LICENSE_2).