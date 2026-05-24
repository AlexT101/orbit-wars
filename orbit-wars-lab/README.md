# Orbit Wars Lab

Local tournament runner + visualizer for the
[Orbit Wars Kaggle competition](https://www.kaggle.com/competitions/orbit-wars).

## Quick start

### Option 1: Docker (recommended)

```bash
git clone https://github.com/automatylicza/orbit-wars-lab.git
cd orbit-wars-lab
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

### Option 2: Native dev (faster iteration)

Requires **Python 3.12** + **pnpm** (`npm i -g pnpm`).
On Windows, use **Git Bash** in VSCode (or standalone Git Bash) rather than
PowerShell or `C:\Windows\System32\bash.exe`.

```bash
git clone https://github.com/automatylicza/orbit-wars-lab.git
cd orbit-wars-lab
bash scripts/dev.sh
```

The first run creates `.venv`, installs missing Python/JS deps, and starts
the backend (:8000) + Vite viewer (:5173) with hot-reload. Open
<http://localhost:5173>.

If you also want the optional PPO RL agent dependencies (`torch` CPU wheel),
run `bash scripts/setup.sh --with-rl` once before starting dev mode.

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

## Credits

Rule-based external agents are redistributed from their authors' public
Kaggle notebooks (links + versions in each agent's `agent.yaml`). 

Orbit Wars lab from: https://github.com/automatylicza/orbit-wars-lab

---

## License

MIT. See [`LICENSE`](LICENSE).
