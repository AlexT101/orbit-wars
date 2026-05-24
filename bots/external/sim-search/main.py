
# ============================================================
# Orbit Wars | Improved Agent
# Combines: Proto heuristics + production-weighted defense +
#           forward sim + learned value function (GBC)
# ============================================================

import math
import time
import os
from collections import namedtuple

# ── Namedtuples (match Kaggle env shapes) ─────────────────────────────────
_Planet = namedtuple("Planet", ["id","owner","x","y","radius","ships","production"])
_Fleet  = namedtuple("Fleet",  ["id","owner","x","y","angle","from_planet_id","ships"])

class _OW:
    Planet = _Planet
    Fleet  = _Fleet
ow = _OW()

def _read(obs, key, default=None):
    if obs is None: return default
    if isinstance(obs, dict): return obs.get(key, default)
    return getattr(obs, key, default)

# ── Tunable constants ──────────────────────────────────────────────────────
MAX_SPEED             = 6.0
MIN_SHIPS_ATTACK      = 5
MIN_SHIPS_COOP_TARGET = 20
COOP_PLANET_CAP       = 8

FORMULA_DIST          = 100
FORMULA_PROD_MULT     = 15
FORMULA_ENEMY_BONUS   = 10
FORMULA_SHIPS_PERCENT = 0.7

SIM_LOOKAHEAD         = 20
SIM_TOP_K             = 6
SIM_TIME_BUDGET_S     = 0.50

# ── Module-level state (reset per game by safety wrapper) ─────────────────
fleet_trajectories         = []
reinforcement_trajectories = []
moving_planets             = []
steps                      = 0
_game_sig                  = None
_last_obs_step             = -1


# ============================================================
# SECTION 1 — Value Function Loader
# Reads GBC tree dump from attached Kaggle Dataset.
# Falls back to ship-lead heuristic if no file found.
# ============================================================

GBC_INIT      = 0.0
GBC_N_TREES   = 0
GBC_TREES     = []
_VALUE_LOADED = False

def _load_value_model():
    global _VALUE_LOADED, GBC_INIT, GBC_N_TREES, GBC_TREES
    if _VALUE_LOADED: return
    paths = []
    if os.path.isdir('/kaggle/input'):
        for sub in os.listdir('/kaggle/input'):
            for fn in ('value_gbc_trees_big.py', 'value_gbc_trees.py'):
                p = os.path.join('/kaggle/input', sub, fn)
                if os.path.exists(p): paths.append(p)
    local = os.path.dirname(os.path.abspath(__file__ if '__file__' in dir() else '.'))
    for fn in ('value_gbc_trees_big.py', 'value_gbc_trees.py'):
        p = os.path.join(local, fn)
        if os.path.exists(p): paths.append(p)
    if not paths: return
    ns = {}
    with open(paths[0]) as f: exec(f.read(), ns)
    GBC_INIT    = ns.get('GBC_INIT', 0.0)
    GBC_N_TREES = ns.get('GBC_N_TREES', 0)
    GBC_TREES   = ns.get('GBC_TREES', [])
    _VALUE_LOADED = True

_load_value_model()


# ============================================================
# SECTION 2 — Value Function Features
# ============================================================

def _value_state_features(planet_states, fleet_states, my_player, n_players, step):
    on_planet_ships = {}
    planet_count    = {}
    production      = {}
    centrality_sum  = {}
    centrality_count= {}

    for s in planet_states:
        o = s['owner']
        on_planet_ships[o] = on_planet_ships.get(o, 0.0) + s['ships']
        if o != -1:
            planet_count[o]      = planet_count.get(o, 0) + 1
            production[o]        = production.get(o, 0.0) + s['production']
            d = math.hypot(s['x'] - 50, s['y'] - 50)
            centrality_sum[o]    = centrality_sum.get(o, 0.0) + max(0, 60 - d)
            centrality_count[o]  = centrality_count.get(o, 0) + 1

    # In-flight ships per owner (new — not in Doc3)
    inflight_ships = {}
    for f in fleet_states:
        inflight_ships[f['owner']] = inflight_ships.get(f['owner'], 0.0) + f['ships']

    my_planets  = planet_count.get(my_player, 0)
    my_ships    = on_planet_ships.get(my_player, 0.0)
    my_inflight = inflight_ships.get(my_player, 0.0)
    my_prod     = production.get(my_player, 0.0)
    my_cent     = (centrality_sum.get(my_player, 0.0) /
                   max(1, centrality_count.get(my_player, 0)))

    enemy_owners = [o for o in planet_count if o != -1 and o != my_player]
    if not enemy_owners:
        be_planets = 0; be_ships = 0.0; be_inflight = 0.0
        be_prod = 0.0; be_cent = 0.0
    else:
        best_o     = max(enemy_owners,
                         key=lambda o: on_planet_ships.get(o,0) + 100*planet_count.get(o,0))
        be_planets  = planet_count.get(best_o, 0)
        be_ships    = on_planet_ships.get(best_o, 0.0)
        be_inflight = inflight_ships.get(best_o, 0.0)
        be_prod     = production.get(best_o, 0.0)
        be_cent     = (centrality_sum.get(best_o, 0.0) /
                       max(1, centrality_count.get(best_o, 0)))

    total_ships   = sum(on_planet_ships.values()) + sum(inflight_ships.values())
    total_planets = len(planet_states)
    total_prod    = sum(production.values())
    sd = lambda a, b: a / b if b > 1e-9 else 0.0

    return [
        # Original 16 (Doc3 compatible)
        step / 500.0,
        n_players / 4.0,
        sd(my_planets,  total_planets),
        sd(be_planets,  total_planets),
        sd(my_ships,    total_ships),
        sd(be_ships,    total_ships),
        sd(my_prod,     total_prod),
        sd(be_prod,     total_prod),
        my_cent / 60.0,
        be_cent / 60.0,
        0.0,                                                          # reserved
        sd(my_ships  - be_ships,   total_ships),
        sd(my_planets - be_planets, total_planets),
        sd(my_prod   - be_prod,    total_prod),
        1.0 if n_players == 2 else 0.0,
        1.0 if n_players == 4 else 0.0,
        # New 4: in-flight awareness
        sd(my_inflight, total_ships),
        sd(be_inflight, total_ships),
        sd(my_ships + my_inflight - be_ships - be_inflight, total_ships),
        1.0 if my_inflight > be_inflight else 0.0,
    ]


def _value_score(features):
    """Walk GBC trees; returns raw logit (higher = better for my_player)."""
    z = GBC_INIT
    for feat_list, thr_list, val_list, left_list, right_list in GBC_TREES:
        node = 0
        while feat_list[node] >= 0:
            node = left_list[node] if features[feat_list[node]] <= thr_list[node] \
                   else right_list[node]
        z += val_list[node]
    return z


# ============================================================
# SECTION 3 — Physics / Geometry Helpers
# ============================================================

def fleet_speed(ships):
    return 1.0 + (MAX_SPEED - 1.0) * (math.log(max(1, ships)) / math.log(1000)) ** 1.5

def angle_to(src, tgt):
    return math.atan2(tgt.y - src.y, tgt.x - src.x)

def collides_segment(x1, y1, x2, y2, cx, cy, r):
    vx, vy = x2 - x1, y2 - y1
    wx, wy = cx - x1, cy - y1
    lsq    = vx*vx + vy*vy
    if lsq == 0:
        return wx*wx + wy*wy <= r*r
    t  = max(0.0, min(1.0, (wx*vx + wy*vy) / lsq))
    dx = x1 + t*vx - cx
    dy = y1 + t*vy - cy
    return dx*dx + dy*dy <= r*r

def sun_collision(planet, spd, angle, ticks=61):
    px, py = planet.x, planet.y
    for tick in range(1, ticks):
        nx = planet.x + math.cos(angle) * spd * tick
        ny = planet.y + math.sin(angle) * spd * tick
        if collides_segment(px, py, nx, ny, 50, 50, 10):
            return True
        px, py = nx, ny
    return False

def get_planet_trajectories(p, vel, ticks=61):
    angle = math.atan2(p.y - 50, p.x - 50)
    r     = math.hypot(p.x - 50, p.y - 50)
    return [(50 + r*math.cos(angle + vel*t),
             50 + r*math.sin(angle + vel*t))
            for t in range(1, ticks)]

def find_intercept(src, tgt, ships, vel):
    """Return (angle, tick) to intercept a moving planet, or (None, None)."""
    spd   = fleet_speed(ships)
    trajs = get_planet_trajectories(tgt, vel)
    for tick, (tx, ty) in enumerate(trajs, 1):
        dx, dy       = tx - src.x, ty - src.y
        dist_to_tgt  = math.hypot(dx, dy) - src.radius
        travel       = spd * tick
        if abs(travel - dist_to_tgt) > tgt.radius:
            continue
        ang = math.atan2(dy, dx)
        if not sun_collision(src, spd, ang):
            return ang, tick
    return None, None

def get_closest_planets_to_target(mine, t):
    return sorted([(m, math.hypot(m.x - t.x, m.y - t.y)) for m in mine],
                  key=lambda k: k[1])


# ============================================================
# SECTION 4 — Forward Simulator
# ============================================================

def _sim_fleet_target(fx, fy, fang, fships, planets_static, lookahead):
    spd = fleet_speed(fships)
    for tick in range(1, lookahead + 1):
        nx = fx + math.cos(fang) * spd * tick
        ny = fy + math.sin(fang) * spd * tick
        for i, p in enumerate(planets_static):
            if collides_segment(fx, fy, nx, ny, p['x'], p['y'], p['radius']):
                return i, tick
        fx, fy = nx, ny
    return None, None


def _sim_capture(state, arrivals):
    """Battle resolution: strongest fleet wins, loses difference in ships."""
    by_owner = {state['owner']: state['ships']}
    for owner, ships in arrivals:
        by_owner[owner] = by_owner.get(owner, 0.0) + ships
    ranked = sorted(by_owner.items(), key=lambda kv: kv[1], reverse=True)
    if len(ranked) == 1:
        state['owner'] = ranked[0][0]
        state['ships'] = ranked[0][1]
    else:
        state['owner'] = ranked[0][0]
        state['ships'] = ranked[0][1] - ranked[1][1]


def simulate_outcome(planets, fleets, my_player,
                     cand_src, cand_angle, cand_ships,
                     lookahead=SIM_LOOKAHEAD):
    # Static planet geometry
    ps    = [{'id': p.id, 'x': float(p.x), 'y': float(p.y),
              'radius': float(p.radius)} for p in planets]
    state = [{'owner': p.owner, 'ships': float(p.ships),
              'production': float(p.production),
              'x': float(p.x), 'y': float(p.y)} for p in planets]
    id2i  = {p['id']: i for i, p in enumerate(ps)}
    arrivals = [{} for _ in ps]

    # Schedule existing in-flight fleets
    for f in fleets:
        idx, tick = _sim_fleet_target(float(f.x), float(f.y), float(f.angle),
                                       int(f.ships), ps, lookahead)
        if idx is not None:
            arrivals[idx].setdefault(tick, []).append((f.owner, float(f.ships)))

    # Schedule candidate fleet; deduct ships from source immediately
    si = id2i.get(cand_src.id)
    if si is not None and cand_ships > 0:
        state[si]['ships'] = max(0.0, state[si]['ships'] - cand_ships)
        ca, sa = math.cos(cand_angle), math.sin(cand_angle)
        lx = ps[si]['x'] + ca * (ps[si]['radius'] + 0.1)
        ly = ps[si]['y'] + sa * (ps[si]['radius'] + 0.1)
        idx, tick = _sim_fleet_target(lx, ly, cand_angle, cand_ships, ps, lookahead)
        if idx is not None:
            arrivals[idx].setdefault(tick, []).append((my_player, float(cand_ships)))

    # 1-ply opponent counter: each opponent's strongest planet sends ~45% ships
    opponents = {p.owner for p in planets if p.owner not in (-1, my_player)}
    for opp in opponents:
        opp_srcs = [p for p in planets if p.owner == opp and p.ships >= 10]
        if not opp_srcs: continue
        src = max(opp_srcs, key=lambda p: p.ships)
        tgts = [p for p in planets if p.owner != opp]
        if not tgts: continue
        tgt = max(tgts,
                  key=lambda t: (50.0 if t.owner == my_player else 0.0)
                                - math.hypot(src.x - t.x, src.y - t.y)
                                - 0.5 * t.ships)
        osh = int(src.ships * 0.45)
        if osh < MIN_SHIPS_ATTACK: continue
        oang = math.atan2(tgt.y - src.y, tgt.x - src.x)
        lx   = src.x + math.cos(oang) * (src.radius + 0.1)
        ly   = src.y + math.sin(oang) * (src.radius + 0.1)
        idx, tick = _sim_fleet_target(lx, ly, oang, osh, ps, lookahead)
        if idx is not None:
            osi = id2i.get(src.id)
            if osi is not None:
                state[osi]['ships'] = max(0.0, state[osi]['ships'] - osh)
            arrivals[idx].setdefault(tick, []).append((opp, float(osh)))

    # Simulate forward tick by tick
    for tick in range(1, lookahead + 1):
        for s in state:
            if s['owner'] != -1:
                s['ships'] += s['production']
        for i, s in enumerate(state):
            arrs = arrivals[i].get(tick)
            if arrs:
                _sim_capture(s, arrs)

    # Score terminal state with value function
    fleet_states    = [{'owner': f.owner, 'ships': float(f.ships)} for f in fleets]
    owners_seen     = {p.owner for p in planets if p.owner != -1}
    feats = _value_state_features(state, fleet_states, my_player,
                                   max(2, len(owners_seen)), 0)
    # Fallback when GBC not loaded: use ship-fraction lead
    if GBC_N_TREES == 0:
        return feats[4] - feats[5]
    return _value_score(feats)


# ============================================================
# SECTION 5 — Target Scoring Heuristic
# ============================================================

def get_custom_score(m, t):
    d   = math.hypot(m.x - t.x, m.y - t.y)
    msh = t.ships + 1
    spd = fleet_speed(msh)
    eta = d / spd
    eb  = t.production if t.owner != -1 else 0
    ep  = eta * t.production if t.owner != -1 else 0
    return ((FORMULA_DIST - d)
            + FORMULA_PROD_MULT   * t.production
            + FORMULA_ENEMY_BONUS * eb
            - FORMULA_SHIPS_PERCENT * (msh + ep)
            - 2 * eta)


# ============================================================
# SECTION 6 — Trajectory & State Refresh Helpers
# ============================================================

def fill_moving_planets(obs):
    planets  = [ow.Planet(*p) for p in _read(obs, "planets", [])]
    init_map = {i[0]: ow.Planet(*i) for i in _read(obs, "initial_planets", [])}
    for p in planets:
        ip = init_map.get(p.id)
        if ip and (p.x, p.y) != (ip.x, ip.y) and p.id not in moving_planets:
            moving_planets.append(p.id)


def refresh_lobs(obs):
    planets = [ow.Planet(*p) for p in _read(obs, "planets", [])]
    player  = _read(obs, "player", -2)
    fleets  = [ow.Fleet(*f)  for f in _read(obs, "fleets", [])]
    return {
        "planets": planets,
        "mine":    [p for p in planets if p.owner == player],
        "targets": [p for p in planets if p.owner != player],
        "player":  player,
        "fleets":  fleets,
    }


def update_fleet_trajectories(fleets):
    for ft in fleet_trajectories[:]:
        found = any(f.from_planet_id == ft["mine"].id
                    and abs(f.angle - ft["angle"]) < 1e-3
                    for f in fleets)
        if found:
            ft["arrive_tick"] = max(0, ft["arrive_tick"] - 1)
        else:
            fleet_trajectories.remove(ft)


def update_reinf_trajectories():
    for rt in reinforcement_trajectories[:]:
        rt["arrive_tick"] -= 1
        if rt["arrive_tick"] <= 0:
            reinforcement_trajectories.remove(rt)


# ============================================================
# SECTION 7 — Threat Detection
# ============================================================

def get_under_attack(mine, fleets, player, vel):
    mov_traj = {m.id: get_planet_trajectories(m, vel)
                for m in mine if m.id in moving_planets}
    under = {}
    seen  = set()
    for f in fleets:
        if f.owner == player: continue
        spd   = fleet_speed(f.ships)
        px, py = f.x, f.y
        for tick in range(1, 61):
            nx = f.x + math.cos(f.angle) * spd * tick
            ny = f.y + math.sin(f.angle) * spd * tick
            for m in mine:
                mx, my = (mov_traj[m.id][tick - 1]
                          if m.id in mov_traj else (m.x, m.y))
                if collides_segment(px, py, nx, ny, mx, my, m.radius):
                    if (m.id, f.id) not in seen:
                        under.setdefault(m.id, {"planet": m, "fleets": []})
                        under[m.id]["fleets"].append({"fleet": f, "arrive_tick": tick})
                        seen.add((m.id, f.id))
            px, py = nx, ny
    return under


# ============================================================
# SECTION 8 — Defense Planning
# ============================================================

def get_reinforcement_plans(mine, under_attack):
    plans = {}
    # Sort threatened planets by production, highest first
    threatened = sorted(
        [m for m in mine if m.id in under_attack],
        key=lambda p: p.production,
        reverse=True,
    )
    for p in threatened:
        att_fleets = sorted(under_attack[p.id]["fleets"],
                            key=lambda a: a["arrive_tick"])
        incoming   = sorted(
            [r for r in reinforcement_trajectories if r["target"].id == p.id],
            key=lambda r: r["arrive_tick"],
        )
        avail  = float(p.ships)
        prev   = 0
        ridx   = 0
        for att in att_fleets:
            tick   = att["arrive_tick"]
            avail += (tick - prev) * p.production
            while ridx < len(incoming) and incoming[ridx]["arrive_tick"] <= tick:
                avail += incoming[ridx]["total_ships"]
                ridx  += 1
            avail -= att["fleet"].ships
            prev   = tick
            if avail < 0:
                needed = max(MIN_SHIPS_ATTACK, int(abs(avail)))
                plans[p] = {
                    "ships_needed":  needed,
                    "needed_by_tick": tick,
                    "production":    p.production,
                }
                break
    return plans


# ============================================================
# SECTION 9 — Required Ships Calculators (from Proto, unchanged)
# ============================================================

def calculate_req_ships(attacking_planets, t, base_ships):
    required = base_ships
    for _ in range(3):
        remainder = required
        max_tick  = 0
        for a_p in attacking_planets:
            p      = a_p["planet"]
            p_ships = min(a_p["ships"], remainder)
            if p_ships > 0:
                p_ships = min(a_p["ships"], max(p_ships, MIN_SHIPS_ATTACK))
            if p_ships <= 0: continue
            spd  = fleet_speed(p_ships)
            d    = math.hypot(p.x - t.x, p.y - t.y)
            tick = math.floor(d / spd)
            if tick > max_tick: max_tick = tick
            remainder -= p_ships
        new_req = base_ships + max_tick * t.production
        if new_req == required: break
        required = new_req
    return required


def calculate_req_ships_moving(attacking_planets, t, base_ships, vel):
    required    = base_ships
    planet_traj = get_planet_trajectories(t, vel)
    for _ in range(3):
        remainder = required
        max_tick  = 0
        for a_p in attacking_planets:
            p      = a_p["planet"]
            p_ships = min(a_p["ships"], remainder)
            if p_ships > 0:
                p_ships = min(a_p["ships"], max(p_ships, MIN_SHIPS_ATTACK))
            if p_ships <= 0: continue
            spd       = fleet_speed(p_ships)
            found_tick = 0
            for tick, (tx, ty) in enumerate(planet_traj, 1):
                d   = math.hypot(p.x - tx, p.y - ty)
                eta = math.floor(d / spd)
                if abs(eta - tick) <= 1:
                    found_tick = tick
                    break
            if found_tick > max_tick: max_tick = found_tick
            remainder -= p_ships
        new_req = base_ships + max_tick * t.production
        if new_req == required: break
        required = new_req
    return required


# ============================================================
# SECTION 10 — Main Agent Implementation
# ============================================================

def _agent_impl(obs):
    global steps, fleet_trajectories, reinforcement_trajectories

    vel   = _read(obs, "angular_velocity", 0.0)
    moves = []

    # ── Startup: detect moving planets ───────────────────────────────────
    if steps < 2:
        steps += 1
        return []
    if steps == 2:
        fill_moving_planets(obs)
        steps = 3

    lobs        = refresh_lobs(obs)
    mine        = lobs["mine"]
    targets     = lobs["targets"]
    all_planets = lobs["planets"]
    fleets      = lobs["fleets"]
    player      = lobs["player"]
    comets      = set(_read(obs, "comet_planet_ids", []) or [])

    update_fleet_trajectories(fleets)
    update_reinf_trajectories()

    if not targets:
        return []

    under_attack = get_under_attack(mine, fleets, player, vel)
    exhausted    = set()
    deadline     = time.perf_counter() + SIM_TIME_BUDGET_S

    # ── DEFENSE PHASE ────────────────────────────────────────────────────
    # Plans sorted by production so best planets defended first (KEY FIX).
    reinf_plans = get_reinforcement_plans(mine, under_attack)
    for p, plan in sorted(reinf_plans.items(),
                          key=lambda kv: kv[1]["production"],
                          reverse=True):
        # Skip if already being reinforced
        if any(r["target"].id == p.id for r in reinforcement_trajectories):
            continue

        needed   = plan["ships_needed"]
        by_tick  = plan["needed_by_tick"]
        # Candidate senders sorted by proximity
        senders  = sorted(
            [m for m in mine if m.id != p.id and m.id not in exhausted],
            key=lambda m: math.hypot(m.x - p.x, m.y - p.y),
        )

        for sender in senders:
            # Available ships after accounting for reserved reinforcements
            avail = sender.ships - sum(
                r["total_ships"] for r in reinforcement_trajectories
                if r["mine"].id == sender.id
            )
            if sender.id in under_attack:
                avail -= sum(a["fleet"].ships
                             for a in under_attack[sender.id]["fleets"])
            avail = max(0, avail)

            to_send = max(MIN_SHIPS_ATTACK, needed)
            if avail < to_send:
                continue

            # Compute intercept angle & tick
            if p.id in moving_planets:
                ang, tick = find_intercept(sender, p, to_send, vel)
            else:
                ang  = math.atan2(p.y - sender.y, p.x - sender.x)
                spd  = fleet_speed(to_send)
                tick = math.floor(math.hypot(p.x - sender.x, p.y - sender.y) / spd)

            # KEY FIX: if fleet arrives too late and planet is valuable,
            # try a smaller (faster) fleet to still arrive before attacker.
            if ang is None or tick is None or tick > by_tick:
                if p.production >= 3:
                    fast_ships = max(MIN_SHIPS_ATTACK, needed // 2)
                    if avail < fast_ships:
                        continue
                    if p.id in moving_planets:
                        ang, tick = find_intercept(sender, p, fast_ships, vel)
                    else:
                        ang  = math.atan2(p.y - sender.y, p.x - sender.x)
                        spd2 = fleet_speed(fast_ships)
                        tick = math.floor(math.hypot(p.x - sender.x, p.y - sender.y) / spd2)
                    if tick is None or tick > by_tick:
                        continue
                    to_send = fast_ships
                else:
                    continue

            if ang is None:
                continue

            moves.append([sender.id, ang, to_send])
            exhausted.add(sender.id)
            reinforcement_trajectories.append({
                "mine":        sender,
                "target":      p,
                "angle":       ang,
                "total_ships": to_send,
                "arrive_tick": tick,
            })
            break

    # ── ATTACK PHASE ─────────────────────────────────────────────────────
    # Track committed moves this turn for multi-source coordination:
    # when planet A launches, planet B's sim sees A's fleet already in flight
    # so B doesn't redundantly target the same planet.
    committed_this_turn = []  # list of (src_planet, angle, ships)

    for m in sorted(mine, key=lambda p: p.ships, reverse=True):
        if m.id in exhausted or m.ships < MIN_SHIPS_ATTACK:
            continue

        # Build heuristic-scored candidate list
        cands = []
        for t in targets:
            score = get_custom_score(m, t)
            if t.id in comets:
                score -= 40           # strong comet penalty
            cands.append((m, t, score))
        cands.sort(key=lambda x: x[2], reverse=True)

        # ── Sim tie-breaker on top-K candidates ──────────────────────────
        now = time.perf_counter()
        if now < deadline and len(cands) > 1:
            # Build adjusted state: subtract ships already committed this turn
            # and add synthetic in-flight fleets for the sim.
            commit_sub   = {}
            synth_fleets = []
            _fid = 9_000_000
            for c_src, c_ang, c_sh in committed_this_turn:
                commit_sub[c_src.id] = commit_sub.get(c_src.id, 0) + c_sh
                ca, sa = math.cos(c_ang), math.sin(c_ang)
                synth_fleets.append(ow.Fleet(
                    _fid, player,
                    c_src.x + ca*(c_src.radius + 0.1),
                    c_src.y + sa*(c_src.radius + 0.1),
                    c_ang, c_src.id, c_sh,
                ))
                _fid += 1

            adj_planets = [
                p._replace(ships=max(0, p.ships - commit_sub[p.id]))
                if p.id in commit_sub else p
                for p in all_planets
            ]
            adj_fleets = fleets + synth_fleets

            simmed = []
            for cand_m, cand_t, cand_h in cands[:SIM_TOP_K]:
                if time.perf_counter() >= deadline:
                    break
                sh  = min(int(cand_m.ships),
                          int(cand_t.ships) + int(cand_t.production)*3 + 5)
                sh  = max(MIN_SHIPS_ATTACK, sh)
                ang = math.atan2(cand_t.y - cand_m.y, cand_t.x - cand_m.x)
                sc  = simulate_outcome(adj_planets, adj_fleets, player,
                                       cand_m, ang, sh)
                simmed.append((cand_m, cand_t, sc, cand_h))

            if simmed:
                # Primary sort: sim score. Tie-break: heuristic score.
                simmed.sort(key=lambda x: (x[2], x[3]), reverse=True)
                cands = [(cm, ct, h) for cm, ct, _s, h in simmed] + cands[SIM_TOP_K:]

        # ── Execute best viable candidate ────────────────────────────────
        owned_count = len(mine)
        total_count = len(all_planets)

        for m, t, _ in cands[:3]:
            m_avail = m.ships
            if m.id in under_attack:
                m_avail -= sum(a["fleet"].ships
                               for a in under_attack[m.id]["fleets"])
            m_avail = max(0, m_avail)
            if m_avail < MIN_SHIPS_ATTACK:
                continue

            # Ships already heading to this target
            en_route = sum(f["total_ships"] for f in fleet_trajectories
                           if f["target"].id == t.id)

            needed_now = t.ships + 1 + (3 * t.production if t.owner != -1 else 0)

            # Sufficiency check: skip if already enough en route,
            # unless we own almost everything (endgame pile-on).
            if owned_count < total_count * 0.75 and en_route >= needed_now:
                continue

            base        = max(MIN_SHIPS_ATTACK, needed_now - en_route)
            angle       = None
            arrive_tick = None
            total_ships = base

            # ── Single-planet attack ──────────────────────────────────────
            if m_avail >= base:
                if t.id in moving_planets:
                    # Iteratively refine ship count for moving owned targets
                    for _ in range(3):
                        angle, arrive_tick = find_intercept(m, t, total_ships, vel)
                        if angle is None:
                            break
                        new = base + (arrive_tick * t.production if t.owner != -1 else 0)
                        if new > m_avail:
                            angle = None
                            break
                        if new == total_ships:
                            break
                        total_ships = new
                else:
                    # Static target
                    angle = angle_to(m, t)
                    d     = math.hypot(t.x - m.x, t.y - m.y)
                    for _ in range(3):
                        spd  = fleet_speed(total_ships)
                        tick = math.floor(d / spd)
                        new  = base + (tick * t.production if t.owner != -1 else 0)
                        if new > m_avail:
                            angle       = None
                            arrive_tick = None
                            break
                        arrive_tick = tick
                        if new == total_ships:
                            break
                        total_ships = new

                if angle is not None and arrive_tick is not None:
                    spd = fleet_speed(total_ships)
                    if sun_collision(m, spd, angle):
                        continue
                    moves.append([m.id, angle, total_ships])
                    exhausted.add(m.id)
                    committed_this_turn.append((m, angle, total_ships))
                    fleet_trajectories.append({
                        "mine":        m,
                        "target":      t,
                        "angle":       angle,
                        "total_ships": total_ships,
                        "arrive_tick": arrive_tick,
                    })
                    break   # move assigned; move to next planet

            # ── Cooperative attack ────────────────────────────────────────
            elif (m_avail < base
                  and len(mine) > 1
                  and t.ships >= MIN_SHIPS_COOP_TARGET):

                # Build list of safe nearby planets
                safe_nearby = []
                for p, dist in get_closest_planets_to_target(mine, t):
                    if p.id == m.id or p.id in exhausted:
                        continue
                    avail = p.ships
                    if p.id in under_attack:
                        avail -= sum(a["fleet"].ships
                                     for a in under_attack[p.id]["fleets"])
                    avail = max(0, avail)
                    if avail >= MIN_SHIPS_ATTACK:
                        safe_nearby.append((p, dist, avail))

                accum            = m_avail
                attacking_planets = [{"planet": m, "ships": m_avail}]
                coop_sent        = False

                for p, dist, p_avail in safe_nearby:
                    if coop_sent:
                        break
                    attacking_planets.append({"planet": p, "ships": p_avail})
                    accum += p_avail
                    if len(attacking_planets) > COOP_PLANET_CAP:
                        break
                    if accum < base:
                        continue

                    # ── Coop: static target ───────────────────────────────
                    if t.id not in moving_planets:
                        if t.owner == -1:
                            # Unowned static: share base_ships among attackers
                            remainder = base
                            planned   = []
                            for a_p in attacking_planets:
                                pp      = a_p["planet"]
                                p_ships = min(a_p["ships"], remainder)
                                if p_ships > 0:
                                    p_ships = min(a_p["ships"],
                                                  max(p_ships, MIN_SHIPS_ATTACK))
                                if p_ships <= 0: continue
                                ang  = angle_to(pp, t)
                                d    = math.hypot(pp.x - t.x, pp.y - t.y)
                                spd  = fleet_speed(p_ships)
                                tick = math.floor(d / spd)
                                if sun_collision(pp, spd, ang): break
                                remainder -= p_ships
                                planned.append([pp, ang, p_ships, tick])
                            if remainder > 0: continue
                            for mv in planned:
                                fleet_trajectories.append({
                                    "mine": mv[0], "target": t,
                                    "angle": mv[1], "total_ships": mv[2],
                                    "arrive_tick": mv[3],
                                })
                                exhausted.add(mv[0].id)
                                mv[0] = mv[0].id
                                moves.append(mv)
                            coop_sent = True

                        else:
                            # Owned static: account for production during transit
                            req_ships = calculate_req_ships(attacking_planets, t, base)
                            if accum < req_ships: continue
                            remainder = req_ships
                            planned   = []
                            for a_p in attacking_planets:
                                pp      = a_p["planet"]
                                p_ships = min(a_p["ships"], remainder)
                                if p_ships > 0:
                                    p_ships = min(a_p["ships"],
                                                  max(p_ships, MIN_SHIPS_ATTACK))
                                if p_ships <= 0: continue
                                ang  = angle_to(pp, t)
                                d    = math.hypot(pp.x - t.x, pp.y - t.y)
                                spd  = fleet_speed(p_ships)
                                tick = math.floor(d / spd)
                                if sun_collision(pp, spd, ang): continue
                                remainder -= p_ships
                                planned.append([pp, ang, p_ships, tick])
                            if remainder > 0: continue
                            for mv in planned:
                                fleet_trajectories.append({
                                    "mine": mv[0], "target": t,
                                    "angle": mv[1], "total_ships": mv[2],
                                    "arrive_tick": mv[3],
                                })
                                exhausted.add(mv[0].id)
                                mv[0] = mv[0].id
                                moves.append(mv)
                            coop_sent = True

                    # ── Coop: moving target ───────────────────────────────
                    else:
                        if t.owner == -1:
                            # Unowned moving
                            remainder = base
                            planned   = []
                            for a_p in attacking_planets:
                                pp      = a_p["planet"]
                                p_ships = min(a_p["ships"], remainder)
                                if p_ships > 0:
                                    p_ships = min(a_p["ships"],
                                                  max(p_ships, MIN_SHIPS_ATTACK))
                                if p_ships <= 0: continue
                                ang, tick = find_intercept(pp, t, p_ships, vel)
                                if ang is None or tick is None: continue
                                planned.append([pp, ang, p_ships, tick])
                                remainder -= p_ships
                            if remainder > 0: continue
                            for mv in planned:
                                fleet_trajectories.append({
                                    "mine": mv[0], "target": t,
                                    "angle": mv[1], "total_ships": mv[2],
                                    "arrive_tick": mv[3],
                                })
                                exhausted.add(mv[0].id)
                                mv[0] = mv[0].id
                                moves.append(mv)
                            coop_sent = True

                        else:
                            # Owned moving
                            req_ships = calculate_req_ships_moving(
                                attacking_planets, t, base, vel)
                            if accum < req_ships: continue
                            remainder = req_ships
                            planned   = []
                            for a_p in attacking_planets:
                                pp      = a_p["planet"]
                                p_ships = min(a_p["ships"], remainder)
                                if p_ships > 0:
                                    p_ships = min(a_p["ships"],
                                                  max(p_ships, MIN_SHIPS_ATTACK))
                                if p_ships <= 0: continue
                                ang, tick = find_intercept(pp, t, p_ships, vel)
                                if ang is None or tick is None: continue
                                remainder -= p_ships
                                planned.append([pp, ang, p_ships, tick])
                            if remainder > 0: continue
                            for mv in planned:
                                fleet_trajectories.append({
                                    "mine": mv[0], "target": t,
                                    "angle": mv[1], "total_ships": mv[2],
                                    "arrive_tick": mv[3],
                                })
                                exhausted.add(mv[0].id)
                                mv[0] = mv[0].id
                                moves.append(mv)
                            coop_sent = True

                if coop_sent:
                    break   # move assigned for this planet; advance to next

    return moves


# ============================================================
# SECTION 11 — Safety Wrapper
# Resets all module state when a new game is detected.
# Never raises: returns [] on any exception.
# ============================================================

def _sanitize(moves):
    out = []
    for m in (moves or []):
        try:
            if len(m) == 3 and int(m[2]) > 0:
                out.append([int(m[0]), float(m[1]), int(m[2])])
        except Exception:
            pass
    return out


def agent(obs, config=None):
    global steps, fleet_trajectories, reinforcement_trajectories
    global moving_planets, _game_sig, _last_obs_step
    try:
        player   = _read(obs, "player", 0)
        obs_step = _read(obs, "step",   0) or 0
        raw_init = _read(obs, "initial_planets", []) or []

        try:
            sig_tail = tuple((int(p[0]), int(p[5]), int(p[6]))
                             for p in raw_init[:4])
        except Exception:
            sig_tail = ()

        sig = (player, sig_tail)

        # Detect new game: reset all persistent state
        if sig != _game_sig or obs_step == 0 or obs_step < _last_obs_step:
            steps                      = 0
            fleet_trajectories         = []
            reinforcement_trajectories = []
            moving_planets             = []
            _game_sig                  = sig

        _last_obs_step = obs_step
        return _sanitize(_agent_impl(obs))

    except Exception:
        return []
