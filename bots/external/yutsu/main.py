"""
Orbit Wars submission - v6_agent
Local arena results (fair 2-sided test, 30 games):
  v6_agent vs defense_agent: 80.0%  (24W-0D-6L)
  v6_agent vs v5_agent:      80.0%  (24W-0D-6L)
  v4_agent vs defense_agent: 46.7%  (negligible improvement)
  v5_agent vs defense_agent: 46.7%  (launch-window hurt due to delayed expansion)

Core improvements over defense_agent:
  - Binary search for exact minimum ships to guarantee capture
  - 20-turn forward simulation (enemy noop): pick best of 3 aggression levels
  - Simulation score: (my_ships - enemy_ships)
                    + (my_prod - enemy_prod) * remaining * 0.5
                    + (my_planets - enemy_planets) * 50
"""
import math
import time

CENTER = 50.0
SUN_R = 10.0
ROTATION_LIMIT = 50.0
MAX_SPEED = 6.0
HORIZON = 500


def _fleet_speed(num_ships):
    if num_ships <= 1:
        return 1.0
    return min(MAX_SPEED,
               1.0 + (MAX_SPEED - 1.0) * (math.log(num_ships) / math.log(1000)) ** 1.5)


def _is_rotating(px, py, radius):
    return (math.hypot(px - CENTER, py - CENTER) + radius) < ROTATION_LIMIT


def _find_intercept(src_x, src_y, tx, ty, t_radius, ships, angular_velocity):
    if not _is_rotating(tx, ty, t_radius):
        dist = math.hypot(src_x - tx, src_y - ty)
        return tx, ty, math.ceil(dist / _fleet_speed(ships))

    orbital_r = math.hypot(tx - CENTER, ty - CENTER)
    theta_0 = math.atan2(ty - CENTER, tx - CENTER)
    speed = _fleet_speed(ships)

    t_guess = math.hypot(src_x - tx, src_y - ty) / speed
    px, py = tx, ty
    for _ in range(20):
        theta_t = theta_0 + angular_velocity * t_guess
        px = CENTER + orbital_r * math.cos(theta_t)
        py = CENTER + orbital_r * math.sin(theta_t)
        t_new = math.hypot(src_x - px, src_y - py) / speed
        if abs(t_new - t_guess) < 0.5:
            break
        t_guess = 0.6 * t_guess + 0.4 * t_new
    return px, py, math.ceil(t_guess)


def _find_intercept_at_launch(src_x, src_y, tx, ty, t_radius, ships, angular_velocity, wait_turns):
    """Intercept point for a fleet launched wait_turns from now."""
    if not _is_rotating(tx, ty, t_radius):
        dist = math.hypot(src_x - tx, src_y - ty)
        return tx, ty, math.ceil(dist / _fleet_speed(ships))

    orbital_r = math.hypot(tx - CENTER, ty - CENTER)
    theta_now = math.atan2(ty - CENTER, tx - CENTER)
    theta_launch = theta_now + angular_velocity * wait_turns
    speed = _fleet_speed(ships)

    px = CENTER + orbital_r * math.cos(theta_launch)
    py = CENTER + orbital_r * math.sin(theta_launch)
    t_guess = math.hypot(src_x - px, src_y - py) / speed

    for _ in range(20):
        theta_t = theta_launch + angular_velocity * t_guess
        px = CENTER + orbital_r * math.cos(theta_t)
        py = CENTER + orbital_r * math.sin(theta_t)
        t_new = math.hypot(src_x - px, src_y - py) / speed
        if abs(t_new - t_guess) < 0.5:
            break
        t_guess = 0.6 * t_guess + 0.4 * t_new
    return px, py, math.ceil(t_guess)


def _ray_hits_sun(x1, y1, x2, y2, margin=1.5):
    dx, dy = x2 - x1, y2 - y1
    fx, fy = x1 - CENTER, y1 - CENTER
    a = dx * dx + dy * dy
    if a == 0:
        return False
    b = 2 * (fx * dx + fy * dy)
    c = fx * fx + fy * fy - (SUN_R + margin) ** 2
    disc = b * b - 4 * a * c
    if disc < 0:
        return False
    t1 = (-b - math.sqrt(disc)) / (2 * a)
    t2 = (-b + math.sqrt(disc)) / (2 * a)
    return (0 < t1 < 1) or (0 < t2 < 1)


def _present_value(production, arrival_turn, gamma=0.99):
    return production * (gamma ** arrival_turn - gamma ** HORIZON) / (1.0 - gamma)


def _detect_incoming(raw_fleets, my_planets_raw, player):
    incoming = {p[0]: 0 for p in my_planets_raw}
    earliest = {p[0]: 999 for p in my_planets_raw}
    for f in raw_fleets:
        if f[1] == player:
            continue
        cx, cy = math.cos(f[4]), math.sin(f[4])
        for mp in my_planets_raw:
            rx, ry = mp[2] - f[2], mp[3] - f[3]
            d_along = rx * cx + ry * cy
            if d_along <= 0:
                continue
            d_perp = abs(rx * (-cy) + ry * cx)
            if d_perp < mp[4] + 0.5:
                incoming[mp[0]] += f[6]
                t_arr = max(1, math.ceil(d_along / max(_fleet_speed(f[6]), 0.1)))
                if t_arr < earliest[mp[0]]:
                    earliest[mp[0]] = t_arr
    return incoming, earliest


def _min_ships_for_capture(src_x, src_y, tx, ty, t_radius, t_ships, t_owner, t_prod,
                           angular_velocity, max_ships, wait_turns=0):
    """
    Binary search for minimum ships to guarantee capture, accounting for
    fleet speed → arrival time → garrison growth.
    Returns (n, aim_x, aim_y, flight_turns) or None.
    """
    hi = min(int(max_ships), 500)
    if hi < 20:
        return None

    def _check(n):
        ax, ay, ft = _find_intercept_at_launch(src_x, src_y, tx, ty, t_radius,
                                               n, angular_velocity, wait_turns)
        if _ray_hits_sun(src_x, src_y, ax, ay):
            return None
        total = wait_turns + ft
        garrison = int(t_ships) if t_owner == -1 else int(t_ships + t_prod * total)
        if n >= garrison + 1:
            return (n, ax, ay, ft)
        return False

    r = _check(hi)
    if not r:
        return None
    best = r

    lo = 20
    while lo < hi:
        mid = (lo + hi) // 2
        r = _check(mid)
        if r:
            hi = mid
            best = r
        else:
            lo = mid + 1

    return best


def _simulate_state(raw_planets, raw_fleets, moves, angular_velocity,
                    step_start, n_steps, player, raw_initial):
    """Simulate n_steps turns: apply moves for player, enemy noop."""
    planets = [list(p) for p in raw_planets]
    fleets  = [list(f) for f in raw_fleets]
    init_pos = {ip[0]: (ip[2], ip[3]) for ip in raw_initial}

    next_fid = max((f[0] for f in fleets), default=-1) + 1
    for mid, angle, n_ships in moves:
        p = next((p for p in planets if p[0] == mid), None)
        if p and p[1] == player and int(p[5]) >= int(n_ships) >= 1:
            p[5] = int(p[5]) - int(n_ships)
            sx = p[2] + math.cos(angle) * (p[4] + 0.1)
            sy = p[3] + math.sin(angle) * (p[4] + 0.1)
            fleets.append([next_fid, player, sx, sy, angle, mid, n_ships])
            next_fid += 1

    for sim_t in range(n_steps):
        actual_step = step_start + sim_t + 1

        for p in planets:
            if p[1] != -1:
                p[5] += p[6]

        new_fleets = []
        combat = {}
        for f in fleets:
            speed = _fleet_speed(f[6])
            ox, oy = f[2], f[3]
            nx = ox + math.cos(f[4]) * speed
            ny = oy + math.sin(f[4]) * speed
            if not (0 <= nx <= 100 and 0 <= ny <= 100):
                continue
            if _ray_hits_sun(ox, oy, nx, ny, margin=0.0):
                continue
            hit = False
            for p in planets:
                if math.hypot(nx - p[2], ny - p[3]) < p[4]:
                    combat.setdefault(p[0], []).append(list(f))
                    hit = True
                    break
            if not hit:
                f[2], f[3] = nx, ny
                new_fleets.append(f)
        fleets = new_fleets

        for p in planets:
            ix, iy = init_pos.get(p[0], (p[2], p[3]))
            orb = math.hypot(ix - CENTER, iy - CENTER)
            if orb + p[4] >= ROTATION_LIMIT:
                continue
            a0 = math.atan2(iy - CENTER, ix - CENTER)
            theta = a0 + angular_velocity * actual_step
            p[2] = CENTER + orb * math.cos(theta)
            p[3] = CENTER + orb * math.sin(theta)

        for pid, arriving in combat.items():
            pl = next((p for p in planets if p[0] == pid), None)
            if pl is None:
                continue
            owner_forces = {}
            for f in arriving:
                owner_forces[f[1]] = owner_forces.get(f[1], 0) + f[6]
            sorted_o = sorted(owner_forces.items(), key=lambda x: -x[1])
            top_o, top_s = sorted_o[0]
            sec_s = sorted_o[1][1] if len(sorted_o) > 1 else 0
            if top_s == sec_s:
                winner, surviving = -1, 0
            else:
                winner, surviving = top_o, top_s - sec_s
            if surviving <= 0:
                pass
            elif winner == pl[1]:
                pl[5] += surviving
            else:
                pl[5] -= surviving
                if pl[5] < 0:
                    pl[1] = winner
                    pl[5] = -pl[5]

    return planets, fleets


def _evaluate_sim(planets, fleets, player, remaining_turns):
    my_s  = sum(p[5] for p in planets if p[1] == player) \
          + sum(f[6] for f in fleets  if f[1] == player)
    en_s  = sum(p[5] for p in planets if p[1] not in (-1, player)) \
          + sum(f[6] for f in fleets  if f[1] not in (-1, player))
    my_p  = sum(p[6] for p in planets if p[1] == player)
    en_p  = sum(p[6] for p in planets if p[1] not in (-1, player))
    my_n  = sum(1    for p in planets if p[1] == player)
    en_n  = sum(1    for p in planets if p[1] not in (-1, player))
    return ((my_s - en_s)
            + (my_p - en_p) * remaining_turns * 0.5
            + (my_n - en_n) * 50)


def _build_attack_moves(player, raw_planets, raw_fleets, angular_velocity, step,
                        remaining, reserves, enemy_mult):
    """
    Build up to 3 attack moves using defense-compatible scoring + binary search.
    enemy_mult: 1.0=default, 2.0=aggressive, 0.5=conservative.
    """
    is_early = step < 80
    my_planets = [p for p in raw_planets if p[1] == player]
    targets    = [p for p in raw_planets if p[1] != player]

    candidates = []
    for mine in my_planets:
        est_ships = max(20, mine[5] // 2)
        for t in targets:
            aim_x, aim_y, arrival = _find_intercept(
                mine[2], mine[3], t[2], t[3], t[4], est_ships, angular_velocity)
            if _ray_hits_sun(mine[2], mine[3], aim_x, aim_y):
                continue
            future_garrison = t[5] if t[1] == -1 else t[5] + t[6] * arrival
            ships_needed = max(int(future_garrison) + 1, 20)
            pv = _present_value(t[6], arrival)
            if t[1] not in (-1, player):
                base_mult = 1.45 if is_early else (2.2 if step < 300 else 3.0)
                pv *= base_mult * enemy_mult
            score = pv / (ships_needed + 0.2 * math.hypot(mine[2] - t[2], mine[3] - t[3]) + 1.0)
            candidates.append((score, mine[0], t[0], aim_x, aim_y, ships_needed))

    candidates.sort(key=lambda x: -x[0])
    moves_out = []
    targeted = {}
    rem = dict(remaining)

    for score, src_id, tgt_id, aim_x_est, aim_y_est, ships_needed in candidates:
        if len(moves_out) >= 3:
            break
        avail = rem[src_id] - reserves[src_id]
        if avail < 5:
            continue
        src = next(p for p in my_planets if p[0] == src_id)
        t   = next(p for p in targets    if p[0] == tgt_id)

        result = _min_ships_for_capture(
            src[2], src[3], t[2], t[3], t[4], t[5], t[1], t[6],
            angular_velocity, avail)
        if result is not None:
            n_send, aim_x, aim_y = result[0], result[1], result[2]
        else:
            n_send, aim_x, aim_y = ships_needed, aim_x_est, aim_y_est

        already = targeted.get(tgt_id, 0)
        room = int(n_send * 1.2) + 3 - already
        if room <= 0:
            continue
        send = min(avail, room, n_send + 8)
        if send < 5:
            continue
        angle = math.atan2(aim_y - src[3], aim_x - src[2])
        moves_out.append([src_id, angle, int(send)])
        rem[src_id] -= send
        targeted[tgt_id] = already + send

    return moves_out


def agent(obs, config=None):
    """
    v6_agent: defense + binary-search fleet sizing + 20-turn forward simulation.

    Generates 5 candidate move sets (3 aggression levels + defense-only + noop),
    simulates each 20 turns (enemy noop), picks best by evaluation score.
    Time-boxed to 0.7s.
    """
    t_start = time.time()

    player = obs.get("player", 0) if isinstance(obs, dict) else obs.player
    raw_planets = obs.get("planets", []) if isinstance(obs, dict) else obs.planets
    raw_fleets  = obs.get("fleets",  []) if isinstance(obs, dict) else obs.fleets
    raw_initial = obs.get("initial_planets", []) if isinstance(obs, dict) else obs.initial_planets
    angular_velocity = (obs.get("angular_velocity", 0.0)
                        if isinstance(obs, dict) else obs.angular_velocity)
    step = obs.get("step", 0) if isinstance(obs, dict) else obs.step

    my_planets = [p for p in raw_planets if p[1] == player]
    if not my_planets:
        return []

    remaining_turns = max(1, HORIZON - step)
    incoming, earliest_arrival = _detect_incoming(raw_fleets, my_planets, player)
    reserves = {mp[0]: max(5, incoming[mp[0]] + mp[6] * 2) for mp in my_planets}
    remaining = {p[0]: p[5] for p in my_planets}

    # Defense moves (shared)
    defense_moves = []
    for mp in my_planets:
        t_arr = earliest_arrival[mp[0]]
        if t_arr >= 999 or mp[6] < 2:
            continue
        future_garrison = mp[5] + mp[6] * t_arr
        deficit = incoming[mp[0]] - future_garrison + 1
        if deficit < 10:
            continue
        candidates_src = sorted(
            [p for p in my_planets if p[0] != mp[0]],
            key=lambda p: math.hypot(p[2] - mp[2], p[3] - mp[3])
        )
        for src in candidates_src:
            if _ray_hits_sun(src[2], src[3], mp[2], mp[3]):
                continue
            d = math.hypot(src[2] - mp[2], src[3] - mp[3])
            t_supply = math.ceil(d / _fleet_speed(max(20, deficit + 5)))
            if t_supply > t_arr:
                continue
            avail = remaining[src[0]] - max(5, src[6] * 2)
            if avail < deficit:
                continue
            send = min(avail, int(deficit) + 5)
            angle = math.atan2(mp[3] - src[3], mp[2] - src[2])
            defense_moves.append([src[0], angle, send])
            remaining[src[0]] -= send
            break

    # Attack candidate sets with 3 aggression levels
    atk_default      = _build_attack_moves(player, raw_planets, raw_fleets, angular_velocity,
                                            step, remaining, reserves, 1.0)
    atk_aggressive   = _build_attack_moves(player, raw_planets, raw_fleets, angular_velocity,
                                            step, remaining, reserves, 2.0)
    atk_conservative = _build_attack_moves(player, raw_planets, raw_fleets, angular_velocity,
                                            step, remaining, reserves, 0.5)

    SIM_TURNS = 20
    move_candidates = [
        defense_moves + atk_default,
        defense_moves + atk_aggressive,
        defense_moves + atk_conservative,
        defense_moves,
        [],
    ]

    best_moves = move_candidates[0]
    best_score = -float("inf")

    for cand in move_candidates:
        if time.time() - t_start > 0.7:
            break
        try:
            pl_after, fl_after = _simulate_state(
                raw_planets, raw_fleets, cand,
                angular_velocity, step, SIM_TURNS, player, raw_initial)
            score = _evaluate_sim(pl_after, fl_after, player, remaining_turns - SIM_TURNS)
        except Exception:
            continue
        if score > best_score:
            best_score = score
            best_moves = cand

    return best_moves