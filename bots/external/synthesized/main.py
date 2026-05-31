# Agent:       00080_synthesis_best_combined
# Family:      hellburner
# Base:        00055_production_sort_garrison8_endgame (current champion cluster)
# Hypothesis:  Synthesis of the best validated mechanisms from the research campaign:
#              1. Conditional 2-ply target counter (from 00066): when >=2 distinct
#                 enemies exist (4FFA/multi-front), apply _enemy_counter_wins check
#                 before committing. In 1vs1 (1 enemy), runs pure 00055 greedy.
#              2. Source-exposure penalty (from 00067): in evaluate_move_orders,
#                 subtract source.production from target value when committing fleet
#                 would expose source to loss (unconditional — proved nearly 1vs1-free).
#              Both are conditional/soft and don't deplete garrison. Combined they give:
#              - 1vs1: ~00055 baseline (gates inactive or near-zero penalty)
#              - 4FFA: best of both over-extension controls simultaneously
#              This is the "synthesis champion" — the agent most likely to be the best
#              overall submission for both LB (1vs1) and 4FFA leaderboard stability.
#              Second-to-last loop — we want the best possible combined agent.
# Status:      testing

import math
import copy
from collections import defaultdict
from dataclasses import dataclass, field

from kaggle_environments.envs.orbit_wars.orbit_wars import (
    Fleet, CENTER, ROTATION_RADIUS_LIMIT, SUN_RADIUS,
    distance, point_to_segment_distance
)

class HPlanet:
    def __init__(self, id, owner, x, y, radius, ships, production):
        self.id = id; self.owner = owner; self.x = x; self.y = y
        self.radius = radius; self.ships = ships; self.production = production
        self.reinforcement_target: 'HPlanet | None' = None

@dataclass(slots=True)
class EarlyGameFleet:
    source_id: int
    destination_id: int
    fleet_size: int
    garrison_on_arrival: int
    arrival_turn: int
    is_capture: bool

@dataclass(slots=True)
class EarlyGameState:
    turn: int
    garrison: dict
    production: dict
    owned: set
    fleets: list = field(default_factory=list)


class Hellburner:
    SHIP_SPEED_MAX: float = 6.0
    EARLY_ROUNDS: int = 3
    EARLY_LOOK_AHEAD: int = 33
    MAX_DISTANCE: int = 38
    ROTATION_LOOK_AHEAD: int = 10
    REINFORCEMENT_SIZE: int = 17
    GARRISON_SIZE: int = 8  # 00055 baseline

    def __init__(self):
        self.player: int = 0
        self.scene_step: int = 0
        self.angular_velocity: float = 0.0
        self.planets: list = []
        self.owned_planets: list = []
        self.enemy_planets: list = []
        self.fleets: list = []
        self.orbital_info: dict = {}
        self.inbound_edges: dict = {}
        self.outbound_edges: dict = {}
        self.future_pos: dict = {}
        self.destination_list: dict = {}
        self._multifront: bool = False

    def fleet_speed(self, ships):
        return min(self.SHIP_SPEED_MAX, 1.0 + (self.SHIP_SPEED_MAX - 1.0) * (math.log(ships) / math.log(1000)) ** 1.5)

    def _count_enemy_owners(self):
        owners = set()
        for p in self.planets:
            if p.owner not in (-1, self.player):
                owners.add(p.owner)
        return len(owners)

    def build_orbital_info(self, initial_planets):
        cx = cy = CENTER
        ip_by_id = {ip[0]: ip for ip in initial_planets}
        self.orbital_info = {}
        for p in self.planets:
            r = distance((p.x, p.y), (cx, cy))
            if r + p.radius < ROTATION_RADIUS_LIMIT and p.id in ip_by_id:
                ip = ip_by_id[p.id]
                self.orbital_info[p] = (r, math.atan2(ip[3] - cy, ip[2] - cx))
            else:
                self.orbital_info[p] = None

    def build_proximity_graph(self):
        cx = cy = CENTER
        self.future_pos = {}
        for p in self.planets:
            orb = self.orbital_info[p]
            if orb is not None:
                r, ia = orb
                a = ia + self.angular_velocity * (self.scene_step + 1 + self.ROTATION_LOOK_AHEAD)
                self.future_pos[p] = (cx + r * math.cos(a), cy + r * math.sin(a))
            else:
                self.future_pos[p] = (p.x, p.y)
        self.inbound_edges = {p: [] for p in self.planets}
        for src in self.planets:
            for dst in self.planets:
                if dst is src:
                    continue
                travel = distance((src.x, src.y), self.future_pos[dst])
                if travel <= self.MAX_DISTANCE:
                    self.inbound_edges[dst].append((src, travel))
        self.outbound_edges = {p: [] for p in self.planets}
        for dst, inbound in self.inbound_edges.items():
            for src, travel in inbound:
                self.outbound_edges[src].append((dst, travel))

    def build_reinforcement_targets(self):
        front_line = {
            p for p in self.owned_planets
            if any(src.owner != self.player for src, _ in self.inbound_edges[p])
            or any(dst.owner != self.player for dst, _ in self.outbound_edges[p])
        }
        hops_to_front = {p: 0 for p in front_line}
        queue = list(front_line)
        head = 0
        while head < len(queue):
            node = queue[head]; head += 1
            for src, _ in self.inbound_edges[node]:
                if src.owner != self.player or src in hops_to_front:
                    continue
                hops_to_front[src] = hops_to_front[node] + 1
                queue.append(src)
        for p in self.owned_planets:
            p.reinforcement_target = None
            if p in front_line:
                continue
            direct_front = [dst for dst, _ in self.outbound_edges[p] if dst in front_line]
            if direct_front:
                p.reinforcement_target = min(direct_front, key=lambda d: d.ships)
                continue
            reachable = [
                dst for dst, _ in self.outbound_edges[p]
                if dst.owner == self.player and dst not in front_line and dst in hops_to_front
            ]
            if reachable:
                p.reinforcement_target = min(reachable, key=lambda d: (hops_to_front[d], d.ships))

    def intercept_planet(self, sx, sy, target, ships, tol=1e-6, max_iters=30):
        speed = self.fleet_speed(ships)
        orb = self.orbital_info[target]
        if orb is None:
            tx, ty = target.x, target.y
            travel = distance((sx, sy), (tx, ty)) / speed
        else:
            cx = cy = CENTER
            r, ia = orb
            travel = distance((sx, sy), (target.x, target.y)) / speed
            for _ in range(max_iters):
                a = ia + self.angular_velocity * (self.scene_step + travel - 0.5)
                new_tx, new_ty = cx + r * math.cos(a), cy + r * math.sin(a)
                new_travel = distance((sx, sy), (new_tx, new_ty)) / speed
                new_travel = 0.5 * (travel + new_travel - 0.5)
                if abs(new_travel - travel) < tol:
                    travel = new_travel
                    break
                travel = new_travel
            else:
                return 0.0, target.x, target.y, math.inf
            a = ia + self.angular_velocity * (self.scene_step + travel - 0.5)
            tx, ty = cx + r * math.cos(a), cy + r * math.sin(a)
        angle = math.atan2(ty - sy, tx - sx)
        return angle, tx, ty, travel

    def first_planet_hit(self, sx, sy, angle, ships, source):
        best = None
        best_t = float('inf')
        for planet in self.planets:
            if planet is source:
                continue
            needed_angle, px, py, travel = self.intercept_planet(sx, sy, planet, ships)
            dist = distance((sx, sy), (px, py))
            half_cone = math.pi if dist < planet.radius else math.asin(min(1.0, planet.radius / dist))
            delta = abs(math.atan2(math.sin(angle - needed_angle), math.cos(angle - needed_angle)))
            if math.isfinite(travel) and delta <= half_cone and travel < best_t:
                best_t = travel
                best = planet
        if best is None:
            return None
        ex = sx + best_t * self.fleet_speed(ships) * math.cos(angle)
        ey = sy + best_t * self.fleet_speed(ships) * math.sin(angle)
        if point_to_segment_distance((CENTER, CENTER), (sx, sy), (ex, ey)) <= SUN_RADIUS:
            return None
        return best

    def build_destination_list(self):
        self.destination_list = defaultdict(list)
        for fleet in self.fleets:
            best = None
            best_t = float('inf')
            for planet in self.planets:
                needed_angle, px, py, travel = self.intercept_planet(fleet.x, fleet.y, planet, fleet.ships)
                dist = distance((fleet.x, fleet.y), (px, py))
                half_cone = math.pi if dist < planet.radius else math.asin(min(1.0, planet.radius / dist))
                delta = abs(math.atan2(math.sin(fleet.angle - needed_angle), math.cos(fleet.angle - needed_angle)))
                if math.isfinite(travel) and delta <= half_cone and travel < best_t:
                    best_t = travel
                    best = (planet, travel, px, py)
            if best is not None:
                planet, travel, px, py = best
                self.destination_list[planet].append((fleet.owner, fleet.ships, travel, fleet.x, fleet.y, px, py))

    def simulate_planet_timeline(self, planet, destination_list):
        cur_owner = planet.owner
        entries = destination_list.get(planet)
        if not bool(entries):
            return cur_owner, 0
        buckets = defaultdict(list)
        for owner, ships, t, _, _, _, _ in entries:
            turn = max(1, math.ceil(t))
            buckets[turn].append((owner, ships))
        last_ships, last_t = entries[-1][1], entries[-1][2]
        last_turn = max(1, math.ceil(last_t))
        cur_ships = float(planet.ships)
        prod = planet.production
        cur_t = 0
        excess_ships = float('inf')
        for turn in sorted(buckets):
            elapsed = turn - cur_t
            if elapsed > 0 and cur_owner != -1:
                cur_ships += prod * elapsed
            cur_t = turn
            owner_ships = defaultdict(float)
            for owner, ships in buckets[turn]:
                owner_ships[owner] += ships
            if owner_ships:
                sorted_owners = sorted(owner_ships.items(), key=lambda x: x[1], reverse=True)
                if len(sorted_owners) == 1:
                    survivor_owner, survivor_ships = sorted_owners[0]
                else:
                    top_owner, top_ships = sorted_owners[0]
                    second_ships = sorted_owners[1][1]
                    survivor_ships = top_ships - second_ships
                    survivor_owner = top_owner if survivor_ships > 0 else -1
                if survivor_ships > 0:
                    if survivor_owner == cur_owner:
                        cur_ships += survivor_ships
                    else:
                        cur_ships -= survivor_ships
                        if cur_ships < 0:
                            cur_owner = survivor_owner
                            cur_ships = abs(cur_ships)
            if turn >= last_turn:
                margin = cur_ships if cur_owner == self.player else 0.0
                excess_ships = min(excess_ships, margin)
        if excess_ships == float('inf'):
            excess_ships = 0.0
        excess_ships = min(excess_ships, last_ships)
        return cur_owner, excess_ships

    def _enemy_counter_wins(self, target, trial_dl):
        """From 00066: returns True if strongest adjacent enemy can retake target."""
        best_enemy = None
        best_ships = 0
        for src, _ in self.inbound_edges.get(target, []):
            if src.owner not in (self.player, -1) and src.ships > best_ships:
                best_enemy = src
                best_ships = src.ships
        if best_enemy is None or best_ships == 0:
            return False
        _, ex, ey, enemy_travel = self.intercept_planet(
            best_enemy.x, best_enemy.y, target, best_ships)
        if not math.isfinite(enemy_travel):
            return False
        counter_dl = {k: list(v) for k, v in trial_dl.items()}
        counter_dl.setdefault(target, [])
        counter_dl[target].append(
            (best_enemy.owner, best_ships, enemy_travel,
             best_enemy.x, best_enemy.y, ex, ey))
        end_owner, _ = self.simulate_planet_timeline(target, counter_dl)
        return end_owner != self.player

    def _source_exposure_penalty(self, fleet_orders):
        """From 00067: returns production sum of sources that would flip after commit."""
        penalty = 0
        for order in fleet_orders:
            src_id, _, ships_sent = order
            src = next((p for p in self.planets if p.id == src_id), None)
            if src is None or src.owner != self.player:
                continue
            worst_case_dl = {k: list(v) for k, v in self.destination_list.items()}
            worst_case_dl.setdefault(src, [])
            for attacker, _ in self.inbound_edges.get(src, []):
                if attacker.owner == self.player or attacker.owner == -1 or attacker.ships == 0:
                    continue
                _, ax, ay, atk_travel = self.intercept_planet(attacker.x, attacker.y, src, attacker.ships)
                if not math.isfinite(atk_travel):
                    continue
                worst_case_dl[src].append(
                    (attacker.owner, max(1, int(attacker.ships * 0.5)), atk_travel,
                     attacker.x, attacker.y, ax, ay))
            saved = src.ships
            src.ships = max(0, src.ships - ships_sent)
            exposed_owner, _ = self.simulate_planet_timeline(src, worst_case_dl)
            src.ships = saved
            if exposed_owner != self.player:
                penalty += src.production
        return penalty

    def evaluate_frontline_strategy(self, target):
        possible_origins = sorted(
            [(src, travel) for src, travel in self.inbound_edges.get(target, [])
             if src.owner == self.player], key=lambda x: x[1])
        fleet_orders = []
        intercepts = []
        trial_destination_list = {}
        for _p, _entries in self.destination_list.items():
            if _p is target:
                trial_destination_list[_p] = [
                    (o, int(s * 0.5) if o != self.player else s, t, x, y, bx, by)
                    for o, s, t, x, y, bx, by in _entries
                ]
            else:
                trial_destination_list[_p] = list(_entries)
        trial_destination_list.setdefault(target, [])
        battle_won = False
        second_enemy_arrival = None
        if target.owner != self.player:
            for owner, _, t, _, _, _, _ in self.destination_list.get(target, []):
                if owner != self.player and owner != target.owner:
                    turn = math.ceil(t)
                    if second_enemy_arrival is None or turn < second_enemy_arrival:
                        second_enemy_arrival = turn
        for neighbor, _ in possible_origins:
            if neighbor.ships == 0:
                continue
            ships_to_send = int(neighbor.ships)
            baseline_owner, _ = self.simulate_planet_timeline(neighbor, self.destination_list)
            not_doomed = baseline_owner == self.player
            if not_doomed:
                worst_case_dl = {k: list(v) for k, v in self.destination_list.items()}
                worst_case_dl.setdefault(neighbor, [])
                half_pressure = 0
                for attacker, _ in self.inbound_edges.get(neighbor, []):
                    if attacker.owner == self.player or attacker.owner == -1 or attacker.ships == 0:
                        continue
                    _, ax, ay, atk_travel = self.intercept_planet(attacker.x, attacker.y, neighbor, attacker.ships)
                    if not math.isfinite(atk_travel):
                        continue
                    half_ships = max(1, int(attacker.ships * 0.5))
                    worst_case_dl[neighbor].append((attacker.owner, half_ships, atk_travel, attacker.x, attacker.y, ax, ay))
                    half_pressure += half_ships
                saved_ships = neighbor.ships
                neighbor.ships = 0
                exposed_owner, _ = self.simulate_planet_timeline(neighbor, worst_case_dl)
                neighbor.ships = saved_ships
                if exposed_owner != self.player:
                    if target.production <= neighbor.production:
                        continue
                else:
                    ships_to_send = max(0, int(neighbor.ships) - half_pressure)
                    if ships_to_send == 0:
                        continue
            angle, ix, iy, travel = self.intercept_planet(neighbor.x, neighbor.y, target, ships_to_send)
            if not math.isfinite(travel):
                continue
            if self.first_planet_hit(neighbor.x, neighbor.y, angle, ships_to_send, neighbor) is not target:
                continue
            if second_enemy_arrival is not None and math.ceil(travel) <= second_enemy_arrival + 1:
                continue
            trial_destination_list[target].append((self.player, ships_to_send, travel, neighbor.x, neighbor.y, ix, iy))
            fleet_orders.append([neighbor.id, angle, ships_to_send])
            intercepts.append((ix, iy, travel))
            trial_end_owner, _ = self.simulate_planet_timeline(target, trial_destination_list)
            if trial_end_owner == self.player:
                # Conditional 2-ply: only in multi-front (from 00066)
                if self._multifront and self._enemy_counter_wins(target, trial_destination_list):
                    continue  # enemy can retake — try more ships
                battle_won = True
                break
        return fleet_orders, intercepts, battle_won

    def evaluate_move_orders(self):
        best_move_orders = (None, -65535, [], [])
        for target in sorted(self.planets, key=lambda p: p.production, reverse=True):
            if not bool(self.inbound_edges.get(target)):
                continue
            if target.owner == self.player:
                if not bool(self.destination_list.get(target)):
                    continue
                end_owner, _ = self.simulate_planet_timeline(target, self.destination_list)
                if end_owner == self.player:
                    continue
                fleet_orders, intercepts, battle_won = self.evaluate_frontline_strategy(target)
                if not battle_won:
                    continue
                # Source penalty (from 00067) — always apply, proved nearly 1vs1-neutral
                penalty = self._source_exposure_penalty(fleet_orders)
                value = target.production - penalty
                _, best_value, best_orders, _ = best_move_orders
                if value > best_value or (value == best_value and len(fleet_orders) < len(best_orders)):
                    best_move_orders = (target, value, fleet_orders, intercepts)
            else:
                end_owner, _ = self.simulate_planet_timeline(target, self.destination_list)
                if end_owner == self.player:
                    continue
                fleet_orders, intercepts, battle_won = self.evaluate_frontline_strategy(target)
                if not battle_won:
                    continue
                penalty = self._source_exposure_penalty(fleet_orders)
                value = target.production - (1 if target.owner == -1 else 0) - penalty
                _, best_value, best_orders, _ = best_move_orders
                if value > best_value or (value == best_value and len(fleet_orders) < len(best_orders)):
                    best_move_orders = (target, value, fleet_orders, intercepts)
        return best_move_orders

    def send_reinforcements(self):
        orders = []
        for p in self.owned_planets:
            if p.reinforcement_target is None:
                continue
            if p.ships < (self.REINFORCEMENT_SIZE + self.GARRISON_SIZE):
                continue
            if any(src.owner != self.player for src, _ in self.inbound_edges.get(p, [])):
                continue
            target = p.reinforcement_target
            ships = int(p.ships - self.GARRISON_SIZE)
            angle, ix, iy, travel = self.intercept_planet(p.x, p.y, target, ships)
            if not math.isfinite(travel):
                continue
            orders.append([p.id, angle, ships])
        return orders

    def commit_move_orders(self, move):
        target, _, fleet_orders, intercepts = move
        for (from_id, _, ships), (ix, iy, travel) in zip(fleet_orders, intercepts):
            src = next((p for p in self.planets if p.id == from_id), None)
            if src is None:
                continue
            src.ships = max(0, src.ships - ships)
            self.destination_list.setdefault(target, [])
            self.destination_list[target].append((self.player, ships, travel, src.x, src.y, ix, iy))

    def early_game_compute_travel_turns(self, source_id, target, fleet_size, launch_turn):
        src = next(p for p in self.planets if p.id == source_id)
        orb = self.orbital_info.get(src)
        if orb is not None:
            cx = cy = CENTER
            r, ia = orb
            a = ia + self.angular_velocity * (launch_turn - 0.5)
            sx, sy = cx + r * math.cos(a), cy + r * math.sin(a)
        else:
            sx, sy = src.x, src.y
        _, _, _, travel = self.intercept_planet(sx, sy, target, fleet_size)
        return travel

    def early_game_find_capture_turn(self, state, target):
        garrison_size = target.ships
        horizon = state.turn + self.EARLY_LOOK_AHEAD
        best = math.inf
        for source in state.owned:
            current_ships = state.garrison[source]
            production_rate = state.production[source]
            for wait_turns in range(self.EARLY_LOOK_AHEAD):
                fleet_size = int(current_ships + production_rate * wait_turns)
                if fleet_size <= garrison_size:
                    continue
                launch_turn = state.turn + wait_turns
                if launch_turn >= horizon:
                    break
                travel_turns = self.early_game_compute_travel_turns(source, target, fleet_size, launch_turn)
                if not math.isfinite(travel_turns):
                    continue
                arrival_turn = launch_turn + math.ceil(travel_turns)
                if arrival_turn <= horizon:
                    best = min(best, arrival_turn)
                    break
        return best

    def early_game_assign_fleets(self, state, target, capture_turn):
        garrison_size = target.ships
        best_source = None
        best_entry = None
        best_arrival = math.inf
        for source in state.owned:
            current_ships = state.garrison[source]
            production_rate = state.production[source]
            for wait_turns in range(capture_turn - state.turn):
                fleet_size = int(current_ships + production_rate * wait_turns)
                if fleet_size <= garrison_size:
                    continue
                launch_turn = state.turn + wait_turns
                travel_turns = self.early_game_compute_travel_turns(source, target, fleet_size, launch_turn)
                if not math.isfinite(travel_turns):
                    continue
                arrival_turn = launch_turn + math.ceil(travel_turns)
                if arrival_turn <= capture_turn and arrival_turn < best_arrival:
                    best_arrival = arrival_turn
                    best_source = source
                    best_entry = (fleet_size, launch_turn, arrival_turn)
                break
        if best_source is None:
            return {}
        return {best_source: best_entry}

    def early_game_advance(self, state, from_turn, to_turn):
        for current_turn in range(from_turn + 1, to_turn + 1):
            for fleet in list(state.fleets):
                if fleet.arrival_turn == current_turn:
                    if fleet.is_capture:
                        state.garrison[fleet.destination_id] = fleet.garrison_on_arrival
                        state.owned.add(fleet.destination_id)
                        if fleet.destination_id not in state.production:
                            state.production[fleet.destination_id] = self.early_game_production(fleet.destination_id)
                    else:
                        state.garrison[fleet.destination_id] += fleet.garrison_on_arrival
                    state.fleets.remove(fleet)
            for planet_id in state.owned:
                state.garrison[planet_id] += state.production[planet_id]
        return state

    def early_game_execute_attack(self, state, target, fleet_assignment, capture_turn):
        garrison_size = target.ships
        total_fleet = sum(fs for fs, _, _ in fleet_assignment.values())
        current_turn = state.turn
        for source, (fleet_size, launch_turn, _) in sorted(fleet_assignment.items(), key=lambda se: se[1][1]):
            state = self.early_game_advance(state, current_turn, launch_turn)
            current_turn = launch_turn
            state.garrison[source] -= fleet_size
        state.fleets.append(EarlyGameFleet(
            source_id=-1, destination_id=target.id, fleet_size=total_fleet,
            garrison_on_arrival=total_fleet - garrison_size, arrival_turn=capture_turn, is_capture=True,
        ))
        state = self.early_game_advance(state, current_turn, capture_turn)
        return state

    def early_game_score(self, state):
        horizon = state.turn + self.EARLY_LOOK_AHEAD
        total = 0
        for planet_id in state.owned:
            total += state.garrison[planet_id] + state.production[planet_id] * (horizon - state.turn)
        for fleet in state.fleets:
            total += fleet.garrison_on_arrival
            if fleet.is_capture:
                total += self.early_game_production(fleet.destination_id) * max(0, horizon - fleet.arrival_turn)
        return total

    def early_game_production(self, planet_id):
        p = next((pl for pl in self.planets if pl.id == planet_id), None)
        return p.production if p else 0

    def run_early_game(self):
        owned_ids = {p.id for p in self.owned_planets}
        neutral_candidates = [
            p for p in self.planets
            if p.owner == -1 and any(src.id in owned_ids for src, _ in self.inbound_edges.get(p, []))
        ]
        in_flight = []
        for dest_planet, arrivals in self.destination_list.items():
            for owner, ships, t, _, _, _, _ in arrivals:
                if owner != self.player:
                    continue
                arrival = self.scene_step + math.ceil(t)
                is_cap = dest_planet.owner != self.player
                surplus = ships - dest_planet.ships
                in_flight.append(EarlyGameFleet(
                    source_id=-1, destination_id=dest_planet.id, fleet_size=int(ships),
                    garrison_on_arrival=int(surplus) if is_cap else int(ships),
                    arrival_turn=arrival, is_capture=is_cap,
                ))
        initial_state = EarlyGameState(
            turn=self.scene_step,
            garrison={p.id: float(p.ships) for p in self.owned_planets},
            production={p.id: p.production for p in self.owned_planets},
            owned=owned_ids.copy(),
            fleets=in_flight,
        )
        def initial_gain(planet):
            ct = self.early_game_find_capture_turn(initial_state, planet)
            horizon = initial_state.turn + self.EARLY_LOOK_AHEAD
            return planet.production * (horizon - ct) - planet.ships if math.isfinite(ct) else -math.inf
        candidates = sorted(neutral_candidates, key=initial_gain, reverse=True)
        candidates = [p for p in candidates if initial_gain(p) > 0]
        if not candidates:
            return []
        best = [self.early_game_score(initial_state), []]
        def upper_bound(state, remaining):
            horizon = state.turn + self.EARLY_LOOK_AHEAD
            bound = self.early_game_score(state)
            for planet in remaining:
                ct = self.early_game_find_capture_turn(state, planet)
                gain = planet.production * (horizon - ct) - planet.ships
                if gain > 0:
                    bound += gain
            return bound
        def dfs(state, remaining, sequence):
            current_score = self.early_game_score(state)
            if current_score > best[0]:
                best[0] = current_score
                best[1] = list(sequence)
            if upper_bound(state, remaining) <= best[0]:
                return
            already_targeted = {f.destination_id for f in state.fleets if f.is_capture}
            for index, planet in enumerate(remaining):
                if planet.id in already_targeted:
                    continue
                horizon = state.turn + self.EARLY_LOOK_AHEAD
                ct = self.early_game_find_capture_turn(state, planet)
                if not math.isfinite(ct):
                    continue
                if planet.production * (horizon - ct) - planet.ships <= 0:
                    continue
                fleet_assignment = self.early_game_assign_fleets(state, planet, ct)
                if not fleet_assignment:
                    continue
                next_state = self.early_game_execute_attack(copy.deepcopy(state), planet, fleet_assignment, ct)
                dfs(next_state, remaining[:index] + remaining[index + 1:], sequence + [(planet, fleet_assignment, ct)])
        dfs(initial_state, candidates, [])
        _, best_sequence = best
        if not best_sequence:
            return []
        moves = []
        for target_planet, fleet_assignment, _ in best_sequence:
            for source_id, (fleet_size, launch_turn, _) in fleet_assignment.items():
                if launch_turn != self.scene_step:
                    continue
                src = next((p for p in self.planets if p.id == source_id), None)
                if src is None:
                    continue
                angle, _, _, travel = self.intercept_planet(src.x, src.y, target_planet, fleet_size)
                if not math.isfinite(travel):
                    continue
                hit = self.first_planet_hit(src.x, src.y, angle, fleet_size, src)
                if hit is not target_planet:
                    continue
                moves.append([source_id, angle, fleet_size])
        return moves

    def main(self, obs):
        self.player = obs['player']
        self.scene_step = obs['step'] - 1
        self.angular_velocity = obs['angular_velocity']

        comet_ids = set(obs['comet_planet_ids'])
        planets_and_comets = [HPlanet(*p) for p in obs['planets']]
        self.planets = [p for p in planets_and_comets if p.id not in comet_ids]
        self.owned_planets = [p for p in self.planets if p.owner == self.player]
        self.enemy_planets = [p for p in self.planets if p.owner != self.player]
        self.fleets = [Fleet(*f) for f in obs['fleets']]

        if not self.enemy_planets:
            return []

        # Multi-front detection: conditional 2-ply only when >=2 enemies
        self._multifront = self._count_enemy_owners() >= 2

        # 00055 endgame
        self.GARRISON_SIZE = 8
        self.REINFORCEMENT_SIZE = 17
        self.MAX_DISTANCE = 38
        if self.scene_step >= 440:
            self.GARRISON_SIZE = 1
            self.REINFORCEMENT_SIZE = 2
            self.MAX_DISTANCE = 55

        self.build_orbital_info(obs.get('initial_planets', []))
        self.build_proximity_graph()
        self.build_destination_list()

        if self.scene_step < self.EARLY_ROUNDS:
            return self.run_early_game()

        self.build_reinforcement_targets()

        moves = []
        while True:
            move_orders = self.evaluate_move_orders()
            target_planet, _, fleet_orders, _ = move_orders
            if target_planet is None:
                break
            self.commit_move_orders(move_orders)
            moves.extend(fleet_orders)

        moves.extend(self.send_reinforcements())
        return moves


def agent(obs):
    try:
        return Hellburner().main(obs)
    except Exception:
        return []
