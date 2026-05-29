import copy
import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from kaggle_environments.envs.orbit_wars.orbit_wars import (
    CENTER,
    ROTATION_RADIUS_LIMIT,
    SUN_RADIUS,
    Fleet,
    distance,
    point_to_segment_distance,
)


class BattlePlanet:
    """Small wrapper around the environment planet tuple with strategy metadata."""

    def __init__(self, planet_id, controller, coord_x, coord_y, planet_radius, ship_count, prod_rate):
        self.planet_id = planet_id
        self.controller = controller
        self.coord_x = coord_x
        self.coord_y = coord_y
        self.planet_radius = planet_radius
        self.ship_count = ship_count
        self.prod_rate = prod_rate
        self.supply_target: "BattlePlanet | None" = None  # Next friendly waypoint toward pressure.


# BattlePlanet -> (orbit_radius, starting_angle) when the planet rotates around the sun.
OrbitMap = dict[BattlePlanet, tuple[float, float] | None]
# BattlePlanet -> coordinates projected a few turns ahead for reachability checks.
ProjectedCoords = dict[BattlePlanet, tuple[float, float]]
# destination -> [(origin, trip_length)] for directed travel opportunities.
RangeGraph = dict[BattlePlanet, list[tuple[BattlePlanet, float]]]
# BattlePlanet -> [(controller, ships, travel_time, start_x, start_y, impact_x, impact_y)].
ArrivalLedger = dict[BattlePlanet, list[tuple[int, float, float, float, float, float, float]]]
# [source_planet_id, launch_angle, ship_count] entries returned to the environment.
CommandList = list[list]
# (impact_x, impact_y, travel_turns) cached beside a command entry.
InterceptRecord = tuple[float, float, float]
# (target planet, score, launch commands, cached intercept records).
StrategyPick = tuple[BattlePlanet | None, int, CommandList, list[InterceptRecord]]


@dataclass(slots=True)
class OpeningFleet:
    origin_id: int
    target_id: int
    launched_ships: int
    landing_garrison: int
    landing_turn: int
    captures_planet: bool


@dataclass(slots=True)
class OpeningState:
    current_turn: int
    stationed_ships: dict
    production_rates: dict
    friendly_ids: set
    active_fleets: list = field(default_factory=list)


class OrbitCommander:
    SHIP_SPEED_CAP: float = 6.0
    OPENING_TURNS: int = 3
    OPENING_HORIZON: int = 33
    MAX_ROUTE_DISTANCE: int = 38
    ORBIT_LOOKAHEAD_TURNS: int = 10
    SUPPLY_BATCH_SIZE: int = 17
    HOME_GARRISON_SIZE: int = 11

    def __init__(self):
        self.my_player_id: int = 0
        self.turn_index: int = 0
        self.orbit_angular_velocity: float = 0.0
        self.all_planets: list[BattlePlanet] = []
        self.friendly_planets: list[BattlePlanet] = []
        self.hostile_planets: list[BattlePlanet] = []
        self.visible_fleets: list[Fleet] = []
        self.planet_orbits: OrbitMap = {}
        self.incoming_routes: RangeGraph = {}
        self.outgoing_routes: RangeGraph = {}
        self.projected_coords: ProjectedCoords = {}
        self.arrival_ledger: ArrivalLedger = {}

    def compute_fleet_speed(self, launched_ships: int | float) -> float:
        # Larger fleets accelerate faster, but the rules clamp speed at the cap.
        return min(
            self.SHIP_SPEED_CAP,
            1.0 + (self.SHIP_SPEED_CAP - 1.0) * (math.log(launched_ships) / math.log(1000)) ** 1.5,
        )

    def index_orbital_paths(self, original_planet_rows: list[Any]) -> None:
        """Cache each planet's orbital radius and initial angle when it rotates."""
        center_x = center_y = CENTER
        original_by_id = {planet_row[0]: planet_row for planet_row in original_planet_rows}
        self.planet_orbits = {}
        for planet_obj in self.all_planets:
            orbit_radius = distance((planet_obj.coord_x, planet_obj.coord_y), (center_x, center_y))
            if orbit_radius + planet_obj.planet_radius < ROTATION_RADIUS_LIMIT and planet_obj.planet_id in original_by_id:
                original_row = original_by_id[planet_obj.planet_id]
                start_angle = math.atan2(original_row[3] - center_y, original_row[2] - center_x)
                self.planet_orbits[planet_obj] = (orbit_radius, start_angle)
            else:
                self.planet_orbits[planet_obj] = None

    def rebuild_route_graphs(self) -> None:
        """Build directed route maps using projected destination positions."""
        center_x = center_y = CENTER
        self.projected_coords = {}
        for planet_obj in self.all_planets:
            orbit_profile = self.planet_orbits[planet_obj]
            if orbit_profile is not None:
                orbit_radius, start_angle = orbit_profile
                projected_angle = start_angle + self.orbit_angular_velocity * (
                    self.turn_index + 1 + self.ORBIT_LOOKAHEAD_TURNS
                )
                self.projected_coords[planet_obj] = (
                    center_x + orbit_radius * math.cos(projected_angle),
                    center_y + orbit_radius * math.sin(projected_angle),
                )
            else:
                self.projected_coords[planet_obj] = (planet_obj.coord_x, planet_obj.coord_y)

        # Incoming routes are keyed by target so attack planning can gather helpers quickly.
        self.incoming_routes = {planet_obj: [] for planet_obj in self.all_planets}
        for origin_planet in self.all_planets:
            for target_planet in self.all_planets:
                if target_planet is origin_planet:
                    continue
                route_length = distance(
                    (origin_planet.coord_x, origin_planet.coord_y),
                    self.projected_coords[target_planet],
                )
                if route_length <= self.MAX_ROUTE_DISTANCE:
                    self.incoming_routes[target_planet].append((origin_planet, route_length))

        # Outgoing routes are the same edges re-keyed by source for reinforcement flow.
        self.outgoing_routes = {planet_obj: [] for planet_obj in self.all_planets}
        for target_planet, inbound_options in self.incoming_routes.items():
            for origin_planet, route_length in inbound_options:
                self.outgoing_routes[origin_planet].append((target_planet, route_length))

    def assign_supply_targets(self) -> None:
        # Front planets touch enemy reachability and should receive backline ships.
        contested_friendly = {
            planet_obj
            for planet_obj in self.friendly_planets
            if any(origin.controller != self.my_player_id for origin, _ in self.incoming_routes[planet_obj])
            or any(target.controller != self.my_player_id for target, _ in self.outgoing_routes[planet_obj])
        }

        # Breadth-first hop counts route safe rear planets toward the nearest front.
        hops_from_front: dict[BattlePlanet, int] = {planet_obj: 0 for planet_obj in contested_friendly}
        scan_queue: list[BattlePlanet] = list(contested_friendly)
        queue_index = 0
        while queue_index < len(scan_queue):
            current_node = scan_queue[queue_index]
            queue_index += 1
            for origin_planet, _ in self.incoming_routes[current_node]:
                if origin_planet.controller != self.my_player_id or origin_planet in hops_from_front:
                    continue
                hops_from_front[origin_planet] = hops_from_front[current_node] + 1
                scan_queue.append(origin_planet)

        for planet_obj in self.friendly_planets:
            planet_obj.supply_target = None
            if planet_obj in contested_friendly:
                continue

            # Prefer direct front-line recipients when available.
            direct_front_options = [target for target, _ in self.outgoing_routes[planet_obj] if target in contested_friendly]
            if direct_front_options:
                planet_obj.supply_target = min(direct_front_options, key=lambda target: target.ship_count)
                continue

            # Otherwise take the owned neighbor with the shortest route to the front.
            relay_options = [
                target
                for target, _ in self.outgoing_routes[planet_obj]
                if target.controller == self.my_player_id and target not in contested_friendly and target in hops_from_front
            ]
            if relay_options:
                planet_obj.supply_target = min(relay_options, key=lambda target: (hops_from_front[target], target.ship_count))

    def aim_at_planet(
        self,
        start_x: float,
        start_y: float,
        target_planet: BattlePlanet,
        launched_ships: int | float,
        tolerance: float = 1e-6,
        iteration_limit: int = 30,
    ) -> tuple[float, float, float, float]:
        """Return launch angle plus intercept coordinates for a moving target."""
        fleet_speed = self.compute_fleet_speed(launched_ships)
        orbit_profile = self.planet_orbits[target_planet]
        if orbit_profile is None:
            impact_x, impact_y = target_planet.coord_x, target_planet.coord_y
            travel_turns = distance((start_x, start_y), (impact_x, impact_y)) / fleet_speed
        else:
            center_x = center_y = CENTER
            orbit_radius, start_angle = orbit_profile
            travel_turns = distance((start_x, start_y), (target_planet.coord_x, target_planet.coord_y)) / fleet_speed
            for _ in range(iteration_limit):
                orbit_angle = start_angle + self.orbit_angular_velocity * (self.turn_index + travel_turns - 0.5)
                candidate_x = center_x + orbit_radius * math.cos(orbit_angle)
                candidate_y = center_y + orbit_radius * math.sin(orbit_angle)
                next_travel_turns = distance((start_x, start_y), (candidate_x, candidate_y)) / fleet_speed
                next_travel_turns = 0.5 * (travel_turns + next_travel_turns - 0.5)
                if abs(next_travel_turns - travel_turns) < tolerance:
                    travel_turns = next_travel_turns
                    break
                travel_turns = next_travel_turns
            else:
                # Slow fleets may never converge on fast orbital targets.
                return 0.0, target_planet.coord_x, target_planet.coord_y, math.inf
            final_angle = start_angle + self.orbit_angular_velocity * (self.turn_index + travel_turns - 0.5)
            impact_x = center_x + orbit_radius * math.cos(final_angle)
            impact_y = center_y + orbit_radius * math.sin(final_angle)

        launch_angle = math.atan2(impact_y - start_y, impact_x - start_x)
        return launch_angle, impact_x, impact_y, travel_turns

    def trace_first_collision(
        self,
        start_x: float,
        start_y: float,
        launch_angle: float,
        launched_ships: int | float,
        origin_planet: BattlePlanet,
    ) -> BattlePlanet | None:
        """Find the first planet hit by this shot, rejecting shots blocked by the sun."""
        earliest_planet = None
        earliest_turns = float("inf")
        for candidate_planet in self.all_planets:
            if candidate_planet is origin_planet:
                continue
            required_angle, impact_x, impact_y, travel_turns = self.aim_at_planet(
                start_x, start_y, candidate_planet, launched_ships
            )
            impact_distance = distance((start_x, start_y), (impact_x, impact_y))
            if impact_distance < candidate_planet.planet_radius:
                angle_window = math.pi
            else:
                angle_window = math.asin(min(1.0, candidate_planet.planet_radius / impact_distance))
            angle_error = abs(math.atan2(math.sin(launch_angle - required_angle), math.cos(launch_angle - required_angle)))
            if math.isfinite(travel_turns) and angle_error <= angle_window and travel_turns < earliest_turns:
                earliest_turns = travel_turns
                earliest_planet = candidate_planet
        if earliest_planet is None:
            return None

        # The sun cancels a fleet before it reaches the planet if the segment crosses it.
        fleet_speed = self.compute_fleet_speed(launched_ships)
        end_x = start_x + earliest_turns * fleet_speed * math.cos(launch_angle)
        end_y = start_y + earliest_turns * fleet_speed * math.sin(launch_angle)
        if point_to_segment_distance((CENTER, CENTER), (start_x, start_y), (end_x, end_y)) <= SUN_RADIUS:
            return None
        return earliest_planet

    def rebuild_arrival_ledger(self) -> None:
        """Map visible fleets to their most likely planet destination."""
        self.arrival_ledger = defaultdict(list)
        for fleet_obj in self.visible_fleets:
            best_hit = None
            best_travel_turns = float("inf")
            for planet_obj in self.all_planets:
                needed_angle, impact_x, impact_y, travel_turns = self.aim_at_planet(
                    fleet_obj.x, fleet_obj.y, planet_obj, fleet_obj.ships
                )
                impact_distance = distance((fleet_obj.x, fleet_obj.y), (impact_x, impact_y))
                if impact_distance < planet_obj.planet_radius:
                    angle_window = math.pi
                else:
                    angle_window = math.asin(min(1.0, planet_obj.planet_radius / impact_distance))
                angle_error = abs(math.atan2(math.sin(fleet_obj.angle - needed_angle), math.cos(fleet_obj.angle - needed_angle)))
                if math.isfinite(travel_turns) and angle_error <= angle_window and travel_turns < best_travel_turns:
                    best_travel_turns = travel_turns
                    best_hit = (planet_obj, travel_turns, impact_x, impact_y)
            if best_hit is not None:
                planet_obj, travel_turns, impact_x, impact_y = best_hit
                self.arrival_ledger[planet_obj].append(
                    (fleet_obj.owner, fleet_obj.ships, travel_turns, fleet_obj.x, fleet_obj.y, impact_x, impact_y)
                )

    def simulate_planet_outcome(self, planet_obj: BattlePlanet, arrival_ledger: ArrivalLedger) -> tuple[int, float]:
        """Resolve queued arrivals and return final owner plus our minimum post-arrival margin."""
        current_owner = planet_obj.controller
        arrival_entries = arrival_ledger.get(planet_obj)
        if not arrival_entries:
            return current_owner, 0

        arrivals_by_turn = defaultdict(list)
        for fleet_owner, fleet_ships, travel_turns, _, _, _, _ in arrival_entries:
            arrival_turn = max(1, math.ceil(travel_turns))
            arrivals_by_turn[arrival_turn].append((fleet_owner, fleet_ships))

        final_entry_ships, final_entry_time = arrival_entries[-1][1], arrival_entries[-1][2]
        final_entry_turn = max(1, math.ceil(final_entry_time))
        current_ships = float(planet_obj.ship_count)
        production_rate = planet_obj.prod_rate
        previous_turn = 0
        safe_margin = float("inf")

        for arrival_turn in sorted(arrivals_by_turn):
            elapsed_turns = arrival_turn - previous_turn
            if elapsed_turns > 0 and current_owner != -1:
                current_ships += production_rate * elapsed_turns
            previous_turn = arrival_turn

            # Same-turn arrivals fight each other first; only the largest stack survives.
            ships_by_owner = defaultdict(float)
            for fleet_owner, fleet_ships in arrivals_by_turn[arrival_turn]:
                ships_by_owner[fleet_owner] += fleet_ships

            if ships_by_owner:
                ranked_stacks = sorted(ships_by_owner.items(), key=lambda entry: entry[1], reverse=True)
                if len(ranked_stacks) == 1:
                    surviving_owner, surviving_ships = ranked_stacks[0]
                else:
                    top_owner, top_ships = ranked_stacks[0]
                    runner_up_ships = ranked_stacks[1][1]
                    surviving_ships = top_ships - runner_up_ships
                    surviving_owner = top_owner if surviving_ships > 0 else -1

                if surviving_ships > 0:
                    if surviving_owner == current_owner:
                        current_ships += surviving_ships
                    else:
                        current_ships -= surviving_ships
                        if current_ships < 0:
                            current_owner = surviving_owner
                            current_ships = abs(current_ships)

            # Track how much slack we have after the relevant committed fleet arrives.
            if arrival_turn >= final_entry_turn:
                owner_margin = current_ships if current_owner == self.my_player_id else 0.0
                safe_margin = min(safe_margin, owner_margin)

        if safe_margin == float("inf"):
            safe_margin = 0.0
        safe_margin = min(safe_margin, final_entry_ships)
        return current_owner, safe_margin

    def plan_target_commitment(self, target_planet: BattlePlanet) -> tuple[CommandList, list[InterceptRecord], bool]:
        """Gather nearby friendly ships until the target is saved or captured."""
        helper_options = sorted(
            [(origin, route_len) for origin, route_len in self.incoming_routes.get(target_planet, []) if origin.controller == self.my_player_id],
            key=lambda entry: entry[1],
        )

        launch_commands: CommandList = []
        intercept_records: list[InterceptRecord] = []
        trial_ledger = {}
        for ledger_planet, ledger_entries in self.arrival_ledger.items():
            if ledger_planet is target_planet:
                # Discount hostile pressure so the bot does not wildly over-defend.
                trial_ledger[ledger_planet] = [
                    (owner_id, int(ship_total * 0.5) if owner_id != self.my_player_id else ship_total, travel, sx, sy, ix, iy)
                    for owner_id, ship_total, travel, sx, sy, ix, iy in ledger_entries
                ]
            else:
                trial_ledger[ledger_planet] = list(ledger_entries)
        trial_ledger.setdefault(target_planet, [])
        objective_satisfied = False

        # Avoid landing before a third-party battle if an enemy is already attacking the owner.
        earliest_enemy_takeover = None
        if target_planet.controller != self.my_player_id:
            for fleet_owner, _, travel_turns, _, _, _, _ in self.arrival_ledger.get(target_planet, []):
                if fleet_owner != self.my_player_id and fleet_owner != target_planet.controller:
                    arrival_turn = math.ceil(travel_turns)
                    if earliest_enemy_takeover is None or arrival_turn < earliest_enemy_takeover:
                        earliest_enemy_takeover = arrival_turn

        for helper_planet, _ in helper_options:
            if helper_planet.ship_count == 0:
                continue

            proposed_ships = int(helper_planet.ship_count)
            baseline_owner, _ = self.simulate_planet_outcome(helper_planet, self.arrival_ledger)
            helper_survives_baseline = baseline_owner == self.my_player_id
            if helper_survives_baseline:
                risk_ledger = {ledger_planet: list(ledger_entries) for ledger_planet, ledger_entries in self.arrival_ledger.items()}
                risk_ledger.setdefault(helper_planet, [])
                reserve_pressure = 0
                for attacker_planet, _ in self.incoming_routes.get(helper_planet, []):
                    if attacker_planet.controller in (self.my_player_id, -1) or attacker_planet.ship_count == 0:
                        continue
                    _, attack_x, attack_y, attack_travel = self.aim_at_planet(
                        attacker_planet.coord_x, attacker_planet.coord_y, helper_planet, attacker_planet.ship_count
                    )
                    if not math.isfinite(attack_travel):
                        continue
                    pressure_ships = max(1, int(attacker_planet.ship_count * 0.5))
                    risk_ledger[helper_planet].append(
                        (attacker_planet.controller, pressure_ships, attack_travel, attacker_planet.coord_x, attacker_planet.coord_y, attack_x, attack_y)
                    )
                    reserve_pressure += pressure_ships

                saved_ship_total = helper_planet.ship_count
                helper_planet.ship_count = 0
                exposed_owner, _ = self.simulate_planet_outcome(helper_planet, risk_ledger)
                helper_planet.ship_count = saved_ship_total

                if exposed_owner != self.my_player_id:
                    # Sacrifice a helper only when the target produces more than the helper.
                    if target_planet.prod_rate <= helper_planet.prod_rate:
                        continue
                else:
                    # Leave enough reserve to absorb plausible half-strength pressure.
                    proposed_ships = max(0, int(helper_planet.ship_count) - reserve_pressure)
                    if proposed_ships == 0:
                        continue

            launch_angle, impact_x, impact_y, travel_turns = self.aim_at_planet(
                helper_planet.coord_x, helper_planet.coord_y, target_planet, proposed_ships
            )
            if not math.isfinite(travel_turns):
                continue
            if self.trace_first_collision(helper_planet.coord_x, helper_planet.coord_y, launch_angle, proposed_ships, helper_planet) is not target_planet:
                continue
            if earliest_enemy_takeover is not None and math.ceil(travel_turns) <= earliest_enemy_takeover + 1:
                continue

            trial_ledger[target_planet].append(
                (self.my_player_id, proposed_ships, travel_turns, helper_planet.coord_x, helper_planet.coord_y, impact_x, impact_y)
            )
            launch_commands.append([helper_planet.planet_id, launch_angle, proposed_ships])
            intercept_records.append((impact_x, impact_y, travel_turns))
            end_owner, surplus_ships = self.simulate_planet_outcome(target_planet, trial_ledger)
            if end_owner == self.my_player_id:
                objective_satisfied = True

                if helper_survives_baseline:
                    # Trim excess from the last fleet while preserving the winning result.
                    retained_surplus = int(surplus_ships // 2)
                    trimmed_ships = max(10, proposed_ships - retained_surplus)
                    if trimmed_ships < proposed_ships:
                        trim_angle, trim_x, trim_y, trim_travel = self.aim_at_planet(
                            helper_planet.coord_x, helper_planet.coord_y, target_planet, trimmed_ships
                        )
                        if math.isfinite(trim_travel):
                            trial_ledger[target_planet][-1] = (
                                self.my_player_id,
                                trimmed_ships,
                                trim_travel,
                                helper_planet.coord_x,
                                helper_planet.coord_y,
                                trim_x,
                                trim_y,
                            )
                            if self.simulate_planet_outcome(target_planet, trial_ledger)[0] == self.my_player_id:
                                proposed_ships, launch_angle, impact_x, impact_y, travel_turns = (
                                    trimmed_ships,
                                    trim_angle,
                                    trim_x,
                                    trim_y,
                                    trim_travel,
                                )
                            else:
                                trial_ledger[target_planet][-1] = (
                                    self.my_player_id,
                                    proposed_ships,
                                    travel_turns,
                                    helper_planet.coord_x,
                                    helper_planet.coord_y,
                                    impact_x,
                                    impact_y,
                                )
                    launch_commands[-1] = [helper_planet.planet_id, launch_angle, proposed_ships]
                    intercept_records[-1] = (impact_x, impact_y, travel_turns)
                break

        return launch_commands, intercept_records, objective_satisfied

    def select_best_objective(self) -> StrategyPick:
        """Score all reachable planets and select the best current objective."""
        preferred_strategy: StrategyPick = (None, -65535, [], [])

        for candidate_target in sorted(self.all_planets, key=lambda planet_obj: planet_obj.ship_count, reverse=True):
            if not self.incoming_routes.get(candidate_target):
                continue  # No nearby owned planet can reach this target quickly.

            if candidate_target.controller == self.my_player_id:
                if not self.arrival_ledger.get(candidate_target):
                    continue  # Safe owned planets with no inbound fleets need no action.
                projected_owner, _ = self.simulate_planet_outcome(candidate_target, self.arrival_ledger)
                if projected_owner == self.my_player_id:
                    continue
                launch_commands, intercept_records, objective_satisfied = self.plan_target_commitment(candidate_target)
                if not objective_satisfied:
                    continue
                objective_value = candidate_target.prod_rate
            else:
                projected_owner, _ = self.simulate_planet_outcome(candidate_target, self.arrival_ledger)
                if projected_owner == self.my_player_id:
                    continue  # Already captured by fleets in flight.
                launch_commands, intercept_records, objective_satisfied = self.plan_target_commitment(candidate_target)
                if not objective_satisfied:
                    continue
                objective_value = candidate_target.prod_rate - (1 if candidate_target.controller == -1 else 0)

            _, best_value, best_commands, _ = preferred_strategy
            if objective_value > best_value or (objective_value == best_value and len(launch_commands) < len(best_commands)):
                preferred_strategy = (candidate_target, objective_value, launch_commands, intercept_records)

        return preferred_strategy

    def create_supply_commands(self) -> CommandList:
        """Move idle backline ships toward assigned friendly supply targets."""
        supply_commands: CommandList = []
        for origin_planet in self.friendly_planets:
            if origin_planet.supply_target is None:
                continue
            if origin_planet.ship_count < (self.SUPPLY_BATCH_SIZE + self.HOME_GARRISON_SIZE):
                continue
            threatened_origin = any(
                source_planet.controller != self.my_player_id for source_planet, _ in self.incoming_routes.get(origin_planet, [])
            )
            if threatened_origin:
                continue
            target_planet = origin_planet.supply_target
            launched_ships = int(origin_planet.ship_count - self.HOME_GARRISON_SIZE)
            launch_angle, _, _, travel_turns = self.aim_at_planet(
                origin_planet.coord_x, origin_planet.coord_y, target_planet, launched_ships
            )
            if not math.isfinite(travel_turns):
                continue
            supply_commands.append([origin_planet.planet_id, launch_angle, launched_ships])
        return supply_commands

    def commit_strategy_to_ledger(self, strategy_pick: StrategyPick) -> None:
        """Subtract planned ships locally so later planning does not double-spend them."""
        target_planet, _, launch_commands, intercept_records = strategy_pick
        for (origin_id, _, launched_ships), (impact_x, impact_y, travel_turns) in zip(launch_commands, intercept_records):
            origin_planet = next((planet_obj for planet_obj in self.all_planets if planet_obj.planet_id == origin_id), None)
            if origin_planet is None:
                continue
            origin_planet.ship_count = max(0, origin_planet.ship_count - launched_ships)
            self.arrival_ledger.setdefault(target_planet, [])
            self.arrival_ledger[target_planet].append(
                (self.my_player_id, launched_ships, travel_turns, origin_planet.coord_x, origin_planet.coord_y, impact_x, impact_y)
            )

    def opening_travel_turns(self, origin_id: int, target_planet: BattlePlanet, launched_ships: int, launch_turn: int) -> float:
        # Rewind or project the origin if it is also orbiting.
        origin_planet = next(planet_obj for planet_obj in self.all_planets if planet_obj.planet_id == origin_id)
        orbit_profile = self.planet_orbits.get(origin_planet)
        if orbit_profile is not None:
            center_x = center_y = CENTER
            orbit_radius, start_angle = orbit_profile
            launch_angle = start_angle + self.orbit_angular_velocity * (launch_turn - 0.5)
            start_x = center_x + orbit_radius * math.cos(launch_angle)
            start_y = center_y + orbit_radius * math.sin(launch_angle)
        else:
            start_x, start_y = origin_planet.coord_x, origin_planet.coord_y
        _, _, _, travel_turns = self.aim_at_planet(start_x, start_y, target_planet, launched_ships)
        return travel_turns

    def opening_first_capture_turn(self, opening_state: OpeningState, target_planet: BattlePlanet) -> float:
        """Return earliest turn when one owned planet can beat the target garrison."""
        target_garrison = target_planet.ship_count
        search_horizon = opening_state.current_turn + self.OPENING_HORIZON
        best_arrival_turn = math.inf
        for origin_id in opening_state.friendly_ids:
            stationed_ships = opening_state.stationed_ships[origin_id]
            production_rate = opening_state.production_rates[origin_id]
            for wait_turns in range(self.OPENING_HORIZON):
                launched_ships = int(stationed_ships + production_rate * wait_turns)
                if launched_ships <= target_garrison:
                    continue
                launch_turn = opening_state.current_turn + wait_turns
                if launch_turn >= search_horizon:
                    break
                travel_turns = self.opening_travel_turns(origin_id, target_planet, launched_ships, launch_turn)
                if not math.isfinite(travel_turns):
                    continue
                arrival_turn = launch_turn + math.ceil(travel_turns)
                if arrival_turn <= search_horizon:
                    best_arrival_turn = min(best_arrival_turn, arrival_turn)
                    break
        return best_arrival_turn

    def opening_assign_capture(self, opening_state: OpeningState, target_planet: BattlePlanet, capture_turn: int) -> dict:
        """Choose the single source that can capture with the earliest landing."""
        target_garrison = target_planet.ship_count
        chosen_origin = None
        chosen_payload = None
        earliest_arrival = math.inf
        for origin_id in opening_state.friendly_ids:
            stationed_ships = opening_state.stationed_ships[origin_id]
            production_rate = opening_state.production_rates[origin_id]
            for wait_turns in range(capture_turn - opening_state.current_turn):
                launched_ships = int(stationed_ships + production_rate * wait_turns)
                if launched_ships <= target_garrison:
                    continue
                launch_turn = opening_state.current_turn + wait_turns
                travel_turns = self.opening_travel_turns(origin_id, target_planet, launched_ships, launch_turn)
                if not math.isfinite(travel_turns):
                    continue
                arrival_turn = launch_turn + math.ceil(travel_turns)
                if arrival_turn <= capture_turn and arrival_turn < earliest_arrival:
                    earliest_arrival = arrival_turn
                    chosen_origin = origin_id
                    chosen_payload = (launched_ships, launch_turn, arrival_turn)
                break
        if chosen_origin is None:
            return {}
        return {chosen_origin: chosen_payload}

    def opening_advance_state(self, opening_state: OpeningState, start_turn: int, end_turn: int) -> OpeningState:
        # Advance production and resolve scheduled captures one turn at a time.
        for current_turn in range(start_turn + 1, end_turn + 1):
            for fleet_obj in list(opening_state.active_fleets):
                if fleet_obj.landing_turn == current_turn:
                    if fleet_obj.captures_planet:
                        opening_state.stationed_ships[fleet_obj.target_id] = fleet_obj.landing_garrison
                        opening_state.friendly_ids.add(fleet_obj.target_id)
                        if fleet_obj.target_id not in opening_state.production_rates:
                            opening_state.production_rates[fleet_obj.target_id] = self.opening_production(fleet_obj.target_id)
                    else:
                        opening_state.stationed_ships[fleet_obj.target_id] += fleet_obj.landing_garrison
                    opening_state.active_fleets.remove(fleet_obj)
            for planet_id in opening_state.friendly_ids:
                opening_state.stationed_ships[planet_id] += opening_state.production_rates[planet_id]
        return opening_state

    def opening_execute_capture(
        self,
        opening_state: OpeningState,
        target_planet: BattlePlanet,
        fleet_assignment: dict,
        capture_turn: int,
    ) -> OpeningState:
        # Deduct launched ships, then schedule a capture fleet at the target.
        target_garrison = target_planet.ship_count
        total_launched = sum(launched for launched, _, _ in fleet_assignment.values())
        current_turn = opening_state.current_turn
        for origin_id, (launched_ships, launch_turn, _) in sorted(fleet_assignment.items(), key=lambda entry: entry[1][1]):
            opening_state = self.opening_advance_state(opening_state, current_turn, launch_turn)
            current_turn = launch_turn
            opening_state.stationed_ships[origin_id] -= launched_ships

        opening_state.active_fleets.append(
            OpeningFleet(
                origin_id=-1,
                target_id=target_planet.planet_id,
                launched_ships=total_launched,
                landing_garrison=total_launched - target_garrison,
                landing_turn=capture_turn,
                captures_planet=True,
            )
        )
        return self.opening_advance_state(opening_state, current_turn, capture_turn)

    def opening_score(self, opening_state: OpeningState) -> int:
        # Score combines present ships and expected production through the opening horizon.
        search_horizon = opening_state.current_turn + self.OPENING_HORIZON
        total_score = 0
        for planet_id in opening_state.friendly_ids:
            total_score += opening_state.stationed_ships[planet_id] + opening_state.production_rates[planet_id] * (
                search_horizon - opening_state.current_turn
            )
        for fleet_obj in opening_state.active_fleets:
            total_score += fleet_obj.landing_garrison
            if fleet_obj.captures_planet:
                total_score += self.opening_production(fleet_obj.target_id) * max(0, search_horizon - fleet_obj.landing_turn)
        return total_score

    def opening_production(self, planet_id: int) -> int:
        planet_obj = next((candidate for candidate in self.all_planets if candidate.planet_id == planet_id), None)
        return planet_obj.prod_rate if planet_obj else 0

    def run_opening_optimizer(self) -> list:
        """Search profitable neutral captures for the first few turns."""
        friendly_ids = {planet_obj.planet_id for planet_obj in self.friendly_planets}
        neutral_targets = [
            planet_obj
            for planet_obj in self.all_planets
            if planet_obj.controller == -1 and any(origin.planet_id in friendly_ids for origin, _ in self.incoming_routes.get(planet_obj, []))
        ]

        # Include friendly fleets already airborne so the search avoids duplicate targets.
        airborne_fleets: list[OpeningFleet] = []
        for destination_planet, arrival_entries in self.arrival_ledger.items():
            for fleet_owner, fleet_ships, travel_turns, _, _, _, _ in arrival_entries:
                if fleet_owner != self.my_player_id:
                    continue
                landing_turn = self.turn_index + math.ceil(travel_turns)
                captures_planet = destination_planet.controller != self.my_player_id
                surplus_ships = fleet_ships - destination_planet.ship_count
                airborne_fleets.append(
                    OpeningFleet(
                        origin_id=-1,
                        target_id=destination_planet.planet_id,
                        launched_ships=int(fleet_ships),
                        landing_garrison=int(surplus_ships) if captures_planet else int(fleet_ships),
                        landing_turn=landing_turn,
                        captures_planet=captures_planet,
                    )
                )

        initial_opening_state = OpeningState(
            current_turn=self.turn_index,
            stationed_ships={planet_obj.planet_id: float(planet_obj.ship_count) for planet_obj in self.friendly_planets},
            production_rates={planet_obj.planet_id: planet_obj.prod_rate for planet_obj in self.friendly_planets},
            friendly_ids=friendly_ids.copy(),
            active_fleets=airborne_fleets,
        )

        def estimate_initial_gain(target_planet: BattlePlanet) -> float:
            capture_turn = self.opening_first_capture_turn(initial_opening_state, target_planet)
            search_horizon = initial_opening_state.current_turn + self.OPENING_HORIZON
            return target_planet.prod_rate * (search_horizon - capture_turn) - target_planet.ship_count if math.isfinite(capture_turn) else -math.inf

        candidate_targets = sorted(neutral_targets, key=estimate_initial_gain, reverse=True)
        candidate_targets = [target for target in candidate_targets if estimate_initial_gain(target) > 0]
        if not candidate_targets:
            return []

        best_result = [self.opening_score(initial_opening_state), []]

        def optimistic_bound(opening_state: OpeningState, remaining_targets: list[BattlePlanet]) -> float:
            search_horizon = opening_state.current_turn + self.OPENING_HORIZON
            bound_score = self.opening_score(opening_state)
            for target_planet in remaining_targets:
                capture_turn = self.opening_first_capture_turn(opening_state, target_planet)
                gain_value = target_planet.prod_rate * (search_horizon - capture_turn) - target_planet.ship_count
                if gain_value > 0:
                    bound_score += gain_value
            return bound_score

        def depth_first_search(opening_state: OpeningState, remaining_targets: list[BattlePlanet], capture_sequence: list) -> None:
            current_score = self.opening_score(opening_state)
            if current_score > best_result[0]:
                best_result[0] = current_score
                best_result[1] = list(capture_sequence)
            if optimistic_bound(opening_state, remaining_targets) <= best_result[0]:
                return

            already_targeted_ids = {fleet_obj.target_id for fleet_obj in opening_state.active_fleets if fleet_obj.captures_planet}
            for target_index, target_planet in enumerate(remaining_targets):
                if target_planet.planet_id in already_targeted_ids:
                    continue
                search_horizon = opening_state.current_turn + self.OPENING_HORIZON
                capture_turn = self.opening_first_capture_turn(opening_state, target_planet)
                if not math.isfinite(capture_turn):
                    continue
                if target_planet.prod_rate * (search_horizon - capture_turn) - target_planet.ship_count <= 0:
                    continue
                fleet_assignment = self.opening_assign_capture(opening_state, target_planet, capture_turn)
                if not fleet_assignment:
                    continue
                next_state = self.opening_execute_capture(copy.deepcopy(opening_state), target_planet, fleet_assignment, capture_turn)
                next_remaining = remaining_targets[:target_index] + remaining_targets[target_index + 1 :]
                depth_first_search(next_state, next_remaining, capture_sequence + [(target_planet, fleet_assignment, capture_turn)])

        depth_first_search(initial_opening_state, candidate_targets, [])
        best_sequence = best_result[1]
        if not best_sequence:
            return []

        # Only emit moves whose launch turn is the current environment step.
        opening_commands: CommandList = []
        for target_planet, fleet_assignment, _ in best_sequence:
            for origin_id, (launched_ships, launch_turn, _) in fleet_assignment.items():
                if launch_turn != self.turn_index:
                    continue
                origin_planet = next((planet_obj for planet_obj in self.all_planets if planet_obj.planet_id == origin_id), None)
                if origin_planet is None:
                    continue
                launch_angle, _, _, travel_turns = self.aim_at_planet(
                    origin_planet.coord_x, origin_planet.coord_y, target_planet, launched_ships
                )
                if not math.isfinite(travel_turns):
                    continue
                first_hit = self.trace_first_collision(
                    origin_planet.coord_x, origin_planet.coord_y, launch_angle, launched_ships, origin_planet
                )
                if first_hit is not target_planet:
                    continue
                opening_commands.append([origin_id, launch_angle, launched_ships])

        return opening_commands

    def main(self, observation: dict[str, Any]) -> list[Any]:
        """Translate the environment observation into launch commands."""
        self.my_player_id = observation["player"]
        self.turn_index = observation["step"] - 1
        self.orbit_angular_velocity = observation["angular_velocity"]

        comet_planet_ids = set(observation["comet_planet_ids"])
        all_bodies = [BattlePlanet(*planet_row) for planet_row in observation["planets"]]
        self.all_planets = [planet_obj for planet_obj in all_bodies if planet_obj.planet_id not in comet_planet_ids]
        self.friendly_planets = [planet_obj for planet_obj in self.all_planets if planet_obj.controller == self.my_player_id]
        self.hostile_planets = [planet_obj for planet_obj in self.all_planets if planet_obj.controller != self.my_player_id]
        self.visible_fleets = [Fleet(*fleet_row) for fleet_row in observation["fleets"]]

        if not self.hostile_planets:
            return []

        self.index_orbital_paths(observation.get("initial_planets", []))
        self.rebuild_route_graphs()
        self.rebuild_arrival_ledger()

        if self.turn_index < self.OPENING_TURNS:
            return self.run_opening_optimizer()

        self.assign_supply_targets()

        planned_commands: CommandList = []
        while True:
            strategy_pick = self.select_best_objective()
            target_planet, _, launch_commands, _ = strategy_pick
            if target_planet is None:
                break
            self.commit_strategy_to_ledger(strategy_pick)
            planned_commands.extend(launch_commands)

        supply_commands = self.create_supply_commands()
        if supply_commands:
            planned_commands.extend(supply_commands)

        return planned_commands


def agent(observation: dict[str, Any]) -> list[Any]:
    commander = OrbitCommander()
    try:
        return commander.main(observation)
    except Exception:
        return []