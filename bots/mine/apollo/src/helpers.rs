#![allow(dead_code)]

use std::rc::Rc;

use rustc_hash::FxHashMap as HashMap;

use crate::constants::{CENTER, LAUNCH_CLEARANCE, MAX_PLAYERS, MAX_SHIP_SPEED, SUN_RADIUS};

use crate::blockers;
pub use crate::blockers::AimResult;
use crate::engine::{Fleet, Planet};
use crate::entity_cache::EntityCache;
use crate::engine::Simulator;
pub use crate::engine::ArrivalEvent;


// ── Basic Helpers ────────────────────────────────────────────────────

#[inline]
pub fn dist(ax: f64, ay: f64, bx: f64, by: f64) -> f64 {
    crate::engine::distance((ax, ay), (bx, by))
}

/// Logarithmic speed curve between 1 and 6 (engine rule).
#[inline]
pub fn fleet_speed(ships: i64) -> f64 {
    crate::engine::fleet_speed(ships.max(1), MAX_SHIP_SPEED)
}


#[inline]
pub fn point_to_segment_distance_sq(
    px: f64, py: f64,
    x1: f64, y1: f64,
    x2: f64, y2: f64,
) -> f64 {
    crate::engine::point_to_segment_distance_sq((px, py), (x1, y1), (x2, y2))
}

#[inline]
pub fn segment_intersects_circle(
    ax: f64, ay: f64,
    bx: f64, by: f64,
    cx: f64, cy: f64,
    r: f64,
) -> bool {
    point_to_segment_distance_sq(cx, cy, ax, ay, bx, by) <= r * r
}

#[inline]
pub fn segment_hits_sun(
    x1: f64, y1: f64,
    x2: f64, y2: f64,
) -> bool {
    point_to_segment_distance_sq(CENTER, CENTER, x1, y1, x2, y2) < SUN_RADIUS * SUN_RADIUS
}

/// Fleet spawns at (radius + 0.1) from planet center, at launch angle
#[inline]
pub fn launch_point(sx: f64, sy: f64, sr: f64, angle: f64) -> (f64, f64) {
    let c = sr + LAUNCH_CLEARANCE;
    (sx + angle.cos() * c, sy + angle.sin() * c)
}

/// Public aim entry point. Delegates to the parametric blocker pipeline in
/// [`crate::blockers`]: lead the target with Newton iteration, then reject
/// the shot if any blocker (sun, static planet, orbiter, comet) covers the
/// resulting `(angle, flight_time)` pair. Pass `launch_turn_offset = 0` to
/// launch now; non-zero offsets evaluate source/target/obstacle positions
/// at the future launch turn (used by the hellburner early-game DFS).
#[inline]
pub fn aim_with_prediction(
    cache: &EntityCache,
    shooter_id: i64,
    target_id: i64,
    ships: i64,
    launch_turn_offset: i64,
) -> Option<AimResult> {
    blockers::aim_with_prediction(cache, shooter_id, target_id, ships, launch_turn_offset)
}

/// Comet-free aim used to compute the turn-invariant base for the invariant-aim
/// cache; see [`crate::blockers::aim_ignoring_comets`].
#[inline]
pub fn aim_ignoring_comets(
    cache: &EntityCache,
    shooter_id: i64,
    target_id: i64,
    ships: i64,
    launch_turn_offset: i64,
) -> Option<AimResult> {
    blockers::aim_ignoring_comets(cache, shooter_id, target_id, ships, launch_turn_offset)
}

/// Returns the engine's player-slot count: `max_owner + 1` across all
/// non-neutral planets and fleets, floored at 2. Engine code indexes into
/// per-player arrays by owner id directly, so the slot count has to be
/// large enough to hold the *highest* owner id — not just the distinct
/// count. E.g. in a 4P game where players 1 and 2 have been wiped but a
/// planet is still owned by player 3, the distinct count is 2 but we need
/// 4 slots (indices 0..=3).
///
/// Player ids are always in `0..MAX_PLAYERS` (or -1 for neutral).
pub fn count_players(planets: &[Planet], fleets: &[Fleet]) -> usize {
    let mut max_owner: i64 = -1;
    for p in planets {
        if p.owner > max_owner { max_owner = p.owner; }
    }
    for f in fleets {
        if f.owner > max_owner { max_owner = f.owner; }
    }
    if max_owner < 0 {
        return 2;
    }
    let n = (max_owner as usize + 1).min(MAX_PLAYERS);
    n.max(2)
}

/// Shortest distance from `(px, py)` to the center of any planet in `set`.
/// Returns `f64::INFINITY` for an empty set so callers can compare freely.
pub fn nearest_distance_to_set(px: f64, py: f64, set: &[Planet]) -> f64 {
    set.iter()
        .map(|p| dist(px, py, p.x, p.y))
        .fold(f64::INFINITY, f64::min)
}

/// `(planet, distance)` pairs sorted ascending by distance from `(tx, ty)`.
pub fn sorted_by_distance_to(
    planets: &[Planet],
    tx: f64, ty: f64,
) -> Vec<(Planet, f64)> {
    let mut out: Vec<(Planet, f64)> = planets
        .iter()
        .map(|p| (p.clone(), dist(p.x, p.y, tx, ty)))
        .collect();
    out.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    out
}


// ── Timeline Helpers ──────────────────────────────────────────────────────────
// Forward simulation with initial timeline and hypothetical queries

/// Per-planet arrival ledger: `{planet_id → [ArrivalEvent, ...]}`.
/// Built once per turn via [`TimelineCache::build`].
pub type ArrivalsByPlanet = HashMap<i64, Vec<ArrivalEvent>>;

/// Same-turn combat resolution for the timeline projection. Aggregates the
/// arrival list into per-player ship totals and defers to the shared
/// [`crate::engine::resolve_combat`] rule (the same one the forward
/// [`Simulator`] uses), so the projection can never silently drift from the
/// simulator's combat. Neutral (`-1`) arrivals are ignored — real fleets always
/// have an owner `>= 0`.
pub fn resolve_arrival_event(
    owner: i64,
    garrison: i64,
    arrivals: &[ArrivalEvent],
) -> (i64, i64) {
    let mut incoming = [0i64; MAX_PLAYERS];
    for ev in arrivals {
        if ev.owner >= 0 && (ev.owner as usize) < MAX_PLAYERS {
            incoming[ev.owner as usize] += ev.ships;
        }
    }
    crate::engine::resolve_combat(owner, garrison, &incoming)
}

/// Clean per-turn arrival list: drops non-positive ships, clamps `turns` to
/// at least 1, drops past `horizon`, sorts by ETA.
pub fn normalize_arrivals(arrivals: &[ArrivalEvent], horizon: i64) -> Vec<ArrivalEvent> {
    let mut out: Vec<ArrivalEvent> = arrivals
        .iter()
        .filter(|ev| ev.ships > 0)
        .map(|ev| ArrivalEvent {
            turns: ev.turns.max(1),
            owner: ev.owner,
            ships: ev.ships,
        })
        .filter(|ev| ev.turns <= horizon)
        .collect();
    out.sort_by_key(|ev| ev.turns);
    out
}

/// Forward-simulated state for one planet across turns `0..=horizon`.
/// Indexable directly by turn — `owner_at[t]` and `ships_at[t]` are the
/// post-combat snapshot at end of turn `t`.
///
/// `owner_at`/`ships_at` are `Rc`-shared: the trajectory is player-agnostic, so
/// every player's baseline built from the same [`ArrivalLedger`] points at one
/// allocation instead of a per-player copy.
#[derive(Debug, Clone)]
pub struct PlanetTimeline {
    pub owner_at: Rc<Vec<i64>>,
    pub ships_at: Rc<Vec<i64>>,
    /// Minimum garrison that, if kept on the planet, still survives every
    /// arrival through `horizon` (binary-searched). Only meaningful when the
    /// planet currently belongs to `player`.
    pub keep_needed: i64,
    /// Smallest garrison observed while `player` continuously owns the
    /// planet. 0 when `player` does not currently own it.
    pub min_owned: i64,
    /// First turn within the horizon where an enemy arrival lands while we
    /// own the planet.
    pub first_enemy: Option<i64>,
    /// Turn we lose the planet to a non-player owner, if it falls within the
    /// horizon.
    pub fall_turn: Option<i64>,
    /// `false` iff even keeping every current ship can't hold the planet.
    pub holds_full: bool,
    pub horizon: i64,
}

/// Player-agnostic forward trajectory of one planet: the per-turn `owner_at` /
/// `ships_at` arrays plus the per-turn arrival buckets used to derive
/// player-specific metrics. Shared (`Rc`) so baselines for different players
/// reuse one allocation. See [`build_trajectory`] / [`finish_timeline`].
#[derive(Debug, Clone)]
pub struct Trajectory {
    owner_at: Rc<Vec<i64>>,
    ships_at: Rc<Vec<i64>>,
    by_turn: Rc<Vec<Vec<ArrivalEvent>>>,
    effective_horizon: i64,
}

/// Build the player-agnostic [`Trajectory`]: production each turn (while owned),
/// then arrivals resolved via `resolve_arrival_event`.
///
/// `expiry_turn`, when set, is the turn the planet leaves the board. Turns
/// at or past expiry are recorded as ownerless with zero ships.
pub fn build_trajectory(
    planet: &Planet,
    arrivals: &[ArrivalEvent],
    horizon: i64,
    expiry_turn: Option<i64>,
) -> Trajectory {
    let horizon = horizon.max(0);
    let effective_horizon = match expiry_turn {
        Some(exp) => horizon.min((exp - 1).max(0)),
        None => horizon,
    };
    let events = normalize_arrivals(arrivals, effective_horizon);

    let len = (horizon + 1) as usize;
    let mut by_turn: Vec<Vec<ArrivalEvent>> = vec![Vec::new(); len];
    for ev in &events {
        by_turn[ev.turns as usize].push(*ev);
    }

    let mut owner = planet.owner;
    let mut garrison = planet.ships;
    let mut owner_at: Vec<i64> = vec![owner; len];
    let mut ships_at: Vec<i64> = vec![garrison.max(0); len];

    for turn in 1..=effective_horizon {
        if owner != -1 {
            garrison += planet.production;
        }
        let group = &by_turn[turn as usize];
        if !group.is_empty() {
            let (no, ng) = resolve_arrival_event(owner, garrison, group);
            owner = no;
            garrison = ng;
        }
        owner_at[turn as usize] = owner;
        ships_at[turn as usize] = garrison.max(0);
    }

    // Past expiry the planet doesn't exist: no owner, no ships.
    for turn in (effective_horizon + 1)..=horizon {
        owner_at[turn as usize] = -1;
        ships_at[turn as usize] = 0;
    }

    Trajectory {
        owner_at: Rc::new(owner_at),
        ships_at: Rc::new(ships_at),
        by_turn: Rc::new(by_turn),
        effective_horizon,
    }
}

/// Derive the player-specific summary metrics for `player` from a shared
/// [`Trajectory`], producing a full [`PlanetTimeline`] that shares the
/// trajectory's `owner_at`/`ships_at` arrays. `keep_needed`/`holds_full` are
/// only computed when `planet` currently belongs to `player`.
pub fn finish_timeline(
    planet: &Planet,
    player: i64,
    horizon: i64,
    traj: &Trajectory,
) -> PlanetTimeline {
    let owner_at = &traj.owner_at;
    let ships_at = &traj.ships_at;
    let by_turn = &traj.by_turn;
    let effective_horizon = traj.effective_horizon;

    let mut min_owned: i64 = if owner_at[0] == player { ships_at[0] } else { 0 };
    let mut first_enemy: Option<i64> = None;
    let mut fall_turn: Option<i64> = None;

    for turn in 1..=effective_horizon {
        let t = turn as usize;
        let prev_owner = owner_at[t - 1];
        let group = &by_turn[t];
        if !group.is_empty() {
            if prev_owner == player
                && first_enemy.is_none()
                && group.iter().any(|ev| ev.owner != -1 && ev.owner != player)
            {
                first_enemy = Some(turn);
            }
            if prev_owner == player && owner_at[t] != player && fall_turn.is_none() {
                fall_turn = Some(turn);
            }
        }
        if owner_at[t] == player {
            min_owned = min_owned.min(ships_at[t]);
        }
    }

    let mut keep_needed: i64 = 0;
    let mut holds_full = true;
    if planet.owner == player {
        let survives = |keep: i64| -> bool {
            let mut sim_owner = planet.owner;
            let mut sim_garrison = keep;
            for turn in 1..=effective_horizon {
                if sim_owner != -1 {
                    sim_garrison += planet.production;
                }
                let group = &by_turn[turn as usize];
                if !group.is_empty() {
                    let (no, ng) =
                        resolve_arrival_event(sim_owner, sim_garrison, group);
                    sim_owner = no;
                    sim_garrison = ng;
                    if sim_owner != player {
                        return false;
                    }
                }
            }
            sim_owner == player
        };

        if survives(planet.ships) {
            let (mut lo, mut hi) = (0i64, planet.ships);
            while lo < hi {
                let mid = lo + (hi - lo) / 2;
                if survives(mid) {
                    hi = mid;
                } else {
                    lo = mid + 1;
                }
            }
            keep_needed = lo;
        } else {
            holds_full = false;
            keep_needed = planet.ships;
        }
    }

    PlanetTimeline {
        owner_at: Rc::clone(&traj.owner_at),
        ships_at: Rc::clone(&traj.ships_at),
        keep_needed,
        min_owned: if planet.owner == player {
            min_owned.max(0)
        } else {
            0
        },
        first_enemy,
        fall_turn,
        holds_full,
        horizon,
    }
}

/// Turn-by-turn rollout of one planet under a given arrival schedule, with the
/// player-specific summary metrics for `player`. Convenience wrapper over
/// [`build_trajectory`] + [`finish_timeline`] for callers that don't have a
/// cached trajectory (hypothetical "what-if" queries).
pub fn simulate_planet_timeline(
    planet: &Planet,
    arrivals: &[ArrivalEvent],
    player: i64,
    horizon: i64,
    expiry_turn: Option<i64>,
) -> PlanetTimeline {
    let traj = build_trajectory(planet, arrivals, horizon, expiry_turn);
    finish_timeline(planet, player, horizon.max(0), &traj)
}

/// Reads `(owner, ships)` out of a timeline at `arrival_turn`. Clamps the
/// query into `[0, horizon]` so callers don't have to bounds-check.
pub fn state_at_timeline(timeline: &PlanetTimeline, arrival_turn: i64) -> (i64, i64) {
    let turn = arrival_turn.max(0).min(timeline.horizon) as usize;
    (timeline.owner_at[turn], timeline.ships_at[turn].max(0))
}

/// Checkpointed re-simulation that writes the post-`start_turn` trajectory into
/// caller-provided buffers, reusing `baseline` for the unchanged prefix. Used by
/// the capture/hold binary searches, which probe many ship counts per query —
/// passing reusable buffers avoids a per-probe `Vec`/`Rc` allocation.
///
/// `owner_at`/`ships_at` are reset to `baseline`'s arrays and then rewritten for
/// turns `>= start_turn`; `by_turn` is a reusable arrival-bucket scratch. After
/// the call, `owner_at[t]`/`ships_at[t]` hold the post-combat state at turn `t`,
/// exactly as the per-turn arrays of `simulate_planet_timeline` would.
///
/// Precondition: arrivals differing from baseline must land at turn
/// `>= start_turn`; earlier arrivals are assumed already in `baseline`.
pub fn simulate_checkpoint_into(
    planet: &Planet,
    baseline: &PlanetTimeline,
    start_turn: i64,
    arrivals: &[ArrivalEvent],
    expiry_turn: Option<i64>,
    owner_at: &mut Vec<i64>,
    ships_at: &mut Vec<i64>,
    by_turn: &mut Vec<Vec<ArrivalEvent>>,
) {
    let horizon = baseline.horizon;
    let start_turn = start_turn.clamp(1, horizon.max(1));
    let effective_horizon = match expiry_turn {
        Some(exp) => horizon.min((exp - 1).max(0)),
        None => horizon,
    };
    let len = (horizon + 1) as usize;

    // Reset the reusable arrival buckets (cleared, not re-allocated). Inline the
    // `normalize_arrivals` filters; ordering within a turn doesn't matter because
    // `resolve_arrival_event` aggregates by owner.
    if by_turn.len() < len {
        by_turn.resize_with(len, Vec::new);
    }
    for bucket in by_turn[..len].iter_mut() {
        bucket.clear();
    }
    for ev in arrivals {
        if ev.ships <= 0 {
            continue;
        }
        let turn = ev.turns.max(1);
        if turn >= start_turn && turn <= effective_horizon {
            by_turn[turn as usize].push(*ev);
        }
    }

    // Seed the buffers from the baseline prefix (no allocation once sized), then
    // rewrite turns >= start_turn below.
    owner_at.clear();
    owner_at.extend_from_slice(&baseline.owner_at);
    ships_at.clear();
    ships_at.extend_from_slice(&baseline.ships_at);

    let checkpoint_idx = (start_turn - 1) as usize;
    let mut owner = owner_at[checkpoint_idx];
    let mut garrison = ships_at[checkpoint_idx];

    for turn in start_turn..=effective_horizon {
        if owner != -1 {
            garrison += planet.production;
        }
        let group = &by_turn[turn as usize];
        if !group.is_empty() {
            let (no, ng) = resolve_arrival_event(owner, garrison, group);
            owner = no;
            garrison = ng;
        }
        owner_at[turn as usize] = owner;
        ships_at[turn as usize] = garrison.max(0);
    }

    // Past expiry the planet doesn't exist.
    for turn in (effective_horizon + 1).max(start_turn)..=horizon {
        owner_at[turn as usize] = -1;
        ships_at[turn as usize] = 0;
    }
}

/// Player-agnostic forward-sim output: the in-flight arrival ledger and the
/// expiry map. Both are functions of simulation state alone — building them once
/// per rollout step and sharing across every player's `TimelineCache` is the
/// main reason to keep this split out.
#[derive(Debug, Clone)]
pub struct ArrivalLedger {
    pub horizon: i64,
    /// Shared so per-player `TimelineCache`s built from the same ledger bump a
    /// refcount instead of deep-cloning the whole arrivals map.
    pub ledger: Rc<ArrivalsByPlanet>,
    /// Turn at which a planet leaves the board, populated only for planets
    /// that expire within `horizon` (i.e. comets near end of life). Missing
    /// entry means the planet survives the entire window.
    pub expiry_at: HashMap<i64, i64>,
    /// Player-agnostic per-planet forward trajectories, computed once here so
    /// every player's baseline ([`TimelineCache::from_ledger`]) reuses the
    /// shared `owner_at`/`ships_at` arrays instead of re-simulating them.
    trajectories: HashMap<i64, Trajectory>,
}

impl ArrivalLedger {
    /// Fork `parent` and walk forward `horizon` turns, collecting per-planet
    /// arrivals plus expiry turns and the player-agnostic trajectories.
    /// `O(horizon * |planets|)`. This is the expensive step that gets shared
    /// across players in [`rollout`].
    pub fn build(parent: &Simulator, horizon: i64, entity_cache: &EntityCache) -> Self {
        let mut sim = parent.fork();
        sim.step_n(horizon, Some(entity_cache));
        let mut ledger = sim.collect_arrivals();
        let mut expiry_at: HashMap<i64, i64> = HashMap::default();
        let mut trajectories: HashMap<i64, Trajectory> =
            HashMap::with_capacity_and_hasher(parent.planets().len(), Default::default());
        for planet in parent.planets() {
            ledger.entry(planet.id).or_default();
            let expiry = expiry_within_horizon(entity_cache, planet.id, horizon);
            if let Some(exp) = expiry {
                expiry_at.insert(planet.id, exp);
            }
            let arrivals = ledger.get(&planet.id).map(|v| v.as_slice()).unwrap_or(&[]);
            trajectories.insert(planet.id, build_trajectory(planet, arrivals, horizon, expiry));
        }
        Self { horizon, ledger: Rc::new(ledger), expiry_at, trajectories }
    }

    pub fn arrivals(&self, planet_id: i64) -> &[ArrivalEvent] {
        self.ledger
            .get(&planet_id)
            .map(|v| v.as_slice())
            .unwrap_or(&[])
    }

    #[inline]
    pub fn expiry(&self, planet_id: i64) -> Option<i64> {
        self.expiry_at.get(&planet_id).copied()
    }

    /// Shared player-agnostic trajectory for a planet, if it was present when
    /// the ledger was built.
    #[inline]
    pub fn trajectory(&self, planet_id: i64) -> Option<&Trajectory> {
        self.trajectories.get(&planet_id)
    }
}

/// One-call cache that holds both the arrival ledger and per-planet baseline
/// timelines, built from a single `Simulator` rollout.
///
/// Typical use: call `TimelineCache::build` once per bot turn, then pass the
/// cache to capture/hold queries (`min_ships_to_own_by`,
/// `reinforcement_needed_to_hold_until`) and any per-planet timeline reads.
/// Subsequent hypothetical-arrival queries pay only for the planets they
/// touch, starting from the baseline checkpoint at the arrival turn.
#[derive(Debug, Clone)]
pub struct TimelineCache {
    pub player: i64,
    pub horizon: i64,
    pub ledger: Rc<ArrivalsByPlanet>,
    pub baselines: HashMap<i64, PlanetTimeline>,
    /// Turn at which a planet leaves the board, populated only for planets
    /// that expire within `horizon` (i.e. comets near end of life). Missing
    /// entry means the planet survives the entire window.
    pub expiry_at: HashMap<i64, i64>,
}

impl TimelineCache {
    /// Build the cache by forking `parent` and walking forward `horizon` turns
    /// to collect the in-flight arrival ledger, then computing per-player
    /// baselines. `O(horizon * |planets|)`.
    pub fn build(
        parent: &Simulator,
        player: i64,
        horizon: i64,
        entity_cache: &EntityCache,
    ) -> Self {
        let ledger = ArrivalLedger::build(parent, horizon, entity_cache);
        Self::from_ledger(parent.planets(), player, &ledger)
    }

    /// Build the cache from a precomputed [`ArrivalLedger`], paying only for
    /// the per-player baseline pass (`O(horizon * |planets|)` but with a much
    /// smaller constant than `build` — no simulation forwarding). Use this when the
    /// same ledger is shared across multiple players (rollout reactive turns).
    ///
    /// `planets` must match the snapshot the ledger was built from.
    pub fn from_ledger(planets: &[Planet], player: i64, ledger: &ArrivalLedger) -> Self {
        let horizon = ledger.horizon;
        let mut baselines =
            HashMap::with_capacity_and_hasher(planets.len(), Default::default());
        for planet in planets {
            // Reuse the ledger's shared trajectory when available (the common
            // path); fall back to a full re-simulation only for a planet the
            // ledger never saw.
            let tl = match ledger.trajectory(planet.id) {
                Some(traj) => finish_timeline(planet, player, horizon, traj),
                None => {
                    let arrivals = ledger.arrivals(planet.id);
                    let expiry = ledger.expiry(planet.id);
                    simulate_planet_timeline(planet, arrivals, player, horizon, expiry)
                }
            };
            baselines.insert(planet.id, tl);
        }
        Self {
            player,
            horizon,
            ledger: Rc::clone(&ledger.ledger),
            baselines,
            expiry_at: ledger.expiry_at.clone(),
        }
    }

    /// Arrival list for a planet (empty if nothing is incoming or the planet
    /// isn't in the cache).
    pub fn arrivals(&self, planet_id: i64) -> &[ArrivalEvent] {
        self.ledger
            .get(&planet_id)
            .map(|v| v.as_slice())
            .unwrap_or(&[])
    }

    /// Baseline timeline for a planet, or `None` if the planet wasn't present
    /// when the cache was built.
    pub fn baseline(&self, planet_id: i64) -> Option<&PlanetTimeline> {
        self.baselines.get(&planet_id)
    }

    /// Turn at which a planet leaves the board, if within the cache's horizon.
    #[inline]
    pub fn expiry(&self, planet_id: i64) -> Option<i64> {
        self.expiry_at.get(&planet_id).copied()
    }
}

/// Returns the planet's expiry turn iff it falls within `horizon`. Static and
/// orbiting planets last the whole game, so they always return `None`.
fn expiry_within_horizon(
    entity_cache: &EntityCache,
    planet_id: i64,
    horizon: i64,
) -> Option<i64> {
    let life = entity_cache.remaining_life(planet_id);
    if life <= horizon { Some(life) } else { None }
}

/// Smallest ship count that, if it lands on `planet` at `arrival_turn` for
/// `attacker_owner`, makes them own the planet by `eval_turn`. Returns 0 if
/// the planet is already theirs at `eval_turn` without extras. Returns
/// `upper_bound + 1` when not achievable within budget.
///
/// `extras` is a slice of additional arrivals to incorporate alongside the
/// timeline cache's in-flight arrivals (e.g. planned commitments from the
/// current planning turn). When empty, the cache's pre-built baseline is
/// reused as the checkpoint — `O(eval_turn - arrival_turn)` per query.
/// When non-empty, a "pre-attacker baseline" is simulated once that already
/// incorporates the extras, then queries checkpoint off it the same way.
///
/// `eval_turn` is clamped to `cache.horizon`; if `arrival_turn > eval_turn`
/// after clamping, returns `upper_bound + 1`.
pub fn min_ships_to_own_by(
    cache: &TimelineCache,
    planet: &Planet,
    attacker_owner: i64,
    arrival_turn: i64,
    eval_turn: i64,
    upper_bound: i64,
    extras: &[ArrivalEvent],
) -> i64 {
    let arrival_turn = arrival_turn.max(1);
    let eval_turn = eval_turn.max(1).min(cache.horizon);
    if arrival_turn > eval_turn {
        return upper_bound + 1;
    }

    let base_arrivals = cache.arrivals(planet.id);
    let expiry = cache.expiry(planet.id);

    // When no extras, the cache's pre-built baseline already reflects all
    // in-flight arrivals. Otherwise simulate a one-shot "with-extras" baseline
    // and use that as the checkpoint.
    let local_baseline: Option<PlanetTimeline> = if extras.is_empty() {
        None
    } else {
        let mut merged = Vec::with_capacity(base_arrivals.len() + extras.len());
        merged.extend_from_slice(base_arrivals);
        merged.extend_from_slice(extras);
        Some(simulate_planet_timeline(planet, &merged, attacker_owner, eval_turn, expiry))
    };
    let baseline: &PlanetTimeline = local_baseline
        .as_ref()
        .or_else(|| cache.baseline(planet.id))
        .expect("planet must be in the timeline cache");

    if state_at_timeline(baseline, eval_turn).0 == attacker_owner {
        return 0;
    }

    let mut scratch: Vec<ArrivalEvent> =
        Vec::with_capacity(base_arrivals.len() + extras.len() + 1);
    scratch.extend_from_slice(base_arrivals);
    scratch.extend_from_slice(extras);
    scratch.push(ArrivalEvent {
        turns: arrival_turn,
        owner: attacker_owner,
        ships: 0,
    });
    let last = scratch.len() - 1;
    let eval_idx = eval_turn.clamp(0, baseline.horizon) as usize;

    // Buffers reused across every binary-search probe (see `simulate_checkpoint_into`).
    let mut owner_buf: Vec<i64> = Vec::new();
    let mut ships_buf: Vec<i64> = Vec::new();
    let mut by_turn_buf: Vec<Vec<ArrivalEvent>> = Vec::new();

    let mut owns_at = |ships: i64| -> bool {
        scratch[last].ships = ships;
        simulate_checkpoint_into(
            planet, baseline, arrival_turn, &scratch, expiry,
            &mut owner_buf, &mut ships_buf, &mut by_turn_buf,
        );
        owner_buf[eval_idx] == attacker_owner
    };

    let hi_init = upper_bound.max(1);
    if !owns_at(hi_init) {
        return hi_init + 1;
    }
    let (mut lo, mut hi) = (1i64, hi_init);
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        if owns_at(mid) {
            hi = mid;
        } else {
            lo = mid + 1;
        }
    }
    lo
}

/// Smallest reinforcement that arrives at `arrival_turn` and keeps
/// `cache.player` in continuous ownership through `hold_until`. If the planet
/// isn't currently `cache.player`'s, collapses to `min_ships_to_own_by` at
/// `hold_until`. `extras` works the same as in `min_ships_to_own_by`.
/// Returns `upper_bound + 1` if no value in `1..=upper_bound` works.
pub fn reinforcement_needed_to_hold_until(
    cache: &TimelineCache,
    planet: &Planet,
    arrival_turn: i64,
    hold_until: i64,
    upper_bound: i64,
    extras: &[ArrivalEvent],
) -> i64 {
    let player = cache.player;
    let arrival_turn = arrival_turn.max(1);
    let hold_until = hold_until.max(arrival_turn).min(cache.horizon);

    if planet.owner != player {
        return min_ships_to_own_by(
            cache,
            planet,
            player,
            arrival_turn,
            hold_until,
            upper_bound,
            extras,
        );
    }

    let base_arrivals = cache.arrivals(planet.id);
    let expiry = cache.expiry(planet.id);

    let local_baseline: Option<PlanetTimeline> = if extras.is_empty() {
        None
    } else {
        let mut merged = Vec::with_capacity(base_arrivals.len() + extras.len());
        merged.extend_from_slice(base_arrivals);
        merged.extend_from_slice(extras);
        Some(simulate_planet_timeline(planet, &merged, player, hold_until, expiry))
    };
    let baseline: &PlanetTimeline = local_baseline
        .as_ref()
        .or_else(|| cache.baseline(planet.id))
        .expect("planet must be in the timeline cache");

    let mut scratch: Vec<ArrivalEvent> =
        Vec::with_capacity(base_arrivals.len() + extras.len() + 1);
    scratch.extend_from_slice(base_arrivals);
    scratch.extend_from_slice(extras);
    scratch.push(ArrivalEvent {
        turns: arrival_turn,
        owner: player,
        ships: 0,
    });
    let last = scratch.len() - 1;

    // Buffers reused across every binary-search probe (see `simulate_checkpoint_into`).
    let mut owner_buf: Vec<i64> = Vec::new();
    let mut ships_buf: Vec<i64> = Vec::new();
    let mut by_turn_buf: Vec<Vec<ArrivalEvent>> = Vec::new();

    let mut holds = |ships: i64| -> bool {
        scratch[last].ships = ships;
        simulate_checkpoint_into(
            planet, baseline, arrival_turn, &scratch, expiry,
            &mut owner_buf, &mut ships_buf, &mut by_turn_buf,
        );
        (arrival_turn..=hold_until).all(|t| owner_buf[t as usize] == player)
    };

    let hi_init = upper_bound.max(1);
    if !holds(hi_init) {
        return hi_init + 1;
    }
    let (mut lo, mut hi) = (1i64, hi_init);
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        if holds(mid) {
            hi = mid;
        } else {
            lo = mid + 1;
        }
    }
    lo
}
