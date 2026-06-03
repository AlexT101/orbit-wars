#![allow(dead_code)]

use std::rc::Rc;

use rustc_hash::FxHashMap as HashMap;

use crate::apollo::constants::{EPISODE_STEPS, MAX_PLAYERS};

use crate::apollo::aim;
pub use crate::apollo::aim::AimResult;
use crate::apollo::cache::EntityCache;
pub use crate::apollo::engine::ArrivalEvent;
use crate::apollo::engine::Simulator;
use crate::apollo::engine::{Fleet, Planet};

// ── Basic Helpers ────────────────────────────────────────────────────

#[inline]
pub fn dist(ax: f64, ay: f64, bx: f64, by: f64) -> f64 {
    crate::apollo::engine::distance((ax, ay), (bx, by))
}

/// Public aim entry point. Delegates to the parametric blocker pipeline in
/// [`crate::apollo::aim`]: lead the target with Newton iteration, then reject
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
    aim::aim_with_prediction(cache, shooter_id, target_id, ships, launch_turn_offset)
}

/// Comet-free aim used to compute the turn-invariant base for the invariant-aim
/// cache; see [`crate::apollo::aim::aim_ignoring_comets`].
#[inline]
pub fn aim_ignoring_comets(
    cache: &EntityCache,
    shooter_id: i64,
    target_id: i64,
    ships: i64,
    launch_turn_offset: i64,
) -> Option<AimResult> {
    aim::aim_ignoring_comets(cache, shooter_id, target_id, ships, launch_turn_offset)
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
        if p.owner > max_owner {
            max_owner = p.owner;
        }
    }
    for f in fleets {
        if f.owner > max_owner {
            max_owner = f.owner;
        }
    }
    if max_owner < 0 {
        return 2;
    }
    let n = (max_owner as usize + 1).min(MAX_PLAYERS);
    n.max(2)
}

// ── Timeline Helpers ──────────────────────────────────────────────────────────
// Forward simulation with initial timeline and hypothetical queries

/// Per-planet arrival ledger: `{planet_id → [ArrivalEvent, ...]}`.
/// Built once per rollout step via [`ArrivalLedger::build`].
pub type ArrivalsByPlanet = HashMap<i64, Vec<ArrivalEvent>>;

/// Same-turn combat resolution for the timeline projection. Aggregates the
/// arrival list into per-player ship totals and defers to the shared
/// [`crate::apollo::engine::resolve_combat`] rule (the same one the forward
/// [`Simulator`] uses), so the projection can never silently drift from the
/// simulator's combat. Neutral (`-1`) arrivals are ignored — real fleets always
/// have an owner `>= 0`.
pub fn resolve_arrival_event(owner: i64, garrison: i64, arrivals: &[ArrivalEvent]) -> (i64, i64) {
    let mut incoming = [0i64; MAX_PLAYERS];
    for ev in arrivals {
        if ev.owner >= 0 && (ev.owner as usize) < MAX_PLAYERS {
            incoming[ev.owner as usize] += ev.ships;
        }
    }
    crate::apollo::engine::resolve_combat(owner, garrison, &incoming)
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
    /// Forward (suffix) minimum of `ships_at` within the run of turns we
    /// continuously own starting at each turn `t`: `owned_suffix_min[t] =
    /// min{ ships_at[u] : u ≥ t, owned by `player` continuously from t }`, and
    /// `0` at turns we don't own. This is the maximum a source can ship out at
    /// launch offset `t` without driving any later owned turn negative —
    /// withdrawing ships at `t` removes them from every turn `≥ t`.
    /// Player-specific.
    pub owned_suffix_min: Rc<Vec<i64>>,
    pub horizon: i64,
}

/// Player-agnostic forward trajectory of one planet: the per-turn `owner_at` /
/// `ships_at` arrays. Shared (`Rc`) so baselines for different players reuse one
/// allocation. See [`build_trajectory`] / [`finish_timeline`].
#[derive(Debug, Clone)]
pub struct Trajectory {
    owner_at: Rc<Vec<i64>>,
    ships_at: Rc<Vec<i64>>,
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
    }
}

/// Derive the player-specific metrics for `player` from a shared
/// [`Trajectory`], producing a [`PlanetTimeline`] that shares the trajectory's
/// `owner_at`/`ships_at` arrays plus the per-turn `owned_suffix_min`.
pub fn finish_timeline(player: i64, horizon: i64, traj: &Trajectory) -> PlanetTimeline {
    let owner_at = &traj.owner_at;
    let ships_at = &traj.ships_at;

    // Forward-min within each continuously-owned run, in one backward sweep:
    // owned turns accumulate the running min of `ships_at`; a non-owned turn
    // resets the accumulator so the gap breaks continuity for earlier turns.
    let len = owner_at.len();
    let mut owned_suffix_min = vec![0i64; len];
    let mut acc = i64::MAX;
    for t in (0..len).rev() {
        if owner_at[t] == player {
            acc = acc.min(ships_at[t].max(0));
            owned_suffix_min[t] = acc;
        } else {
            acc = i64::MAX;
            owned_suffix_min[t] = 0;
        }
    }

    PlanetTimeline {
        owner_at: Rc::clone(&traj.owner_at),
        ships_at: Rc::clone(&traj.ships_at),
        owned_suffix_min: Rc::new(owned_suffix_min),
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
    finish_timeline(player, horizon.max(0), &traj)
}

/// Maximum ships withdrawable at launch `offset` without driving any later
/// owned turn negative — the forward-min of `ships_at` over the owned run
/// starting at `offset` (see [`PlanetTimeline::owned_suffix_min`]). Returns 0
/// at offsets the planet isn't owned by `player`. Clamps `offset` into range.
pub fn available_at_timeline(timeline: &PlanetTimeline, offset: i64) -> i64 {
    let turn = offset.max(0).min(timeline.horizon) as usize;
    timeline.owned_suffix_min[turn]
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
    pub fn build(parent: &Simulator, horizon: i64, cache: &EntityCache) -> Self {
        // Clamp the look-ahead to the game end.
        let horizon = horizon.min((EPISODE_STEPS - parent.step_count()).max(1));
        let mut sim = parent.fork();
        sim.step_n(horizon, Some(cache));
        let mut ledger = sim.collect_arrivals();
        let mut expiry_at: HashMap<i64, i64> = HashMap::default();
        let mut trajectories: HashMap<i64, Trajectory> =
            HashMap::with_capacity_and_hasher(parent.planets().len(), Default::default());
        for planet in parent.planets() {
            ledger.entry(planet.id).or_default();
            let expiry = expiry_within_horizon(cache, planet.id, horizon);
            if let Some(exp) = expiry {
                expiry_at.insert(planet.id, exp);
            }
            let arrivals = ledger.get(&planet.id).map(|v| v.as_slice()).unwrap_or(&[]);
            trajectories.insert(
                planet.id,
                build_trajectory(planet, arrivals, horizon, expiry),
            );
        }
        Self {
            horizon,
            ledger: Rc::new(ledger),
            expiry_at,
            trajectories,
        }
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

/// Per-player baseline timelines built from a shared [`ArrivalLedger`].
///
/// Typical use: build the player-agnostic ledger once per rollout step, then
/// call [`TimelineCache::from_ledger`] for each player that needs baseline
/// ownership/availability reads. Hypothetical-arrival queries can reuse the
/// cached arrivals and expiry map while simulating only the touched planet.
#[derive(Debug, Clone)]
pub struct TimelineCache {
    pub horizon: i64,
    pub ledger: Rc<ArrivalsByPlanet>,
    pub baselines: HashMap<i64, PlanetTimeline>,
    /// Turn at which a planet leaves the board, populated only for planets
    /// that expire within `horizon` (i.e. comets near end of life). Missing
    /// entry means the planet survives the entire window.
    pub expiry_at: HashMap<i64, i64>,
}

impl TimelineCache {
    /// Build the cache from a precomputed [`ArrivalLedger`], paying only for
    /// the per-player baseline pass (`O(horizon * |planets|)` but with a much
    /// smaller constant than [`ArrivalLedger::build`] — no simulation
    /// forwarding). Use this when the same ledger is shared across multiple
    /// players (rollout reactive turns).
    ///
    /// `planets` must match the snapshot the ledger was built from.
    pub fn from_ledger(planets: &[Planet], player: i64, ledger: &ArrivalLedger) -> Self {
        let horizon = ledger.horizon;
        let mut baselines = HashMap::with_capacity_and_hasher(planets.len(), Default::default());
        for planet in planets {
            // Reuse the ledger's shared trajectory when available (the common
            // path); fall back to a full re-simulation only for a planet the
            // ledger never saw.
            let tl = match ledger.trajectory(planet.id) {
                Some(traj) => finish_timeline(player, horizon, traj),
                None => {
                    let arrivals = ledger.arrivals(planet.id);
                    let expiry = ledger.expiry(planet.id);
                    simulate_planet_timeline(planet, arrivals, player, horizon, expiry)
                }
            };
            baselines.insert(planet.id, tl);
        }
        Self {
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

/// Returns the planet's expiry turn iff it actually leaves the board within
/// `horizon`. `off_board_turn == EPISODE_STEPS` is the "lasts the whole game"
/// sentinel (all static/orbiting planets, plus comets that never leave), so
/// those return `None`. Game-end is not an expiry — otherwise every planet
/// would be zeroed at the evaluation horizon in the final `HORIZON` turns.
fn expiry_within_horizon(cache: &EntityCache, planet_id: i64, horizon: i64) -> Option<i64> {
    let entity = cache.get(planet_id)?;
    if entity.off_board_turn >= EPISODE_STEPS {
        return None;
    }
    let life = (entity.off_board_turn - cache.current_turn).max(0);
    (life <= horizon).then_some(life)
}
