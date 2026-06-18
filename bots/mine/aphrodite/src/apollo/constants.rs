#![allow(dead_code)]

use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::LazyLock;

// ── RUNTIME AGENT CONFIG (config.json / config_4p.json) ─────────────────────
// The *agent* (strategy) constants below are parsed once, at first access, from
// JSON so tuning can try new values WITHOUT recompiling. There are TWO configs,
// selected per turn by player count: `config.json` (2p) and `config_4p.json`
// (4p). The frozen baseline clone keeps its compile-time constants instead.
//
// MODE is set once per turn from the live player count (`set_mode_for_alive`,
// called from lib.rs at the start of each move). A 4p game that collapses to
// 1v1 switches back to the 2p config for its endgame. Each read is a relaxed
// atomic load + array index + field read (negligible; these are not in the
// simulation hot loop).
//
// Path resolution per file: `APOLLO_CONFIG` / `APOLLO_CONFIG_4P` env var if set,
// else `<crate dir>/config.json` / `config_4p.json` (CARGO_MANIFEST_DIR is baked
// in at build time, stable regardless of worker CWD). A missing/malformed file
// PANICS rather than silently falling back, so a run can never quietly measure
// the wrong constants.
//
// ENGINE constants (game rules / physics) are deliberately NOT configurable.
// Excluded agent constants (early-game pre-pass, REACTIVE_TURNS, AIM_HORIZON,
// the cone/nudge sim knobs, and the Config source caps) stay compile-time.

// 0 = 2p (config.json), 1 = 4p (config_4p.json).
static MODE: AtomicU8 = AtomicU8::new(0);

/// Select the active config for the rest of the turn from the live player count.
/// Call once at the start of each move, before strategy runs.
pub fn set_mode_for_alive(alive: usize) {
    MODE.store(u8::from(alive >= 3), Ordering::Relaxed);
}

#[inline]
fn agent() -> &'static AgentConsts {
    &AGENT[MODE.load(Ordering::Relaxed) as usize]
}

/// Tunable agent constants, parsed once per config file at first access.
#[derive(Clone, Copy)]
struct AgentConsts {
    rotation_look_ahead_turns: i64,
    offset_lookahead: i64,
    enemy_offset_lookahead: i64,
    reinforcement_pressure_turns: i64,
    reinforcement_pressure_decay: f64,
    frontier_pressure_ratio: f64,
    ally_pressure_ratio: f64,
    horizon: i64,
    max_distance: f64,
    // ── Scoring / valuation (phase-2 tunables) ──────────────────────────────
    // `timeline_delta_score` = w_production·Σ production·Δowner
    //                          + w_final_ships·Δsigned_final_ships
    //                          − w_ship_cost·ships_committed.
    // `score_w_production` is PINNED at 1.0 (scale anchor: the plan score is
    // argmax-compared, so multiplying all three weights is a no-op). Defaults
    // below reproduce the pre-phase-2 behavior exactly.
    score_w_production: f64,
    score_w_ship_cost: f64,
    score_w_final_ships: f64,
    score_per_ship_smoothing: f64,
    capture_min_score: f64,
    score_enemy_capture_bonus: f64,
    default_strategy: i64,
    // ── Neutral-capture discipline (phase-3 tunables) ───────────────────────
    // Discourage sinking ships into slow-payback / marginal NEUTRAL captures.
    // All default to no-ops (the two penalties = 0) so play is unchanged until
    // tuned. See tuning/PHASE3_DESIGN.md.
    neutral_payback_turns: f64,
    neutral_payback_penalty: f64,
    lead_gate: f64,
    neutral_capture_penalty: f64,
}

// [0] = 2p, [1] = 4p.
static AGENT: LazyLock<[AgentConsts; 2]> = LazyLock::new(|| {
    [
        parse_consts("APOLLO_CONFIG", "config.json"),
        parse_consts("APOLLO_CONFIG_4P", "config_4p.json"),
    ]
});

fn parse_consts(env_key: &str, default_name: &str) -> AgentConsts {
    let path = std::env::var_os(env_key)
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(default_name));
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| {
        panic!(
            "apollo: failed to read agent config at {}: {e}",
            path.display()
        )
    });
    let v: serde_json::Value = serde_json::from_str(&text)
        .unwrap_or_else(|e| panic!("apollo: invalid JSON in {}: {e}", path.display()));
    let pname = path.display().to_string();
    let i = |k: &str| -> i64 {
        v.get(k)
            .and_then(serde_json::Value::as_i64)
            .unwrap_or_else(|| panic!("apollo: {pname} missing integer key '{k}'"))
    };
    let f = |k: &str| -> f64 {
        v.get(k)
            .and_then(serde_json::Value::as_f64)
            .unwrap_or_else(|| panic!("apollo: {pname} missing number key '{k}'"))
    };
    AgentConsts {
        rotation_look_ahead_turns: i("rotation_look_ahead_turns"),
        offset_lookahead: i("offset_lookahead"),
        enemy_offset_lookahead: i("enemy_offset_lookahead"),
        reinforcement_pressure_turns: i("reinforcement_pressure_turns"),
        reinforcement_pressure_decay: f("reinforcement_pressure_decay"),
        frontier_pressure_ratio: f("frontier_pressure_ratio"),
        ally_pressure_ratio: f("ally_pressure_ratio"),
        horizon: i("horizon"),
        max_distance: f("max_distance"),
        score_w_production: f("score_w_production"),
        score_w_ship_cost: f("score_w_ship_cost"),
        score_w_final_ships: f("score_w_final_ships"),
        score_per_ship_smoothing: f("score_per_ship_smoothing"),
        capture_min_score: f("capture_min_score"),
        score_enemy_capture_bonus: f("score_enemy_capture_bonus"),
        default_strategy: i("default_strategy"),
        neutral_payback_turns: f("neutral_payback_turns"),
        neutral_payback_penalty: f("neutral_payback_penalty"),
        lead_gate: f("lead_gate"),
        neutral_capture_penalty: f("neutral_capture_penalty"),
    }
}

// ── ENGINE CONSTANTS ────────────────────────────────────────────────────────
// These are constants used by `engine.rs` to implement the game rules and physics and should never be changed

// Game rules
pub const BOARD_SIZE: f64 = 100.0;
pub const CENTER: f64 = BOARD_SIZE / 2.0; // Sun / orbital center.
pub const EPISODE_STEPS: i64 = 500;
pub const MAX_PLAYERS: usize = 4;
pub const TOTAL_OVERAGE_TIME: f64 = 60.0; // Total overage time (in seconds).

// Physics rules
pub const SUN_RADIUS: f64 = 10.0; // Radius of sun's destruction zone.
pub const MAX_SHIP_SPEED: f64 = 6.0; // Maximum fleet speed (reached at ~1000 ships).
pub const LAUNCH_CLEARANCE: f64 = 0.1; // Fleets spawn at `planet_radius + LAUNCH_CLEARANCE` from the planet center.
pub const ROTATION_LIMIT: f64 = 50.0; // `orbital_radius + planet_radius >= ROTATION_LIMIT` → planet is static.
pub const COMET_RADIUS: f64 = 1.0;
pub const COMET_PRODUCTION: i64 = 1;
pub const COMET_SPEED: f64 = 4.0;

// Generation rules
pub const ANG_VEL_MIN: f64 = 0.025; // Orbital angular velocities sampled uniformly from [ANG_VEL_MIN, ANG_VEL_MAX] at reset.
pub const ANG_VEL_MAX: f64 = 0.05;
pub const PLANET_CLEARANCE: f64 = 7.0; // Minimum gap required between adjacent planets at generation time.
pub const MIN_PLANET_GROUPS: i64 = 5;
pub const MAX_PLANET_GROUPS: i64 = 10;
pub const MIN_STATIC_GROUPS: i64 = 3;
pub const COMET_SPAWN_STEPS: [i64; 5] = [50, 150, 250, 350, 450]; // Game steps on which a new comet group spawns.

// ── AGENT CONSTANTS ────────────────────────────────────────────────────────
// Specific to our bot for internal decisions

// Turn rules — TUNABLE (config.json / config_4p.json), selected by MODE.
// Read with `name()` at the call site.
#[inline]
pub fn rotation_look_ahead_turns() -> i64 {
    agent().rotation_look_ahead_turns
} // Number of turns to look ahead when estimating future position of planets
#[inline]
pub fn offset_lookahead() -> i64 {
    agent().offset_lookahead
} // Max per-source launch delay considered by attack planning and reinforcement hold checks. Offset 0 emits now; delayed attack offsets become reservations so later choices cannot spend those ships.
#[inline]
pub fn enemy_offset_lookahead() -> i64 {
    agent().enemy_offset_lookahead
} // Max enemy launch delay considered when estimating reinforcement pressure.
#[inline]
pub fn reinforcement_pressure_turns() -> i64 {
    agent().reinforcement_pressure_turns
} // Enemy planets within this many turns contribute to reinforcement pressure.
#[inline]
pub fn reinforcement_pressure_decay() -> f64 {
    agent().reinforcement_pressure_decay
} // Enemy pressure multiplier at REINFORCEMENT_PRESSURE_TURNS; turns 0/1 contribute fully.
#[inline]
pub fn frontier_pressure_ratio() -> f64 {
    agent().frontier_pressure_ratio
} // Frontier planets only reinforce when the pressure sink is at least this much higher-pressure.
#[inline]
pub fn ally_pressure_ratio() -> f64 {
    agent().ally_pressure_ratio
} // Enemy targets are only attacked when our pressure on them is at least this fraction of the enemy pressure on them.

// Scoring / valuation — TUNABLE (phase 2). See AgentConsts for the formula.
#[inline]
pub fn score_w_production() -> f64 {
    agent().score_w_production
} // PINNED at 1.0 (scale anchor for the argmax-compared plan score).
#[inline]
pub fn score_w_ship_cost() -> f64 {
    agent().score_w_ship_cost
} // Weight on the `− ships_committed` capture-cost term (capture stinginess).
#[inline]
pub fn score_w_final_ships() -> f64 {
    agent().score_w_final_ships
} // Weight on the horizon signed-ship-delta term vs the production-control integral.
#[inline]
pub fn score_per_ship_smoothing() -> f64 {
    agent().score_per_ship_smoothing
} // The additive denominator in the ScorePerShip key `score / (smoothing + ships)`.
#[inline]
pub fn capture_min_score() -> f64 {
    agent().capture_min_score
} // A winning commitment is only admitted when its timeline-delta score exceeds this gate.
#[inline]
pub fn score_enemy_capture_bonus() -> f64 {
    agent().score_enemy_capture_bonus
} // Magnitude of an enemy-owned planet in owner_value (1.0 ⇒ the original symmetric 2:1 enemy-vs-neutral capture value).
#[inline]
pub fn default_strategy() -> i64 {
    agent().default_strategy
} // Reply-policy strategy run directly by plan() and placed first in the search set: 0 = ScorePerShip, 1 = ScoreFirst.

// Neutral-capture discipline — TUNABLE (phase 3). See tuning/PHASE3_DESIGN.md.
#[inline]
pub fn neutral_payback_turns() -> f64 {
    agent().neutral_payback_turns
} // Turns-to-recoup (garrison/production) above which a neutral capture is surcharged.
#[inline]
pub fn neutral_payback_penalty() -> f64 {
    agent().neutral_payback_penalty
} // Surcharge steepness per excess payback-turn; 0 = disabled (no-op).
#[inline]
pub fn lead_gate() -> f64 {
    agent().lead_gate
} // If our ship lead would stay >= this after the buy, waive the payback surcharge.
#[inline]
pub fn neutral_capture_penalty() -> f64 {
    agent().neutral_capture_penalty
} // Flat score penalty on neutral captures (bites marginal neutrals hardest); 0 = no-op.

// Early-game expansion pre-pass (see early_game.rs)
pub const EARLY_GAME_END: i64 = 8; // The DFS expansion pre-pass runs on steps [0, EARLY_GAME_END). No valuation cliff (each plan's objective extends to the full horizon and greedy always runs on top), but it is a hard stop on chain re-derivation: chains whose later hops would launch at/after this step are handed to the (chain-unaware) greedy planner. See early_game.rs.
pub const EARLY_GAME_MAX_CANDIDATES: usize = 10; // Capture targets kept by earliest probe arrival; EARLY_GAME_VALUE_PICKS more are unioned in by value bound.
pub const EARLY_GAME_VALUE_PICKS: usize = 5; // Reachable neutrals with the highest value bound (production·(window − earliest arrival) − garrison) unioned into the candidate set regardless of arrival rank.
pub const EARLY_GAME_MAX_CHILD_FUND: usize = 4; // Per target, highest-production remaining neutrals considered for the min+child funding variant.
pub const EARLY_GAME_NODE_BUDGET: u64 = 50_000; // Hard cap on early-game DFS nodes; best plan found so far is kept on exhaustion.
pub const EARLY_GAME_PROBE_SHIPS: i64 = 1000; // Upper clamp on the reachability probe fleet — fleet speed saturates at 1000 ships, so a larger probe can't arrive earlier. The probe itself is sized from exact achievable ships (owned + producible over the window).
pub const EARLY_GAME_FERRY_PROBES: usize = 8; // Max launch offsets probed per (source, target) for the ferry variant each node (plan-dependent ship counts bypass the geometry row cache).

pub const REACTIVE_TURNS: i64 = 2; // Number of turns to forward simulate ally/enemy steps during rollouts

// `search_candidates_subsets`: number of top-ranked targets whose 2^k include/
// exclude combinations seed the diversified candidate sweep.
pub const SUBSET_TOP_TARGETS: usize = 3;

// Fixed look-ahead used by the aimer when capping a shot's feasible arrival turn
pub const AIM_HORIZON: i64 = 30;

#[derive(Clone, Copy, Debug)]
pub struct Config {
    /// Number of turns to look into the future (rollout/ledger walk length).
    pub horizon: i64,
    /// Maximum distance between planets for us to consider fleet travel.
    pub max_distance: f64,
    /// Upper bound on the number of inbound sources precomputed per target.
    pub max_sources_to_consider: usize,
    /// Upper bound on the number of sources used in a single attack plan.
    pub max_sources: usize,
}

// `horizon` and `max_distance` are TUNABLE per mode (config.json / config_4p.json);
// the source caps are held fixed for now (see top-of-file note). Unlike the loose
// constants (which read the per-turn MODE), `for_alive` selects directly from its
// `alive` argument so the Config stays correct even inside rollouts that simulate
// eliminations.
impl Config {
    #[inline]
    pub fn for_alive(alive: usize) -> Config {
        let a = &AGENT[usize::from(alive >= 3)];
        Config {
            horizon: a.horizon,
            max_distance: a.max_distance,
            max_sources_to_consider: 16,
            max_sources: 4,
        }
    }
}

// Simulation rules
pub const NUDGE_SCAN: i64 = 32; // Baseline number of angle steps per side scanned inside a blocked target's valid aim cone to find an alternate recoverable angle after the direct angle fails.
                                // Coarsest angular step (radians) the cone scan will use. The cone half-width is
                                // the target's *swept-chord* span during the turn (not just its disk radius), so
                                // it can be wide for a fast/long-turn target; the probe count scales up to keep
                                // the step at or below this so a thin hitting window in the widened region isn't
                                // stepped over. ~0.11° — comfortably under the ~0.16° narrowest windows observed.
pub const MAX_CONE_STEP: f64 = 0.002;
// Upper bound on cone-scan probes per side, to cap worst-case cost when the cone
// is very wide (e.g. a near/degenerate target with a huge angular span).
pub const MAX_CONE_PROBES: i64 = 256;

// Nudge scan notes:
// Measured over 30 seeds × 5 ship counts × all shooter/target pairs:
// Of 58,684 blocked-direct shots, only 7,444 (~13%) are nudge-recoverable at all (a clear angle exists in the cone)
// The other ~87% return None at any N. Recovery of those 7,444 vs a thorough n=256 reference:

// NUDGE_SCAN	recovered	cost rel.
// 1	0%	—
// 2	55%	0.08×
// 3	75%	0.13×
// 4	83%	0.17×
// 6	90%	0.25×
// 8	94%	0.33×
// 16	96%	0.67×
// 24	96.6%	1×
// 48	98.7%	2×

// For bot submissions, NUDGE_SCAN should be increased if we have remaining runtime.
// 32 balances test runtime against nudge recovery coverage for local testing.
