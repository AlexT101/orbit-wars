#![allow(dead_code)]

use std::sync::LazyLock;

// ── RUNTIME AGENT CONFIG (config.json) ──────────────────────────────────────
// The *agent* (strategy) constants below are loaded once, at first access, from
// a JSON file so the tuning loop can try new values WITHOUT recompiling the
// Rust extension. Only `bots/mine/apollo` does this; the frozen baseline clone
// keeps its compile-time constants.
//
// Path resolution:
//   * `APOLLO_CONFIG` env var if set (absolute path), else
//   * `<crate dir>/config.json` (CARGO_MANIFEST_DIR is baked in at build time,
//     so it is stable regardless of the worker process CWD).
//
// A missing or malformed file PANICS rather than silently falling back, so a
// tuning run can never quietly measure the wrong constants.
//
// ENGINE constants (game rules / physics) are deliberately NOT in config.json:
// they mirror the engine and must never drift. The excluded agent constants
// (early-game pre-pass, REACTIVE_TURNS, AIM_HORIZON, the cone/nudge sim knobs,
// and the Config source caps) are likewise kept compile-time for now; promote
// them to config.json the same way if they ever need tuning.

static AGENT_CONFIG: LazyLock<serde_json::Value> = LazyLock::new(|| {
    let path = std::env::var_os("APOLLO_CONFIG")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("config.json"));
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| {
        panic!("apollo: failed to read agent config at {}: {e}", path.display())
    });
    serde_json::from_str(&text)
        .unwrap_or_else(|e| panic!("apollo: invalid JSON in {}: {e}", path.display()))
});

fn cfg_i64(key: &str) -> i64 {
    AGENT_CONFIG
        .get(key)
        .and_then(serde_json::Value::as_i64)
        .unwrap_or_else(|| panic!("apollo: config.json missing integer key '{key}'"))
}

fn cfg_f64(key: &str) -> f64 {
    AGENT_CONFIG
        .get(key)
        .and_then(serde_json::Value::as_f64)
        .unwrap_or_else(|| panic!("apollo: config.json missing number key '{key}'"))
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

// Turn rules — TUNABLE: loaded from config.json (see top of file).
// Each is a `LazyLock<T>`; read with `*NAME` at the call site.
pub static ROTATION_LOOK_AHEAD_TURNS: LazyLock<i64> = LazyLock::new(|| cfg_i64("rotation_look_ahead_turns")); // Number of turns to look ahead when estimating future position of planets
pub static OFFSET_LOOKAHEAD: LazyLock<i64> = LazyLock::new(|| cfg_i64("offset_lookahead")); // Max per-source launch delay considered by attack planning and reinforcement hold checks. Offset 0 emits now; delayed attack offsets become reservations so later choices cannot spend those ships.
pub static ENEMY_OFFSET_LOOKAHEAD: LazyLock<i64> = LazyLock::new(|| cfg_i64("enemy_offset_lookahead")); // Max enemy launch delay considered when estimating reinforcement pressure.
pub static REINFORCEMENT_PRESSURE_TURNS: LazyLock<i64> = LazyLock::new(|| cfg_i64("reinforcement_pressure_turns")); // Enemy planets within this many turns contribute to reinforcement pressure.
pub static REINFORCEMENT_PRESSURE_DECAY: LazyLock<f64> = LazyLock::new(|| cfg_f64("reinforcement_pressure_decay")); // Enemy pressure multiplier at REINFORCEMENT_PRESSURE_TURNS; turns 0/1 contribute fully.
pub static FRONTIER_PRESSURE_RATIO: LazyLock<f64> = LazyLock::new(|| cfg_f64("frontier_pressure_ratio")); // Frontier planets only reinforce when the pressure sink is at least this much higher-pressure.
pub static ALLY_PRESSURE_RATIO: LazyLock<f64> = LazyLock::new(|| cfg_f64("ally_pressure_ratio")); // Enemy targets are only attacked when our pressure on them is at least this fraction of the enemy pressure on them.

// Early-game expansion pre-pass (see early_game.rs)
pub const EARLY_GAME_END: i64 = 0; // The DFS expansion pre-pass runs on steps [0, EARLY_GAME_END). No valuation cliff (each plan's objective extends to the full horizon and greedy always runs on top), but it is a hard stop on chain re-derivation: chains whose later hops would launch at/after this step are handed to the (chain-unaware) greedy planner. See early_game.rs.
pub const EARLY_GAME_MAX_CANDIDATES: usize = 10; // Capture targets kept by earliest probe arrival; EARLY_GAME_VALUE_PICKS more are unioned in by value bound.
pub const EARLY_GAME_VALUE_PICKS: usize = 5; // Reachable neutrals with the highest value bound (production·(window − earliest arrival) − garrison) unioned into the candidate set regardless of arrival rank.
pub const EARLY_GAME_MAX_CHILD_FUND: usize = 4; // Per target, highest-production remaining neutrals considered for the min+child funding variant.
pub const EARLY_GAME_NODE_BUDGET: u64 = 50_000; // Hard cap on early-game DFS nodes; best plan found so far is kept on exhaustion.
pub const EARLY_GAME_PROBE_SHIPS: i64 = 1000; // Upper clamp on the reachability probe fleet — fleet speed saturates at 1000 ships, so a larger probe can't arrive earlier. The probe itself is sized from exact achievable ships (owned + producible over the window).
pub const EARLY_GAME_FERRY_PROBES: usize = 8; // Max launch offsets probed per (source, target) for the ferry variant each node (plan-dependent ship counts bypass the geometry row cache).

pub const REACTIVE_TURNS: i64 = 2; // Number of turns to forward simulate ally/enemy steps during rollouts

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

// `horizon` and `max_distance` are TUNABLE (config.json); the source caps are
// held fixed for now (see top-of-file note).
static CONFIG_2P: LazyLock<Config> = LazyLock::new(|| Config {
    horizon: cfg_i64("horizon"),
    max_distance: cfg_f64("max_distance"),
    max_sources_to_consider: 16,
    max_sources: 4,
});

const CONFIG_4P: Config = Config {
    horizon: 30,
    max_distance: 38.0,
    max_sources_to_consider: 16,
    max_sources: 4,
};

impl Config {
    #[inline]
    pub fn for_alive(alive: usize) -> Config {
        if alive >= 3 {
            CONFIG_4P
        } else {
            *CONFIG_2P
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
