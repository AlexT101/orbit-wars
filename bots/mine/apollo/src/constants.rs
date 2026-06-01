#![allow(dead_code)]

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

// Turn rules
pub const HORIZON: i64 = 30; // Number of turns to look into the future.
pub const ROTATION_LOOK_AHEAD_TURNS: i64 = 10; // Number of turns to look ahead when estimating future position of planets
pub const OFFSET_LOOKAHEAD: i64 = 5; // Max base launch delay swept per target. Offset 0 emits now; winning delayed offsets become reservations so later choices cannot spend those ships.
pub const MAX_COORD_DELAY: i64 = 5; // Max extra launch delay a source may add beyond the subset's base offset while trying to coordinate arrivals near the subset's natural latest arrival.
pub const A_S_LOOKAHEAD: i64 = 3; // Max turns past the natural latest arrival that coordinated schedules may target, letting delayed sources grow extra production before launch.

pub const REACTIVE_TURNS: i64 = 2; // Number of turns to forward simulate ally/enemy steps during rollouts
pub const OPENING_TURNS: i64 = 3; // Number of turns at the start where we focus on economy over combat

// Distance rules
pub const MAX_DISTANCE: f64 = 38.0; // Maximum distance between planets for us to consider fleet travel

// Simulation rules
pub const NUDGE_SCAN: i64 = 16; // Number of angle steps per side scanned inside a blocked target's valid aim cone to find an alternate recoverable angle after the direct angle fails.

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

// For bot submissions, NUDGE_SCAN should be increased if we have remaining runtime
// For local testing, keep NUDGE_SCAN = 16 to balance test runtime with nudge recovery coverage
