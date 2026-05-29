#![allow(dead_code)]

// ── ENGINE CONSTANTS ────────────────────────────────────────────────────────
// These are constants used by `engine.rs` to implement the game rules and physics and should never be changed

// Game rules
pub const BOARD_SIZE: f64 = 100.0;
pub const CENTER: f64 = BOARD_SIZE / 2.0;                           // Sun / orbital center.
pub const EPISODE_STEPS: i64 = 500;
pub const MAX_PLAYERS: usize = 4;
pub const TOTAL_OVERAGE_TIME: f64 = 60.0;                               // Total overage time (in seconds).

// Physics rules
pub const SUN_RADIUS: f64 = 10.0;                                   // Radius of sun's destruction zone.
pub const MAX_SHIP_SPEED: f64 = 6.0;                                // Maximum fleet speed (reached at ~1000 ships).
pub const LAUNCH_CLEARANCE: f64 = 0.1;                              // Fleets spawn at `planet_radius + LAUNCH_CLEARANCE` from the planet center.
pub const ROTATION_LIMIT: f64 = 50.0;                               // `orbital_radius + planet_radius >= ROTATION_LIMIT` → planet is static.
pub const COMET_RADIUS: f64 = 1.0;
pub const COMET_PRODUCTION: i64 = 1;
pub const COMET_SPEED: f64 = 4.0;

// Generation rules
pub const ANG_VEL_MIN: f64 = 0.025;                                 // Orbital angular velocities sampled uniformly from [ANG_VEL_MIN, ANG_VEL_MAX] at reset.
pub const ANG_VEL_MAX: f64 = 0.05;
pub const PLANET_CLEARANCE: f64 = 7.0;                              // Minimum gap required between adjacent planets at generation time.
pub const MIN_PLANET_GROUPS: i64 = 5;
pub const MAX_PLANET_GROUPS: i64 = 10;
pub const MIN_STATIC_GROUPS: i64 = 3;
pub const COMET_SPAWN_STEPS: [i64; 5] = [50, 150, 250, 350, 450];   // Game steps on which a new comet group spawns.


// ── SIMULATION CONSTANTS ────────────────────────────────────────────────────────
// Specific to our bot for internal decisions

pub const HORIZON: i64 = 30;                                        // Maximum turns to look into the future.
