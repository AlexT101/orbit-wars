//! Strategy entry points exposed to the PyO3 layer. Concrete strategy logic
//! lives in dedicated modules (e.g. [`crate::obnext`]); functions here are
//! thin orchestrators over the strategy-agnostic [`WorldState`].

#![allow(dead_code)]

use crate::engine::Planet;
use crate::world::WorldState;

/// Nearest-sniper baseline: for each owned planet, send `garrison + 1` ships
/// at the closest non-owned planet when affordable.
pub fn nearest_sniper(world: &WorldState) -> Vec<(i64, f64, i64)> {
    let mut moves = Vec::new();
    if world.my_planets.is_empty() {
        return moves;
    }
    let targets: Vec<&Planet> = world
        .enemy_planets
        .iter()
        .chain(world.neutral_planets.iter())
        .collect();
    if targets.is_empty() {
        return moves;
    }
    for m in &world.my_planets {
        let mut nearest: Option<&Planet> = None;
        let mut best = f64::INFINITY;
        for t in &targets {
            let dx = m.x - t.x;
            let dy = m.y - t.y;
            let d = (dx * dx + dy * dy).sqrt();
            if d < best {
                best = d;
                nearest = Some(*t);
            }
        }
        let Some(t) = nearest else { continue };
        let needed = t.ships + 1;
        if m.ships >= needed {
            let angle = (t.y - m.y).atan2(t.x - m.x);
            moves.push((m.id, angle, needed));
        }
    }
    moves
}

pub fn obnext(world: &WorldState) -> Vec<(i64, f64, i64)> {
    crate::obnext::plan(world)
}
