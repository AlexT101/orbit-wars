//! Strategy logic

use crate::engine::Planet;

/// Nearest-sniper baseline: for each owned planet, send `garrison + 1` ships
/// at the closest non-owned planet when affordable.
pub fn nearest_sniper(player: i64, planets: &[Planet]) -> Vec<(i64, f64, i64)> {
    let mut moves = Vec::new();
    let mine: Vec<&Planet> = planets.iter().filter(|p| p.owner == player).collect();
    let targets: Vec<&Planet> = planets.iter().filter(|p| p.owner != player).collect();
    if targets.is_empty() {
        return moves;
    }
    for m in &mine {
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
