use crate::sim::{alive_players, player_score, tick};
use crate::GameState;

const TERMINAL_STEP: i64 = 500;

pub fn evaluate_external(state: &GameState, me: i32) -> f64 {
    let mut s = state.clone();
    for _ in 0..15 {
        if alive_players(&s) <= 1 || s.step >= TERMINAL_STEP {
            break;
        }
        tick(&mut s);
    }
    raw_score_legacy(&s, me)
}

fn raw_score_legacy(state: &GameState, me: i32) -> f64 {
    let my = player_score(state, me) as f64;
    let mut opp = 0.0_f64;
    for p in &state.planets {
        if p.owner != -1 && p.owner != me {
            opp += p.ships as f64;
        }
    }
    for f in &state.fleets {
        if f.owner != me {
            opp += f.ships as f64;
        }
    }
    let my_planets = state.planets.iter().filter(|p| p.owner == me).count() as i64;
    let opp_planets = state
        .planets
        .iter()
        .filter(|p| p.owner != me && p.owner != -1)
        .count() as i64;
    let planet_bonus = (my_planets - opp_planets) as f64 * 5.0;
    let total = my + opp;
    if total < 1.0 {
        return planet_bonus / 10.0;
    }
    ((my - opp) + planet_bonus) / (total + 10.0)
}
