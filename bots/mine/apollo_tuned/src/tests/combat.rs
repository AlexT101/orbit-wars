//! Pins the combat unification: `resolve_arrival_event` now delegates to the
//! shared `engine::resolve_combat` (the same rule the forward `Simulator`
//! uses). `old_resolve_arrival_event` below is a verbatim copy of the
//! pre-unification `HashMap`-based implementation; the new path must reproduce
//! it across the real input domain (fleet owners are always `>= 0`, ships
//! always `> 0`).

use std::collections::HashMap;

use crate::engine::ArrivalEvent;
use crate::helpers::resolve_arrival_event;

/// Verbatim copy of the original `resolve_arrival_event` body.
fn old_resolve_arrival_event(owner: i64, garrison: i64, arrivals: &[ArrivalEvent]) -> (i64, i64) {
    if arrivals.is_empty() {
        return (owner, garrison.max(0));
    }
    let mut by_owner: HashMap<i64, i64> = HashMap::new();
    for ev in arrivals {
        *by_owner.entry(ev.owner).or_insert(0) += ev.ships;
    }
    if by_owner.is_empty() {
        return (owner, garrison.max(0));
    }
    let mut sorted: Vec<(i64, i64)> = by_owner.into_iter().collect();
    // Original sorted only by ships desc; for owner>=0 / ships>0 inputs the top
    // is either unique or a tie (which neutralises), so the owner tie-break
    // never affects the result. Sort by (ships desc, owner asc) here just to
    // make this reference deterministic regardless of HashMap iteration order.
    sorted.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));

    let (top_owner, top_ships) = sorted[0];
    let (survivor_owner, survivor_ships) = if sorted.len() > 1 {
        let second_ships = sorted[1].1;
        if top_ships == second_ships {
            (-1i64, 0i64)
        } else {
            (top_owner, top_ships - second_ships)
        }
    } else {
        (top_owner, top_ships)
    };

    if survivor_ships <= 0 {
        return (owner, garrison.max(0));
    }
    if owner == survivor_owner {
        return (owner, garrison + survivor_ships);
    }
    let new_garrison = garrison - survivor_ships;
    if new_garrison < 0 {
        (survivor_owner, -new_garrison)
    } else {
        (owner, new_garrison)
    }
}

/// Tiny deterministic LCG (no `rand`/`Date`).
struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0 >> 16
    }
    fn range(&mut self, lo: i64, hi: i64) -> i64 {
        lo + (self.next() % ((hi - lo + 1) as u64)) as i64
    }
}

#[test]
fn resolve_arrival_event_matches_pre_unification() {
    let mut rng = Lcg(0xC0FF_EE42);
    let mut cases = 0u64;

    for planet_owner in [-1i64, 0, 1, 2, 3] {
        for &garrison in &[0i64, 1, 7, 50, 400] {
            for _ in 0..4000 {
                // Real domain: fleet owners in 0..4, ships strictly positive.
                let n = rng.range(0, 6) as usize;
                let arrivals: Vec<ArrivalEvent> = (0..n)
                    .map(|_| ArrivalEvent {
                        turns: 1,
                        owner: rng.range(0, 3),
                        ships: rng.range(1, 250),
                    })
                    .collect();

                let got = resolve_arrival_event(planet_owner, garrison, &arrivals);
                let want = old_resolve_arrival_event(planet_owner, garrison, &arrivals);
                assert_eq!(
                    got, want,
                    "owner={planet_owner} garrison={garrison} arrivals={arrivals:?}"
                );
                cases += 1;
            }
        }
    }
    assert!(cases > 50_000, "expected a broad grid, only ran {cases}");
}
