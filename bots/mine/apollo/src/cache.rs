//! Pre-computed per-entity tables for the bot
//! Refreshed at start of game and when new comets spawn
//! Lookups are relative to current turn, i.e. `EntityCache::position(id, turns_ahead)`

use std::f64::consts::FRAC_PI_2;

use parking_lot::Mutex;

use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet};

use crate::aim;
use crate::constants::{
    CENTER, COMET_RADIUS, COMET_SPAWN_STEPS, EPISODE_STEPS, LAUNCH_CLEARANCE, MAX_SHIP_SPEED,
    ROTATION_LIMIT,
};
use crate::engine::{fleet_speed, CometGroup, Planet};

/// Aim solver result tuple: `(angle, turns, target_x, target_y,
/// fractional_flight_time)`. See [`crate::aim::AimResult`].
pub use crate::aim::AimResult;

#[derive(Clone, Copy)]
struct CachedAim {
    result: Option<AimResult>,
    /// Absolute game turn at which this entry was stored. Used to decide
    /// whether a [`COMET_SPAWN_STEPS`] event has occurred since (in which
    /// case `Some` results need re-verification — new comets can only block
    /// paths, never enable them, so `None` entries are valid forever).
    stored_at_turn: i64,
}

/// How a cached invariant aim is carried from its base turn to any other turn.
///
/// * `StaticFixed` — both endpoints are static planets and the chosen flight
///   path stays *outside* the rotation disc (closest approach to center ≥
///   [`ROTATION_LIMIT`]). No orbiting body can ever reach the path, and the sun
///   + static planets are fixed, so the whole (angle, turns, point, flight) is
///   literally unchanged every turn.
/// * `OrbitingRotating` — both endpoints are orbiting planets and the path
///   stays *inside* a shrunk disc no static body reaches. Shooter, target, and
///   all orbiting planets rotate rigidly by `ω·Δturn`, so the aim at turn `T`
///   is the base aim rotated by `ω·(T − base_turn)`: bearing gains the angle,
///   the target point rotates about center, turns/flight_time are unchanged.
///
/// In both cases only comets can change the verdict turn-to-turn; the
/// per-turn comet gate ([`crate::aim::comet_blocks_path`]) handles them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum InvariantMode {
    StaticFixed,
    OrbitingRotating,
}

/// A cross-turn-reusable aim result for a disc-qualified static→static or
/// orbiting→orbiting shot.
#[derive(Clone, Copy)]
struct InvariantAim {
    /// Absolute launch turn the `base` result was solved at.
    base_turn: i64,
    base: AimResult,
    mode: InvariantMode,
}

/// Per-`(src, target, ships)` invariant-cache entry. `Disqualified` records a
/// same-kind pair whose path fails the disc condition (or has no comet-free
/// shot at all): a permanent verdict — qualification is turn-invariant — so we
/// never re-attempt the (dual-scan) populate for it and just fall back to the
/// exact per-turn solver.
#[derive(Clone, Copy)]
enum InvariantEntry {
    Aim(InvariantAim),
    Disqualified,
}

/// Verdict from [`EntityCache::invariant_aim_lookup`] telling the caller how to
/// obtain this turn's shot.
pub enum InvariantVerdict {
    /// Carried-and-comet-cleared result — use it directly, no solve.
    Use(AimResult),
    /// Not invariant-cacheable this turn (mixed kinds, disqualified, or a comet
    /// gates the carried base): solve normally with [`crate::aim::aim_with_prediction`].
    SingleSolve,
    /// Unknown (cache miss) and potentially cacheable: solve the comet-free base
    /// with [`crate::aim::aim_ignoring_comets`] and feed it to
    /// [`EntityCache::invariant_aim_store`] to populate the entry.
    DualSolve,
}

/// Verdict from [`EntityCache::aim_cache_lookup`]. `Stale` means an entry
/// existed but failed post-comet re-verification and was evicted; the caller
/// must recompute.
pub enum AimCacheVerdict {
    Miss,
    Hit(Option<AimResult>),
    Stale,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EntityKind {
    StaticPlanet,
    OrbitingPlanet,
    Comet,
}

#[derive(Debug, Clone)]
pub struct Entity {
    pub id: i64,
    pub kind: EntityKind,
    pub radius: f64,
    pub orbital_radius: f64, // 0.0 for comets since they don't orbit sun
    pub positions: Vec<Option<[f64; 2]>>, // Pre-computed positions, or None if not on board (comets)
    /// First absolute turn at which the entity is off the board, precomputed at
    /// build time (read by [`crate::helpers`] when zeroing comet value past its
    /// lifetime). Comet paths are a single contiguous on-board window, so this is
    /// `last_on_board + 1` (clamped to `EPISODE_STEPS`). Planets never leave:
    /// `EPISODE_STEPS`.
    pub(crate) off_board_turn: i64,
    /// Capsule approximation of the on-board arc as `[ax, ay, bx, by]`: the chord
    /// from the first to the last on-board position. Together with [`Self::bulge`]
    /// it lets [`crate::aim::comet_blocks_path`] reject, in O(1) with no
    /// `sqrt`, any comet whose arc can't reach the fleet's swept segment. Unlike
    /// an axis-aligned box this hugs a long *diagonal* arc tightly. Planets store
    /// a degenerate chord (they never use this path).
    pub(crate) chord: [f64; 4],
    /// Max perpendicular deviation of the on-board arc from [`Self::chord`], so
    /// the capsule `chord ± bulge` contains the whole arc. `INFINITY` for planets
    /// and never-on-board comets (the capsule then never rejects — a false reject
    /// would silently miss a real block, so the degenerate case stays permissive).
    pub(crate) bulge: f64,
    /// Upper bound on the entity's per-turn displacement (so its distance from any
    /// fixed point can change by at most this much per turn). `0` for static
    /// planets, `angular_velocity · orbital_radius` for orbiters (chord ≤ arc),
    /// the max on-board step for comets. When the fleet speed exceeds this, the
    /// fleet's monotonic outward radius makes the radially-reachable turns a
    /// contiguous band, which [`crate::aim::blocked_on_path`] binary-searches
    /// instead of scanning every turn.
    pub(crate) max_step: f64,
}

impl Entity {
    #[inline]
    pub fn is_static(&self) -> bool {
        matches!(self.kind, EntityKind::StaticPlanet)
    }

    #[inline]
    pub fn is_comet(&self) -> bool {
        matches!(self.kind, EntityKind::Comet)
    }
}

/// Game-wide pre-computed position tables, keyed by planet/comet id.
pub struct EntityCache {
    pub current_turn: i64,
    pub angular_velocity: f64,
    /// Obstacles in a contiguous `Vec` so the hot per-obstacle scans
    /// ([`crate::aim::blocked_on_path`] / `cone_clear_impossible`) iterate
    /// cache-locally over the inline scalar fields instead of chasing `HashMap`
    /// buckets. Id lookups go through [`Self::id_to_idx`].
    pub(crate) entities: Vec<Entity>,
    /// `entity id → index into [`Self::entities`]`. Rebuilt on build /
    /// [`Self::refresh_comets`] (≤44 entries, so rebuilding is trivial).
    id_to_idx: HashMap<i64, usize>,
    /// Per-turn aim cache. `aim_cache[abs_launch_turn]` maps
    /// `(src, target, ships)` to a cached aim result. Indexed by the
    /// **absolute launch turn** (= `current_turn + launch_turn_offset` at
    /// storage time) rather than the storage turn, so that:
    ///   * Delayed-launch entries from the hellburner early-game DFS and
    ///     launch-now entries from later real turns share the slot whose
    ///     `abs_launch` matches, giving cross-turn reuse for free.
    ///   * Rollout forward-sim entries (stored at the rollout's notion of
    ///     `current_turn`) likewise share slots with the bot's real turns.
    aim_cache: Mutex<Vec<HashMap<(i64, i64, i64), CachedAim>>>,
    /// Cross-turn invariant aim cache, keyed by `(src, target, ships)` (no turn
    /// dimension — the entry is reusable at *every* turn via [`InvariantMode`]).
    /// See [`EntityCache::invariant_aim_lookup`] / `invariant_aim_store`.
    invariant_aim: Mutex<HashMap<(i64, i64, i64), InvariantEntry>>,
    /// Closest a static planet *body* gets to center: `min(orbital_r − radius)`
    /// over static planets, or `+∞` if there are none. An orbiting→orbiting
    /// flight segment lying entirely within this radius cannot be struck by any
    /// static planet, which is the qualifying condition for `OrbitingRotating`.
    static_inner_limit: f64,
    /// Ids of every comet entity currently in the cache (≤4 per spawned group).
    /// Maintained on build / [`Self::refresh_comets`] so the per-turn comet gate
    /// ([`crate::aim::comet_blocks_path`]) iterates only comets instead of
    /// scanning the whole entity map. Empty most of the early game.
    pub(crate) comet_ids: Vec<i64>,
}

impl EntityCache {
    pub fn build(
        initial_planets: &[Planet],
        comets: &[CometGroup],
        comet_planet_ids: &[i64],
        angular_velocity: f64,
        current_step: i64,
    ) -> Self {
        let comet_set: HashSet<i64> = comet_planet_ids.iter().copied().collect();
        let mut entities: Vec<Entity> = Vec::with_capacity(initial_planets.len());

        for ip in initial_planets {
            if comet_set.contains(&ip.id) {
                continue;
            }
            entities.push(build_planet_entity(ip, angular_velocity));
        }

        for group in comets {
            for (idx, &pid) in group.planet_ids.iter().enumerate() {
                entities.push(build_comet_entity(pid, group, idx, current_step));
            }
        }

        let aim_cache = (0..EPISODE_STEPS).map(|_| HashMap::default()).collect();

        // Tightest inner reach of any static planet body. Static planets never
        // move, so this is fixed for the whole game (comets are never static).
        let static_inner_limit = entities
            .iter()
            .filter(|e| e.is_static())
            .map(|e| e.orbital_radius - e.radius)
            .fold(f64::INFINITY, f64::min);

        let comet_ids = entities
            .iter()
            .filter(|e| e.is_comet())
            .map(|e| e.id)
            .collect();
        let id_to_idx = entities
            .iter()
            .enumerate()
            .map(|(i, e)| (e.id, i))
            .collect();

        Self {
            current_turn: current_step,
            angular_velocity,
            entities,
            id_to_idx,
            aim_cache: Mutex::new(aim_cache),
            invariant_aim: Mutex::new(HashMap::default()),
            static_inner_limit,
            comet_ids,
        }
    }

    /// Drop expired comet entries and add newly-spawned ones.
    pub fn refresh_comets(
        &mut self,
        comets: &[CometGroup],
        comet_planet_ids: &[i64],
        current_step: i64,
    ) {
        let comet_set: HashSet<i64> = comet_planet_ids.iter().copied().collect();

        // Drop expired comets (preserves order, keeps planets and live comets).
        self.entities
            .retain(|ent| !ent.is_comet() || comet_set.contains(&ent.id));

        // Append newly-spawned comets not already present. Existing comets keep
        // their precomputed positions (they were retained, not rebuilt).
        let mut present: HashSet<i64> = self.entities.iter().map(|e| e.id).collect();
        for group in comets {
            for (idx, &pid) in group.planet_ids.iter().enumerate() {
                if present.insert(pid) {
                    self.entities
                        .push(build_comet_entity(pid, group, idx, current_step));
                }
            }
        }

        self.id_to_idx = self
            .entities
            .iter()
            .enumerate()
            .map(|(i, e)| (e.id, i))
            .collect();
        self.comet_ids = self
            .entities
            .iter()
            .filter(|e| e.is_comet())
            .map(|e| e.id)
            .collect();
    }

    #[inline]
    pub fn set_current_turn(&mut self, turn: i64) {
        self.current_turn = turn;
    }

    #[inline]
    pub fn get(&self, id: i64) -> Option<&Entity> {
        self.entities.get(*self.id_to_idx.get(&id)?)
    }

    /// Look up a cached aim result for a shot launching at
    /// `current_turn + launch_turn_offset`.
    ///
    /// `Some` entries stored across a [`COMET_SPAWN_STEPS`] boundary are
    /// re-verified against the obstacle set at the entry's launch turn:
    /// passing entries are returned (with `stored_at_turn` refreshed so the
    /// next lookup is free); failing entries are evicted and reported as
    /// `Stale`. `None` entries never need re-verification — a fresh comet
    /// can only block paths, never enable them.
    pub fn aim_cache_lookup(
        &self,
        src: i64,
        target: i64,
        ships: i64,
        launch_turn_offset: i64,
    ) -> AimCacheVerdict {
        let abs_launch = self.current_turn + launch_turn_offset;
        if abs_launch < 0 || (abs_launch as usize) >= EPISODE_STEPS as usize {
            return AimCacheVerdict::Miss;
        }
        let slot = abs_launch as usize;
        let key = (src, target, ships);

        let entry = {
            let map = self.aim_cache.lock();
            match map[slot].get(&key) {
                None => return AimCacheVerdict::Miss,
                Some(e) => *e,
            }
        };

        match entry.result {
            None => AimCacheVerdict::Hit(None),
            Some(result) => {
                if !comet_spawn_crossed(entry.stored_at_turn, self.current_turn) {
                    return AimCacheVerdict::Hit(Some(result));
                }
                let (angle, _turns, _tx, _ty, flight_time) = result;
                if aim::shot_still_clear(
                    self,
                    src,
                    target,
                    ships,
                    angle,
                    flight_time,
                    launch_turn_offset,
                ) {
                    self.aim_cache.lock()[slot].insert(
                        key,
                        CachedAim {
                            result: Some(result),
                            stored_at_turn: self.current_turn,
                        },
                    );
                    AimCacheVerdict::Hit(Some(result))
                } else {
                    self.aim_cache.lock()[slot].remove(&key);
                    AimCacheVerdict::Stale
                }
            }
        }
    }

    /// Store an aim result and, for free, populate its three quartet siblings.
    ///
    /// The whole obstacle field (sun at CENTER + every planet + every comet) is
    /// invariant under rotation by 90° about CENTER **at every absolute turn**
    /// — orbital motion and comet motion commute with that rotation, and each
    /// quartet's four members are exactly the four reflections of the same base
    /// point (see `rust_engine/src/lib.rs`). Hence rotating *both* endpoints to
    /// their corresponding quartet members (member `i` → member `j` via the
    /// rotation `(TAG[j] − TAG[i])·90°`) yields a congruent aim problem whose
    /// solution is this one rotated by the same `k` quarter-turns: `angle` gains
    /// `k·π/2`, the target point rotates about CENTER, and `turns`/`flight_time`
    /// are unchanged. `None` (no feasible shot) propagates too. Both endpoints
    /// are rotated by the *same* `k`, so same-quartet `src`/`target` pairs need
    /// no special handling. Siblings share the `abs_launch` slot (the symmetry
    /// holds at that exact turn) and a fresh `stored_at_turn`.
    pub fn aim_cache_store(
        &self,
        src: i64,
        target: i64,
        ships: i64,
        launch_turn_offset: i64,
        result: Option<AimResult>,
    ) {
        let abs_launch = self.current_turn + launch_turn_offset;
        if abs_launch < 0 || (abs_launch as usize) >= EPISODE_STEPS as usize {
            return;
        }
        let slot = abs_launch as usize;
        let mut map = self.aim_cache.lock();

        // Original entry (always stored, preserving prior behavior).
        map[slot].insert(
            (src, target, ships),
            CachedAim {
                result,
                stored_at_turn: self.current_turn,
            },
        );

        // Three rotated siblings. Gated on existence so we never populate ids for
        // a comet member that has expired or not yet spawned.
        for k in 1..=3 {
            let sib_src = rot_sibling(src, k);
            let sib_target = rot_sibling(target, k);
            if !self.id_to_idx.contains_key(&sib_src) || !self.id_to_idx.contains_key(&sib_target) {
                continue;
            }
            map[slot].insert(
                (sib_src, sib_target, ships),
                CachedAim {
                    result: rotate_aim_result(result, k),
                    stored_at_turn: self.current_turn,
                },
            );
        }
    }

    /// Cross-turn fast-path lookup for a static→static or orbiting→orbiting
    /// shot, skipping `lead_target_from` and the whole per-entity planet sweep.
    /// Returns:
    ///   * [`InvariantVerdict::Use`] — the carried base, derived for this turn
    ///     (identity for static, rotated `ω·Δturn` for orbiting) and confirmed
    ///     comet-clear by the per-turn gate. No solve needed.
    ///   * [`InvariantVerdict::SingleSolve`] — not cacheable this turn (mixed
    ///     kinds, a recorded disqualification, or a comet gates the base): solve
    ///     once with `aim_with_prediction`.
    ///   * [`InvariantVerdict::DualSolve`] — cache miss for a potentially
    ///     cacheable pair: solve the comet-free base with `aim_ignoring_comets`
    ///     and feed it to [`Self::invariant_aim_store`].
    ///
    /// See [`InvariantMode`] for the carry rules. Composes with the per-turn
    /// `aim_cache`: callers consult that first and only reach here on a miss.
    pub fn invariant_aim_lookup(
        &self,
        src: i64,
        target: i64,
        ships: i64,
        launch_turn_offset: i64,
    ) -> InvariantVerdict {
        let abs_launch = self.current_turn + launch_turn_offset;
        // `abs_launch >= 1` dodges the orbital-angle seam at turns 0/1
        // (`positions[0]` and `positions[1]` share an angle — see
        // `build_planet_entity`), keeping the `ω·Δturn` carry exact. Out of
        // range ⇒ just solve normally without populating.
        if abs_launch < 1 || abs_launch >= EPISODE_STEPS {
            return InvariantVerdict::SingleSolve;
        }
        // Mixed-kind pairs can never be invariant-cached; resolve them directly
        // without ever attempting (or recording) a populate.
        if !self.same_invariant_kind(src, target) {
            return InvariantVerdict::SingleSolve;
        }

        let entry = {
            let map = self.invariant_aim.lock();
            match map.get(&(src, target, ships)) {
                None => return InvariantVerdict::DualSolve,
                Some(e) => *e,
            }
        };
        let aim = match entry {
            InvariantEntry::Disqualified => return InvariantVerdict::SingleSolve,
            InvariantEntry::Aim(a) => a,
        };

        let derived = match aim.mode {
            InvariantMode::StaticFixed => aim.base,
            InvariantMode::OrbitingRotating => {
                let dtheta = self.angular_velocity * (abs_launch - aim.base_turn) as f64;
                rotate_aim_continuous(aim.base, dtheta)
            }
        };

        // Sun + planets are fixed (static) / rotation-equivariant (orbiting) for
        // a disc-qualified path; only a comet can change the verdict this turn.
        let (angle, _turns, _tx, _ty, flight_time) = derived;
        let v = fleet_speed(ships.max(1), MAX_SHIP_SPEED);
        if aim::comet_blocks_path(self, src, target, angle, flight_time, v, launch_turn_offset) {
            return InvariantVerdict::SingleSolve;
        }
        InvariantVerdict::Use(derived)
    }

    /// Populate the invariant entry for `(src, target, ships)` from a
    /// [`crate::aim::aim_ignoring_comets`] `comet_free` base (the shot clear
    /// of sun + planets only). A comet-dodging nudge is turn-specific and would
    /// not reproduce, so the comet-free base — not the with-comet `actual` — is
    /// what carries: a nudge around the sun / a fixed (static) or rigidly
    /// rotating (orbiting) planet reproduces, and the per-turn gate in
    /// [`Self::invariant_aim_lookup`] handles comets.
    ///
    /// Records [`InvariantEntry::Disqualified`] when there is no comet-free shot
    /// or the path fails the disc condition (a turn-invariant verdict), so the
    /// caller never retries the dual-scan populate. Otherwise stores the base
    /// plus its three quartet siblings (disc conditions and entity kinds are
    /// 90°-rotation invariant, so a qualifying base implies qualifying siblings).
    ///
    /// Call only on a [`InvariantVerdict::DualSolve`] (same-kind, in range).
    pub fn invariant_aim_store(
        &self,
        src: i64,
        target: i64,
        ships: i64,
        launch_turn_offset: i64,
        comet_free: Option<AimResult>,
    ) {
        let abs_launch = self.current_turn + launch_turn_offset;
        if abs_launch < 1 || abs_launch >= EPISODE_STEPS {
            return;
        }
        let mode = match (
            self.get(src).map(|e| e.kind),
            self.get(target).map(|e| e.kind),
        ) {
            (Some(EntityKind::StaticPlanet), Some(EntityKind::StaticPlanet)) => {
                InvariantMode::StaticFixed
            }
            (Some(EntityKind::OrbitingPlanet), Some(EntityKind::OrbitingPlanet)) => {
                InvariantMode::OrbitingRotating
            }
            _ => return,
        };

        let mut map = self.invariant_aim.lock();
        // No comet-free shot, or out of disc ⇒ permanent disqualification.
        let res = match comet_free {
            Some(r) if self.path_disc_qualified(src, ships, launch_turn_offset, &r, mode) => r,
            _ => {
                map.insert((src, target, ships), InvariantEntry::Disqualified);
                return;
            }
        };

        for k in 0..=3 {
            let sib_src = rot_sibling(src, k);
            let sib_target = rot_sibling(target, k);
            if !self.id_to_idx.contains_key(&sib_src) || !self.id_to_idx.contains_key(&sib_target) {
                continue;
            }
            let Some(base) = rotate_aim_result(Some(res), k) else {
                continue;
            };
            map.insert(
                (sib_src, sib_target, ships),
                InvariantEntry::Aim(InvariantAim {
                    base_turn: abs_launch,
                    base,
                    mode,
                }),
            );
        }
    }

    /// True iff `src` and `target` are the same invariant-cacheable kind
    /// (static→static or orbiting→orbiting). Comets and mixed pairs are not.
    fn same_invariant_kind(&self, src: i64, target: i64) -> bool {
        matches!(
            (
                self.get(src).map(|e| e.kind),
                self.get(target).map(|e| e.kind)
            ),
            (
                Some(EntityKind::StaticPlanet),
                Some(EntityKind::StaticPlanet)
            ) | (
                Some(EntityKind::OrbitingPlanet),
                Some(EntityKind::OrbitingPlanet)
            )
        )
    }

    /// True iff the flight segment for `res` satisfies `mode`'s disc condition:
    /// `StaticFixed` requires the segment's closest approach to center to be
    /// `≥ ROTATION_LIMIT` (no orbiting body can reach it); `OrbitingRotating`
    /// requires the whole segment to lie within [`Self::static_inner_limit`]
    /// (no static body can reach it). The segment is the launch-ring → arrival
    /// ray, identical to the one [`crate::aim::shot_blocked_exact`] sweeps.
    fn path_disc_qualified(
        &self,
        src: i64,
        ships: i64,
        launch_turn_offset: i64,
        res: &AimResult,
        mode: InvariantMode,
    ) -> bool {
        let Some([lx, ly]) = self.position(src, launch_turn_offset) else {
            return false;
        };
        let (angle, _turns, _tx, _ty, flight_time) = *res;
        let shooter_radius = self.get(src).map(|e| e.radius).unwrap_or(0.0);
        let launch_offset = shooter_radius + LAUNCH_CLEARANCE;
        let v = fleet_speed(ships.max(1), MAX_SHIP_SPEED);
        let ring_d = launch_offset + flight_time * v;
        let (ux, uy) = (angle.cos(), angle.sin());
        let sx = lx + launch_offset * ux;
        let sy = ly + launch_offset * uy;
        let ex = lx + ring_d * ux;
        let ey = ly + ring_d * uy;
        match mode {
            InvariantMode::StaticFixed => {
                point_seg_dist(CENTER, CENTER, sx, sy, ex, ey) >= ROTATION_LIMIT
            }
            InvariantMode::OrbitingRotating => {
                let ds = ((sx - CENTER).powi(2) + (sy - CENTER).powi(2)).sqrt();
                let de = ((ex - CENTER).powi(2) + (ey - CENTER).powi(2)).sqrt();
                ds.max(de) <= self.static_inner_limit
            }
        }
    }

    /// Drop all cached aim results for launches at absolute `turn`. Called
    /// once per bot turn (with `turn = current_turn - 1`) to release slots
    /// whose launch time has passed and that can no longer be queried.
    pub fn clear_aim_cache_slot(&mut self, turn: i64) {
        if turn < 0 || (turn as usize) >= EPISODE_STEPS as usize {
            return;
        }
        if let Some(slot) = self.aim_cache.get_mut().get_mut(turn as usize) {
            slot.clear();
        }
    }

    #[inline]
    pub fn position(&self, id: i64, turns_ahead: i64) -> Option<[f64; 2]> {
        let entity = self.get(id)?;
        let abs = self.current_turn + turns_ahead;
        // Table holds indices `[0, EPISODE_STEPS]` (one past the step count) so a
        // final-tick intercept can read the target's last position.
        if abs < 0 || abs > EPISODE_STEPS {
            return None;
        }
        entity.positions[abs as usize]
    }

    /// Position of `id` at an **absolute** game turn, indexed directly into the
    /// precomputed table (unlike [`position`], which is relative to
    /// `current_turn`). Used by [`crate::engine::Simulator`] during rollout,
    /// where the simulator tracks its own absolute step and `current_turn` reflects
    /// a different (per-bot-turn) notion. Returns `None` when the entity is
    /// unknown, off-board (comets), or the turn is outside `[0, EPISODE_STEPS]`.
    #[inline]
    pub fn position_abs(&self, id: i64, abs_step: i64) -> Option<[f64; 2]> {
        // `[0, EPISODE_STEPS]` inclusive — see `position`. Index `EPISODE_STEPS`
        // is the post-move position for the final step, which the engine's
        // orbital loop reads as `position_abs(id, turn_step + 1)` on step
        // `EPISODE_STEPS - 1` (its trig fallback now only fires without a cache).
        if abs_step < 0 || abs_step > EPISODE_STEPS {
            return None;
        }
        self.get(id)?.positions[abs_step as usize]
    }
}

/// `true` iff any [`COMET_SPAWN_STEPS`] tick falls in `(stored, current]`.
/// Comets always spawn on these fixed game steps, so this is the exact
/// criterion for "the obstacle set may have grown since the cache write."
#[inline]
fn comet_spawn_crossed(stored: i64, current: i64) -> bool {
    if current <= stored {
        return false;
    }
    COMET_SPAWN_STEPS
        .iter()
        .any(|&s| s > stored && s <= current)
}

/// Member-index ⇄ rotation-tag map for a quartet. The four members of a base
/// point are its four D4 reflections, generated in this index order (engine
/// `generate_planets` / `generate_comet_paths`); ordering them by the
/// rotation taking member 0 → member i gives tags `[0, 1, 3, 2]` (members 2 and
/// 3 are swapped vs. naive index order). `TAG` is its own inverse, so the same
/// table maps both directions. The rotation from member `i` to member `j` is
/// `(TAG[j] − TAG[i]) mod 4` quarter-turns CCW about CENTER.
const TAG: [i64; 4] = [0, 1, 3, 2];

/// Id of the quartet sibling reached by rotating `id`'s position `k`
/// quarter-turns CCW about CENTER. Ids are assigned 4-per-group on a `/4`
/// boundary (planets `0..4G`; each comet group's four ids start at `max_id+1`,
/// itself a multiple of 4), so `id & 3` is the member index and `id - (id & 3)`
/// the quartet base.
#[inline]
pub(crate) fn rot_sibling(id: i64, k: i64) -> i64 {
    let m = (id & 3) as usize;
    let rotated = TAG[((TAG[m] + k) & 3) as usize];
    (id - id % 4) + rotated
}

/// Rotate `(x, y)` `k` quarter-turns CCW about CENTER. Exact (each step is the
/// integer-coefficient map `(x, y) → (-y, x)` about CENTER).
#[inline]
fn rot_point(x: f64, y: f64, k: i64) -> (f64, f64) {
    let mut dx = x - CENTER;
    let mut dy = y - CENTER;
    for _ in 0..(k & 3) {
        let (nx, ny) = (-dy, dx);
        dx = nx;
        dy = ny;
    }
    (CENTER + dx, CENTER + dy)
}

/// Rotate an aim result `k` quarter-turns CCW: bearing gains `k·π/2`, the target
/// point rotates about CENTER, `turns`/`flight_time` are invariant. `None`
/// (no feasible shot) maps to `None`.
#[inline]
fn rotate_aim_result(result: Option<AimResult>, k: i64) -> Option<AimResult> {
    result.map(|(angle, turns, tx, ty, flight_time)| {
        let (rx, ry) = rot_point(tx, ty, k);
        (
            angle + (k & 3) as f64 * FRAC_PI_2,
            turns,
            rx,
            ry,
            flight_time,
        )
    })
}

/// Rotate an aim result by a continuous angle `dtheta` (radians) CCW about
/// center: bearing gains `dtheta`, the target point rotates about center,
/// turns/flight_time are unchanged. The continuous analogue of
/// [`rotate_aim_result`] (which is restricted to quarter-turns); used to carry
/// an `OrbitingRotating` invariant aim from its base turn to the current turn.
#[inline]
fn rotate_aim_continuous(result: AimResult, dtheta: f64) -> AimResult {
    let (angle, turns, tx, ty, flight_time) = result;
    let (c, s) = (dtheta.cos(), dtheta.sin());
    let dx = tx - CENTER;
    let dy = ty - CENTER;
    (
        angle + dtheta,
        turns,
        CENTER + dx * c - dy * s,
        CENTER + dx * s + dy * c,
        flight_time,
    )
}

/// Euclidean distance from point `(px, py)` to segment `(ax, ay)–(bx, by)`.
#[inline]
fn point_seg_dist(px: f64, py: f64, ax: f64, ay: f64, bx: f64, by: f64) -> f64 {
    let abx = bx - ax;
    let aby = by - ay;
    let l2 = abx * abx + aby * aby;
    let t = if l2 <= 0.0 {
        0.0
    } else {
        (((px - ax) * abx + (py - ay) * aby) / l2).clamp(0.0, 1.0)
    };
    let qx = ax + t * abx;
    let qy = ay + t * aby;
    ((px - qx).powi(2) + (py - qy).powi(2)).sqrt()
}

fn build_planet_entity(planet: &Planet, angular_velocity: f64) -> Entity {
    let dx = planet.x - CENTER;
    let dy = planet.y - CENTER;
    let orbital_radius = (dx * dx + dy * dy).sqrt();
    let is_static = orbital_radius + planet.radius >= ROTATION_LIMIT;
    let kind = if is_static {
        EntityKind::StaticPlanet
    } else {
        EntityKind::OrbitingPlanet
    };

    // One extra slot past `EPISODE_STEPS`: a fleet launched on the last actable
    // step can still collide that step, and the aimer reads the target position
    // at index `EPISODE_STEPS` for that final-tick intercept (see `lead_target_from`
    // and the engine's matching trig fallback in `Simulator::step_with_actions`).
    let cap = EPISODE_STEPS as usize + 1;
    let mut positions = Vec::with_capacity(cap);
    if is_static {
        for _ in 0..cap {
            positions.push(Some([planet.x, planet.y]));
        }
    } else {
        let init_angle = dy.atan2(dx);
        for t in 0..(EPISODE_STEPS + 1) {
            let effective = (t - 1).max(0);
            let angle = init_angle + angular_velocity * effective as f64;
            positions.push(Some([
                CENTER + orbital_radius * angle.cos(),
                CENTER + orbital_radius * angle.sin(),
            ]));
        }
    }

    Entity {
        id: planet.id,
        kind,
        radius: planet.radius,
        orbital_radius,
        positions,
        off_board_turn: EPISODE_STEPS, // planets stay on the board all game
        // Planets never feed the comet gate; `INFINITY` bulge never rejects.
        chord: [0.0; 4],
        bulge: f64::INFINITY,
        // Static: fixed. Orbiter: per-turn chord 2·R·sin(ω/2) ≤ R·ω.
        max_step: if is_static {
            0.0
        } else {
            angular_velocity * orbital_radius
        },
    }
}

fn build_comet_entity(pid: i64, group: &CometGroup, idx: usize, current_step: i64) -> Entity {
    // `+ 1` mirrors `build_planet_entity`: the trailing index `EPISODE_STEPS`
    // stays `None` for comets (no final-tick comet intercepts are planned), but
    // its presence keeps `position`/`position_abs` in-bounds at that index.
    let cap = EPISODE_STEPS as usize + 1;
    let mut positions = vec![None; cap];

    let mut first_on_board: i64 = -1;
    let mut last_on_board: i64 = -1;
    if let Some(path) = group.paths.get(idx) {
        let base = group.path_index - current_step;
        for t in 0..(EPISODE_STEPS as i64) {
            let pi = base + t;
            if pi >= 0 && (pi as usize) < path.len() {
                let p = path[pi as usize];
                positions[t as usize] = Some([p[0], p[1]]);
                if first_on_board < 0 {
                    first_on_board = t;
                }
                last_on_board = t;
            }
        }
    }
    // First off-board turn = one past the last on-board turn (clamped). 0 if the
    // comet is never on board within range.
    let off_board_turn = if last_on_board < 0 {
        0
    } else {
        (last_on_board + 1).min(EPISODE_STEPS)
    };

    // Capsule bound: chord from the first to last on-board point, plus the max
    // perpendicular deviation of the arc from it. Tight even for a long diagonal
    // arc. A degenerate window (never on board) stays permissive (`INFINITY`) so
    // it can never be the cause of a (correctness-critical) false reject.
    let (chord, bulge, max_step) = if first_on_board >= 0 {
        let a = positions[first_on_board as usize].unwrap();
        let b = positions[last_on_board as usize].unwrap();
        let abx = b[0] - a[0];
        let aby = b[1] - a[1];
        let len2 = abx * abx + aby * aby;
        let mut max_dev2 = 0.0_f64;
        let mut max_step2 = 0.0_f64;
        let window = &positions[first_on_board as usize..=last_on_board as usize];
        for (i, slot) in window.iter().enumerate() {
            if let Some(p) = slot {
                let dev2 = if len2 > 1e-12 {
                    // (perpendicular distance)^2 = (cross / |AB|)^2.
                    let cross = (p[0] - a[0]) * aby - (p[1] - a[1]) * abx;
                    cross * cross / len2
                } else {
                    // Degenerate chord (single distinct point): deviate from A.
                    let dx = p[0] - a[0];
                    let dy = p[1] - a[1];
                    dx * dx + dy * dy
                };
                max_dev2 = max_dev2.max(dev2);
                // Per-turn step to the previous on-board point (contiguous window).
                if let Some(Some(prev)) = window.get(i.wrapping_sub(1)) {
                    let sx = p[0] - prev[0];
                    let sy = p[1] - prev[1];
                    max_step2 = max_step2.max(sx * sx + sy * sy);
                }
            }
        }
        ([a[0], a[1], b[0], b[1]], max_dev2.sqrt(), max_step2.sqrt())
    } else {
        ([0.0; 4], f64::INFINITY, 0.0)
    };

    Entity {
        id: pid,
        kind: EntityKind::Comet,
        radius: COMET_RADIUS,
        orbital_radius: 0.0,
        positions,
        off_board_turn,
        chord,
        bulge,
        max_step,
    }
}
