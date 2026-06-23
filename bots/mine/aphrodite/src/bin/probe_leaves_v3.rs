//! Leaf-variance probe driver (4p / summary_v3).
//!
//! Reads one JSON observation per line on stdin (player+step+planets+fleets, the
//! same shape `extract_v3` consumes), and runs a full DUCT search on each via
//! `duct::best_move`. With the env vars below set, every value-net leaf the search
//! evaluates is dumped (145-d `summary_v3`, tagged by a monotonic search id), so
//! `probe_v3.py` can measure each feature's within-search sibling variance.
//!
//! Env:
//!   APHRODITE_VALUE_NET_PATH      = the v3 model the search evaluates under
//!   APHRODITE_DUMP_LEAVES_PATH    = output leaf dump file
//!   APHRODITE_DUMP_FEATURES=v3    = dump 145-d summary_v3 (not 65-d v2)
//!   APHRODITE_PROBE_BUDGET_MS     = per-search budget (default 150)

use aphrodite::{duct, parse_state};
use serde_json::Value;
use std::io::{self, BufRead};
use std::time::{Duration, Instant};

fn main() -> io::Result<()> {
    let budget: u64 = std::env::var("APHRODITE_PROBE_BUDGET_MS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(150);
    let stdin = io::stdin();
    let mut n = 0u64;
    for line in stdin.lock().lines() {
        let line = match line {
            Ok(l) => l,
            Err(_) => break,
        };
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let v: Value = match serde_json::from_str(line) {
            Ok(v) => v,
            Err(_) => continue,
        };
        let state = parse_state(&v);
        let now = Instant::now();
        let deadline = now + Duration::from_millis(budget);
        let _ = duct::best_move(
            &state,
            state.player,
            budget,
            deadline,
            deadline,
            &[],
            &[],
            &[],
            &[],
            &[],
            &[],
        );
        n += 1;
        if n % 25 == 0 {
            eprintln!("  ran {n} searches",);
        }
    }
    eprintln!("done: {n} searches");
    Ok(())
}
