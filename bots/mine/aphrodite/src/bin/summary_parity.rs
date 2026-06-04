//! Compare Rust-side `value_net::summary_features::extract` against a
//! sample of states from a captured dataset. The dataset must have been
//! collected with raw 2728-d features (so we can also derive the
//! reference summary via Python and compare via stdout pipe).
//!
//! Usage:
//!     summary_parity <obs_json>    # one JSON observation per line on stdin
//!
//! Prints `[step,player,feature_0,feature_1,...,feature_18]` per input
//! line so a Python driver can diff against `summary_features.py`.

use aphrodite::{parse_state, value_net};
use serde_json::Value;
use std::io::{self, BufRead, Write};

fn main() -> io::Result<()> {
    let stdin = io::stdin();
    let stdout = io::stdout();
    let mut out = stdout.lock();
    for line in stdin.lock().lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let v: Value = match serde_json::from_str(&line) {
            Ok(v) => v,
            Err(_) => continue,
        };
        let state = parse_state(&v);
        let feats = value_net::summary_features::extract(&state, state.player);
        let mut s = format!("{},{}", state.step, state.player);
        for x in feats.iter() {
            s.push(',');
            s.push_str(&format!("{:.6}", x));
        }
        writeln!(out, "{}", s)?;
        out.flush()?;
    }
    Ok(())
}
