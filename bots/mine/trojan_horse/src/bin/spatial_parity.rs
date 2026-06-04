//! Parity probe for the 13-d spatial features.
//!
//! Reads one JSON observation per line on stdin, writes one text line per obs:
//!     step player f0 f1 ... f12
//! so `train/check_spatial_parity.py` can diff Rust vs the Python
//! `spatial_features.compute` implementation.

use alphaow_bot::{parse_state, value_net};
use serde_json::Value;
use std::io::{self, BufRead, Write};

fn main() -> io::Result<()> {
    let stdin = io::stdin();
    let stdout = io::stdout();
    let mut out = stdout.lock();
    let mut buf = String::new();
    let mut handle = stdin.lock();
    loop {
        buf.clear();
        let n = handle.read_line(&mut buf)?;
        if n == 0 {
            break;
        }
        let line = buf.trim_end();
        if line.is_empty() {
            continue;
        }
        let v: Value = match serde_json::from_str(line) {
            Ok(v) => v,
            Err(_) => continue,
        };
        let state = parse_state(&v);
        let feats = value_net::summary_features_spatial::extract(&state, state.player);
        let mut s = format!("{} {}", state.step, state.player);
        for f in feats.iter() {
            s.push_str(&format!(" {:.6}", f));
        }
        s.push('\n');
        out.write_all(s.as_bytes())?;
        out.flush()?;
    }
    Ok(())
}
