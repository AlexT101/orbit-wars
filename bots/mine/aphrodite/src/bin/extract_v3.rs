//! Fast 4p (FFA) feature extractor — `summary_v3` + decisiveness aux.
//!
//! Reads one JSON observation per line on stdin, writes one binary record per
//! line on stdout:
//!     step:i64, player:i32, summary_v3:[f32; 145], aux:[f32; 9]
//! → 8 + 4 + 580 + 36 = 628 bytes per record.
//!
//! See `train/FEATURE_SPEC_V3_4P.md`. Counterpart of `extract_v2` for the 4p
//! value-net redesign; 2p extraction stays on `extract_v2`.

use aphrodite::{parse_state, value_net};
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
        // Match the inference path: pressure features read `offset_lookahead`
        // via the apollo MODE, so set it from the live player count here too
        // (otherwise extraction would always use the 2p config).
        aphrodite::apollo::constants::set_mode_for_alive(aphrodite::sim::alive_players(&state));
        let (feats, aux) = value_net::summary_features_v3::extract_with_aux(&state, state.player);
        out.write_all(&state.step.to_le_bytes())?;
        out.write_all(&state.player.to_le_bytes())?;
        unsafe {
            let fb = std::slice::from_raw_parts(feats.as_ptr() as *const u8, feats.len() * 4);
            out.write_all(fb)?;
            let ab = std::slice::from_raw_parts(aux.as_ptr() as *const u8, aux.len() * 4);
            out.write_all(ab)?;
        }
        out.flush()?;
    }
    Ok(())
}
