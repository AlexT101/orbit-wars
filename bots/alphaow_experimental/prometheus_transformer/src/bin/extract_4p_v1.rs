//! Fast 4P feature extractor.
//!
//! Reads one JSON observation per line on stdin, writes:
//!     step:i64, player:i32, features_4p_v1:[f32; 236]

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
        let feats = value_net::summary_features_4p_v1::extract(&state, state.player);
        out.write_all(&state.step.to_le_bytes())?;
        out.write_all(&state.player.to_le_bytes())?;
        unsafe {
            let bytes = std::slice::from_raw_parts(feats.as_ptr() as *const u8, feats.len() * 4);
            out.write_all(bytes)?;
        }
        out.flush()?;
    }
    Ok(())
}
