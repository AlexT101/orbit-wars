//! Fast transformer-token extractor.
//!
//! Reads one JSON observation per line on stdin, writes fixed binary records:
//!   step:i64, player:i32, tokens:[77,24]f32, mask:[77]u8, summary_v2/v3:f32[]
//!
//! Labels are attached by Python replay processors from the final reward.

use alphaow_bot::{parse_state, value_net};
use serde_json::Value;
use std::io::{self, BufRead, Write};

fn main() -> io::Result<()> {
    if std::env::args().any(|a| a == "--record-bytes") {
        let bytes = 8
            + 4
            + value_net::TRANSFORMER_MAX_TOKENS * value_net::TRANSFORMER_TOKEN_DIM * 4
            + value_net::TRANSFORMER_MAX_TOKENS
            + value_net::summary_features_v3::DIM * 4;
        println!("{}", bytes);
        return Ok(());
    }
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
        let (tokens, mask) = value_net::transformer_tokens(&state, state.player);
        let summary = value_net::summary_features_v3::extract(&state, state.player);
        out.write_all(&state.step.to_le_bytes())?;
        out.write_all(&state.player.to_le_bytes())?;
        unsafe {
            let token_bytes = std::slice::from_raw_parts(
                tokens.as_ptr() as *const u8,
                value_net::TRANSFORMER_MAX_TOKENS * value_net::TRANSFORMER_TOKEN_DIM * 4,
            );
            out.write_all(token_bytes)?;
        }
        for &m in &mask {
            out.write_all(&[if m { 1 } else { 0 }])?;
        }
        unsafe {
            let summary_bytes = std::slice::from_raw_parts(summary.as_ptr() as *const u8, summary.len() * 4);
            out.write_all(summary_bytes)?;
        }
        out.flush()?;
    }
    Ok(())
}
