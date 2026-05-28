//! Orbit Wars bot — daemon mode.
//!
//! Reads one JSON observation per line on stdin, writes one JSON moves
//! array per line on stdout. The Python wrapper spawns the binary once and
//! pipes observations each turn.

use duck_bot::{duct, parse_state};
use serde_json::{json, Value};
use std::io::{self, BufRead, Write};

fn main() -> io::Result<()> {
    let stdin = io::stdin();
    let stdout = io::stdout();
    let mut out = stdout.lock();
    let debug = std::env::var("OW_DEBUG").is_ok();
    let mut err = io::stderr();
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
            writeln!(out, "[]")?;
            out.flush()?;
            continue;
        }
        let v: Value = match serde_json::from_str(line) {
            Ok(v) => v,
            Err(_) => {
                writeln!(out, "[]")?;
                out.flush()?;
                continue;
            }
        };
        let state = parse_state(&v);
        let actions = duct::best_move(&state, state.player, 500);
        let mv: Vec<(i64, f64, i64)> = actions
            .into_iter()
            .filter(|a| a.3 == state.player)
            .map(|a| (a.0, a.1, a.2))
            .collect();
        if debug {
            let me = state.player;
            let my_count = state.planets.iter().filter(|p| p.owner == me).count();
            let my_ships: i64 = state.planets.iter().filter(|p| p.owner == me).map(|p| p.ships).sum();
            let neutral = state.planets.iter().filter(|p| p.owner == -1).count();
            let enemy = state.planets.iter().filter(|p| p.owner != me && p.owner != -1).count();
            writeln!(
                err,
                "[duck p{}] step={} planets={}(m)/{}(n)/{}(e) ships={} fleets={} moves={}",
                me, state.step, my_count, neutral, enemy, my_ships, state.fleets.len(), mv.len()
            ).ok();
        }
        let arr: Vec<Value> = mv
            .into_iter()
            .map(|(fid, ang, ships)| json!([fid, ang, ships]))
            .collect();
        writeln!(out, "{}", Value::Array(arr))?;
        out.flush()?;
    }
    Ok(())
}
