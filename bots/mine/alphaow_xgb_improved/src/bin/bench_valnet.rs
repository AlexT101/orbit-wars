//! Micro-benchmark for value_net inference. Measures feature extraction
//! and forward-pass cost separately.
//!
//! Usage:
//!     ALPHAOW_VALUE_NET_PATH=<path> cargo run --release --bin bench_valnet -- [iters]

use alphaow_bot::{parse_state, value_net};
use serde_json::Value;
use std::time::Instant;

const SAMPLE_OBS: &str = r#"{"player":0,"step":50,"angular_velocity":0.04,"planets":[[0,0,30.0,30.0,2.0,15,3],[1,1,70.0,70.0,2.0,12,3],[2,-1,50.0,20.0,1.5,8,2],[3,-1,50.0,80.0,1.5,8,2],[4,-1,20.0,50.0,1.7,5,2],[5,-1,80.0,50.0,1.7,5,2],[6,-1,35.0,45.0,1.0,3,1],[7,-1,65.0,55.0,1.0,3,1]],"initial_planets":[[0,0,30.0,30.0,2.0,15,3],[1,1,70.0,70.0,2.0,12,3],[2,-1,50.0,20.0,1.5,8,2],[3,-1,50.0,80.0,1.5,8,2],[4,-1,20.0,50.0,1.7,5,2],[5,-1,80.0,50.0,1.7,5,2],[6,-1,35.0,45.0,1.0,3,1],[7,-1,65.0,55.0,1.0,3,1]],"fleets":[[0,0,40.0,40.0,0.5,0,8],[1,1,60.0,60.0,3.0,1,8]],"comets":[],"comet_planet_ids":[]}"#;

fn main() {
    let iters: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(10_000);
    let v: Value = serde_json::from_str(SAMPLE_OBS).unwrap();
    let state = parse_state(&v);

    // Warmup: ensure weights are loaded.
    let warm = value_net::predict(&state, 0);
    println!("warmup predict = {:?} (None means no weights loaded)", warm);
    let has_net = value_net::is_ready();

    // 1. Feature extraction alone.
    let t = Instant::now();
    let mut sink = 0.0f32;
    for _ in 0..iters {
        let f = value_net::extract_features(&state, 0);
        sink += f.current[0] + f.extrap[0] + f.dist[0];
    }
    let feat_ms = t.elapsed().as_secs_f64() * 1000.0;
    println!(
        "extract_features × {} = {:.2} ms ({:.1} µs/call), sink={}",
        iters,
        feat_ms,
        feat_ms * 1000.0 / iters as f64,
        sink
    );

    if !has_net {
        println!("(no weights loaded; skipping forward bench)");
        return;
    }

    // 2. End-to-end predict (features + forward).
    let t = Instant::now();
    let mut acc = 0.0f64;
    for _ in 0..iters {
        if let Some(v) = value_net::predict(&state, 0) {
            acc += v;
        }
    }
    let pred_ms = t.elapsed().as_secs_f64() * 1000.0;
    println!(
        "predict × {} = {:.2} ms ({:.1} µs/call), avg_value={:.4}",
        iters,
        pred_ms,
        pred_ms * 1000.0 / iters as f64,
        acc / iters as f64,
    );
    println!(
        "forward-pass alone ≈ {:.1} µs/call",
        (pred_ms - feat_ms) * 1000.0 / iters as f64
    );
}
