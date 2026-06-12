//! Emit a stub value-network weights file in the format expected by
//! `alphaow_bot::value_net`. Useful for sanity-checking the inference
//! path before any real training has happened.
//!
//! Usage:
//!     cargo run --release --bin init_weights -- <out_path> [hidden] [scale]
//!
//! Defaults: hidden=64, scale=0.0 (all-zero weights → constant 0 output).
//! Set scale>0 for a deterministic pseudo-random init (xorshift-seeded
//! from the hidden size).

use alphaow_bot::value_net::INPUT_DIM;
use std::env;
use std::fs::File;
use std::io::Write;

fn xorshift(state: &mut u64) -> u64 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x
}

fn pseudo_normal(state: &mut u64) -> f32 {
    // Cheap "almost gaussian" — sum 6 uniforms minus 3, scaled.
    let mut acc = 0.0f64;
    for _ in 0..6 {
        let u = (xorshift(state) >> 11) as f64 / (1u64 << 53) as f64;
        acc += u;
    }
    (acc - 3.0) as f32 / 6.0f32.sqrt()
}

fn main() -> std::io::Result<()> {
    let args: Vec<String> = env::args().collect();
    let out_path = args.get(1).cloned().unwrap_or_else(|| {
        eprintln!("usage: init_weights <out_path> [hidden=64] [scale=0.0]");
        std::process::exit(2);
    });
    let hidden: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(64);
    let scale: f32 = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(0.0);

    let mut rng: u64 = (hidden as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15) ^ 0xdead_beef;

    let mut buf: Vec<u8> = Vec::new();
    buf.extend_from_slice(&0x564f4157u32.to_le_bytes()); // magic "AOWV"
    buf.extend_from_slice(&1u32.to_le_bytes()); // version
    buf.extend_from_slice(&(INPUT_DIM as u32).to_le_bytes());
    buf.extend_from_slice(&(hidden as u32).to_le_bytes());

    let push_f32 = |buf: &mut Vec<u8>, x: f32| buf.extend_from_slice(&x.to_le_bytes());
    // W1
    let std_w1 = scale * (2.0 / INPUT_DIM as f32).sqrt(); // He init
    for _ in 0..(hidden * INPUT_DIM) {
        let v = if scale > 0.0 { pseudo_normal(&mut rng) * std_w1 } else { 0.0 };
        push_f32(&mut buf, v);
    }
    // B1
    for _ in 0..hidden { push_f32(&mut buf, 0.0); }
    // W2
    let std_w2 = scale * (1.0 / hidden as f32).sqrt();
    for _ in 0..hidden {
        let v = if scale > 0.0 { pseudo_normal(&mut rng) * std_w2 } else { 0.0 };
        push_f32(&mut buf, v);
    }
    // B2
    push_f32(&mut buf, 0.0);

    let mut f = File::create(&out_path)?;
    f.write_all(&buf)?;
    println!(
        "wrote {} bytes (input_dim={}, hidden={}, scale={}) to {}",
        buf.len(),
        INPUT_DIM,
        hidden,
        scale,
        out_path
    );
    Ok(())
}
