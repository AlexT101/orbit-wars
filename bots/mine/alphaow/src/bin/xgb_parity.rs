//! Tiny binary: load an XGB JSON model + a file of input vectors,
//! print the Rust prediction for each. Used to check parity against
//! Python xgboost. Input vector file: one line per sample, comma-
//! separated f32 values.

use alphaow_bot::xgb;
use std::env;
use std::fs;
use std::io::{self, BufRead};

fn main() -> io::Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 3 {
        eprintln!("usage: xgb_parity <model.json> <samples.csv>");
        std::process::exit(1);
    }
    let bytes = fs::read(&args[1])?;
    let model = xgb::load(&bytes).expect("failed to load XGB model");
    eprintln!(
        "loaded model num_feature={} base_logit={:.6}",
        model.num_feature, model.base_score_logit
    );

    let f = fs::File::open(&args[2])?;
    let rdr = io::BufReader::new(f);
    for line in rdr.lines() {
        let line = line?;
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let vec: Vec<f32> = line
            .split(',')
            .filter_map(|s| s.trim().parse::<f32>().ok())
            .collect();
        if vec.len() != model.num_feature {
            eprintln!(
                "warn: sample dim {} != model num_feature {}, skipping",
                vec.len(),
                model.num_feature
            );
            continue;
        }
        let logit = model.predict_logit(&vec);
        // Python returns sigmoid(logit) = probability.
        let prob = 1.0f32 / (1.0 + (-logit).exp());
        let value = model.predict_value(&vec);
        println!("logit={:.6}  prob={:.6}  value={:.6}", logit, prob, value);
    }
    Ok(())
}
