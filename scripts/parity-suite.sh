#!/bin/sh
set -eu
cd "$(dirname "$0")/.."
mkdir -p output/validation
python3 scripts/make_fixtures.py
cargo build --locked
for name in sine missing constant negative multivariate covariates all-missing; do
  .venv/bin/python scripts/reference.py "tests/fixtures/$name.json" --output "output/validation/$name-reference.json"
  target/debug/energy-timestral infer --request "tests/fixtures/$name.json" --output "output/validation/$name-gpu.json"
  .venv/bin/python scripts/parity.py "output/validation/$name-reference.json" "output/validation/$name-gpu.json"
done
