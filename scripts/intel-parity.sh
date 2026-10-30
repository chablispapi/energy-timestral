#!/bin/sh
set -eu
cd "$(dirname "$0")/.."
mkdir -p output/validation
for name in sine missing constant negative multivariate covariates all-missing; do
  .venv/bin/python scripts/reference.py "tests/fixtures/$name.json" --output "output/validation/$name-reference.json"
  ssh -o BatchMode=yes -o ConnectTimeout=10 chablis@192.168.0.167 \
    'cd ~/Projects/energy-timestral; target/release/energy-timestral infer --intel --model models/5m --request /dev/stdin --output /tmp/energy-parity.json && cat /tmp/energy-parity.json' \
    < "tests/fixtures/$name.json" > "output/validation/$name-intel.json"
  .venv/bin/python scripts/parity.py "output/validation/$name-reference.json" "output/validation/$name-intel.json"
done
