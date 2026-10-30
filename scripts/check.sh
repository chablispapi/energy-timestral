#!/bin/sh
set -eu
cd "$(dirname "$0")/.."
cargo test --locked
keelc check
keelc check keel/tests.kl
mkdir -p build output/validation
keelc run keel/tests.kl > output/validation/keel-reference.txt
keelc build keel/tests.kl -o build/keel-tests
KEEL_STATS=1 build/keel-tests > output/validation/keel-native.txt 2> output/validation/keel-stats.txt
cmp output/validation/keel-reference.txt output/validation/keel-native.txt
grep 'live=0' output/validation/keel-stats.txt
grep 'Keel metrics, missing values and framing PASS' output/validation/keel-native.txt
