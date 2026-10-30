#!/bin/sh
set -eu
cd "$(dirname "$0")/.."
test -f models/5m/model.safetensors
COPYFILE_DISABLE=1 tar --no-xattrs -cf - Cargo.toml Cargo.lock src reference/PIN.json models/5m |
  ssh chablis@192.168.0.167 'mkdir -p ~/Projects/energy-timestral; tar -xf - -C ~/Projects/energy-timestral; cd ~/Projects/energy-timestral; CARGO_BUILD_JOBS=1 ~/.cargo/bin/cargo build --release --locked'
