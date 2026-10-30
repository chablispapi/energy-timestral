#!/bin/sh
set -eu
exec ssh -o ServerAliveInterval=30 chablis@192.168.0.167 \
  'cd ~/Projects/energy-timestral; exec target/release/energy-timestral serve --intel --model models/5m'
