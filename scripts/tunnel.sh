#!/bin/sh
set -eu
exec ssh -N -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 \
  -L 7419:127.0.0.1:7419 chablis@192.168.0.167
