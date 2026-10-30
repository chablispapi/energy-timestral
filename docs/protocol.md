# Worker protocol v1

The worker binds to loopback and is reached over SSH forwarding. Each UTF-8 body
is prefixed with its byte length as an unsigned 32-bit big-endian integer. Length
must be 1–16,777,216 bytes. Reads/writes may be fragmented. A connection can carry
several sequential requests. The worker and Keel client use 120-second timeouts.

Request header (tabs between fields):

```
ENERGY1  id  horizon  max_context  symmetric  clip_nonnegative  trace  expected_checkpoint
TARGET   value  value ...
PAST     value  value ...
KNOWN    value  value ...
CATPAST  value  value ...
CATKNOWN value  value ...
END
```

The optional final header field pins SHA256(config.json bytes + model.safetensors
bytes). The application always supplies it. Booleans are 0/1. Values are signed
integers scaled by 1,000, or `NA`. All TARGET/PAST rows have context length T;
KNOWN rows have T + horizon values. One request is one multivariate series;
multiple TARGET rows are forecast jointly. Calendar/categorical codes use the
same fixed-point transport convention. Missing masks are retained separately.

Success:

```
ENERGY1 id OK elapsed_ms device checkpoint_fingerprint
LEVELS 100 500 1000 ... 9500 9900
Q target_index horizon_index q01 q05 ... q99
END
```

Each Q row has 21 fixed-point values; indexes start at zero. Errors use an
`ENERGY1 id ERROR message` header and END. An error never returns partial
forecasts or selects a CPU backend. JSON `infer` exposes raw FP32 quantiles and
optional layer traces for tests; JSON is not the Keel transport.

Canonical result records are P (origin, target timestamp, model, actual,
quantiles), M (per-origin metric sums), and W (the worker response header). Reports
preserve missing target/prediction counts and source/model fingerprints.
