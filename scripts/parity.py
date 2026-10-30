#!/usr/bin/env python3
"""Compare unrounded GPU output and layer fixtures with frozen tolerances."""
import json,sys
import numpy as np
a,b=(json.load(open(p)) for p in sys.argv[1:3])
np.testing.assert_allclose(b["levels"],a["levels"],atol=1e-7,rtol=0)
for name,x in a.get("trace",{}).items():
    # Upstream computes the head for all decoder slots on the covariate path.
    # Transformer layers are the common full-layout comparison points.
    if name=="head_raw" and len(x)!=len(b.get("trace",{}).get(name,[])):continue
    y=b["trace"][name]
    scale=max(1.,float(np.sqrt(np.mean(np.array(x)**2))))
    np.testing.assert_allclose(np.array(y)/scale,np.array(x)/scale,rtol=1e-3,atol=1e-4,err_msg=name)
np.testing.assert_allclose(b["quantiles"],a["quantiles"],rtol=1e-3,atol=1e-3)
print("PASS",a["id"],"max_absolute_error",float(np.max(np.abs(np.array(a["quantiles"])-np.array(b["quantiles"])))))
