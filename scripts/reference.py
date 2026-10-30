#!/usr/bin/env python3
"""Generate original Timestral fixtures on the Mac; never run on the N4020."""
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "reference/upstream/src"))
import numpy as np
import torch
from timestral import TimestralForecaster

p=argparse.ArgumentParser()
p.add_argument("request");p.add_argument("--model",default="models/5m");p.add_argument("--output",required=True)
a=p.parse_args();r=json.loads(Path(a.request).read_text())
torch.set_num_threads(2)
f=TimestralForecaster.from_pretrained(a.model,max_context=r.get("max_context",672))
traces={};hooks=[]
if r.get("trace"):
    def hook(name):
        def save(_module,_args,out):
            if name not in traces: traces[name]=out.detach().float().flatten().tolist()
        return save
    for i,l in enumerate(f.model.transformer.layers):hooks.append(l.register_forward_hook(hook(f"layer_{i}")))
    hooks.append(f.model.transformer.layers[0].register_forward_pre_hook(lambda _m,args:traces.setdefault("embedding",args[0].detach().float().flatten().tolist()) and None))
    hooks.append(f.model.head.projection.register_forward_hook(hook("head_raw")))
def array(key):
    rows=r.get(key,[])
    return np.array(rows,dtype=np.float64 if key.startswith("categorical") else np.float32)[None] if rows else None
out=f.forecast(array("context"),prediction_length=r["horizon"],past_covariates=array("past"),known_covariates=array("known"),categorical_past_covariates=array("categorical_past"),categorical_known_covariates=array("categorical_known"),symmetric_averaging=r.get("symmetric",True),clip_nonnegative=r.get("clip_nonnegative",True))
Path(a.output).write_text(json.dumps({"id":r["id"],"levels":out.quantile_levels,"quantiles":out.quantiles[0].tolist(),"trace":traces},allow_nan=False)+"\n")
