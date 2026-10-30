#!/usr/bin/env python3
"""Deterministic test inputs, with no external electricity data."""
import csv,json,math
from pathlib import Path
root=Path(__file__).resolve().parents[1]
out=root/'tests/fixtures';out.mkdir(exist_ok=True)
def save(name,context,h=48,**kw):
    r=dict(id=name,context=context,horizon=h,max_context=672,symmetric=True,clip_nonnegative=True,trace=True)
    r.update(kw);(out/f'{name}.json').write_text(json.dumps(r,allow_nan=False)+'\n')
save('sine',[[100+8*math.sin(t*math.tau/24)+.01*t for t in range(672)]])
x=[None if t<50 or 120<t<140 else 20+math.sin(t/5) for t in range(257)]
save('missing',[x],33)
save('constant',[[5.]*64])
save('negative',[[-30+20*math.sin(t/7) for t in range(96)]],clip_nonnegative=False)
save('multivariate',[[30+math.sin(t/5) for t in range(96)],[3+math.cos(t/9) for t in range(96)]])
save('covariates',[[100+math.sin(t/5) for t in range(96)]],past=[[None if t%9==0 else math.cos(t/8) for t in range(96)]],known=[[t%24 for t in range(144)],[float(t%24<12) for t in range(144)]],categorical_known=[[None if t%7==0 else float(t%3) for t in range(144)]])
save('all-missing',[[None]*64])
data=root/'data';data.mkdir(exist_ok=True)
from datetime import datetime,timezone,timedelta
start=datetime(2023,11,1,tzinfo=timezone.utc)
with (data/'synthetic.csv').open('w') as f:
    w=csv.writer(f,lineterminator='\n');w.writerow(['timestamp','duration_seconds','value'])
    for i in range(24*100):
        ts=start+timedelta(hours=i)
        w.writerow([ts.isoformat(),3600,100+8*math.sin(i*math.tau/24)+.01*i])
print('generated deterministic fixtures and data/synthetic.csv')

start=datetime(2026,2,1,tzinfo=timezone.utc)
with (data/'synthetic-2026.csv').open('w') as f:
    w=csv.writer(f,lineterminator='\n');w.writerow(['timestamp','duration_seconds','value'])
    for i in range(24*300):
        ts=start+timedelta(hours=i)
        w.writerow([ts.isoformat(),3600,100+8*math.sin(i*math.tau/24)+.01*i])
