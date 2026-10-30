#!/usr/bin/env python3
"""Real-data proof of cutoff isolation and forecasting without future actuals."""
import json
import subprocess
import uuid
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def main():
    dest=ROOT/'output/validation'/('svk-future-'+uuid.uuid4().hex[:8]);dest.mkdir()
    original=ROOT/'data/svk_2026/load.tsv'
    changed=dest/'changed.tsv'
    cutoff=1772928000 # March 8 UTC
    rows=[]
    for line in original.read_text().splitlines():
        c=line.split('\t')
        if int(c[0])>=cutoff:c[1]='NA'
        rows.append('\t'.join(c))
    changed.write_text('\n'.join(rows)+'\n')
    def run(name,data,day,to):
        out=dest/name;cfg=dest/f'{name}.toml'
        cfg.write_text(f'data = {json.dumps(str(data))}\noutput = {json.dumps(str(out))}\nfrom = "{day}"\nto = "{to}"\nfeatures = "calendar"\ncontext = 672\n')
        r=subprocess.run([str(ROOT/'target/debug/energy-timestral'),'forecast','--config',str(cfg)],cwd=ROOT,capture_output=True,text=True)
        (dest/f'{name}.log').write_text(r.stdout+r.stderr)
        assert r.returncode==0,r.stderr
        assert 'live=0' in r.stderr
        return out
    a=run('original',original,'2026-03-08','2026-03-09')
    b=run('unavailable',changed,'2026-03-08','2026-03-09')
    assert (a/f'covariates-{cutoff}.tsv').read_bytes()==(b/f'covariates-{cutoff}.tsv').read_bytes()
    def points(d):
        return [line.split('\t') for line in (d/'results.tsv').read_text().splitlines() if line.startswith('P\t') and line.split('\t')[3]=='timestral']
    assert [p[5:] for p in points(a)]==[p[5:] for p in points(b)]
    c=run('future-no-actuals',original,'2026-09-01','2026-09-02')
    assert len(points(c))==48 and all(p[4]=='NA' for p in points(c))
    summary={'status':'PASS','checks':['real-data unavailable future observations do not change calendar inputs or forecasts','forecasting beyond source coverage works without actuals'],'artifacts':str(dest.relative_to(ROOT))}
    (ROOT/'output/validation/svk-future.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
