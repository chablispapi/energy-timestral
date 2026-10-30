#!/usr/bin/env python3
"""Real-worker checks for leakage, framing, metric units and checkpoint failures."""
import json,socket,struct,subprocess,uuid
from pathlib import Path
root=Path(__file__).resolve().parents[1]
out=root/'output/validation'/('integration-'+uuid.uuid4().hex[:8]);out.mkdir(parents=True)
pin=json.loads((root/'reference/PIN.json').read_text())['fingerprint']
def recv(s,n):
    out=b''
    while len(out)<n:
        part=s.recv(n-len(out))
        assert part,'disconnected during frame'
        out+=part
    return out
def send(body):
    b=body.encode();frame=struct.pack('!I',len(b))+b
    with socket.create_connection(('127.0.0.1',7419),timeout=120) as s:
        for i in range(0,len(frame),7):s.sendall(frame[i:i+7])
        return recv(s,struct.unpack('!I',recv(s,4))[0]).decode()
header=f'ENERGY1\tfragmented\t1\t32\t1\t0\t0\t{pin}\n'
row='TARGET'+('\t1000'*32)+'\nEND\n'
assert '\tOK\t' in send(header+row)
assert '\tERROR\t' in send(header.replace(pin,'0'*64)+row)
assert '\tERROR\t' in send(header+'TARGET\tNA\nKNOWN\t1000\nEND\n')

def run(name,data,checkpoint=pin,derived=False):
    cfg=out/f'{name}.toml';dest=out/name
    text=f'''data = {json.dumps(str(data))}
output = {json.dumps(str(dest))}
from = "2026-06-05"
to = "2026-06-06"
target = "price"
features = "residual"
context = 672
synthetic = true
checkpoint = "{checkpoint}"
load_data = {json.dumps(str(data))}
wind_data = {json.dumps(str(data))}
solar_data = {json.dumps(str(data))}
'''
    if not derived:text=text.replace('features = "residual"','features = "history"')
    cfg.write_text(text)
    r=subprocess.run([str(root/'target/debug/energy-timestral'),'forecast','--config',str(cfg)],cwd=root,capture_output=True,text=True)
    (out/f'{name}.log').write_text(r.stdout+r.stderr)
    return r,dest

original=root/'data/synthetic-2026.tsv'
changed=out/'changed.tsv'
cutoff=1780560000 # 2026-06-04 08:00 UTC = 10:00 Stockholm
# Keep published current-day prices in the primary target unchanged, but alter
# future load/generation inputs separately below.
lines=[]
for line in original.read_text().splitlines():
    v=line.split('\t')
    if int(v[0])>=cutoff:v[1]='999999999'
    lines.append('\t'.join(v))
changed.write_text('\n'.join(lines)+'\n')
r,a=run('original',original,derived=True);assert r.returncode==0,r.stderr
# Keep price target file identical; only change the three auxiliary inputs.
cfg=out/'changed-aux.toml';text=(out/'original.toml').read_text().replace(str(out/'original'),str(out/'changed-aux'))
for field in ('load_data','wind_data','solar_data'):
    text=text.replace(f'{field} = {json.dumps(str(original))}',f'{field} = {json.dumps(str(changed))}')
cfg.write_text(text)
r=subprocess.run([str(root/'target/debug/energy-timestral'),'forecast','--config',str(cfg)],cwd=root,capture_output=True,text=True)
(out/'changed-aux.log').write_text(r.stdout+r.stderr);assert r.returncode==0,r.stderr
b=out/'changed-aux'
assert (a/'covariates-1780560000.tsv').read_text()==(b/'covariates-1780560000.tsv').read_text(),'future electricity leaked into covariates'
def predictions(path):
    return [s.split('\t')[5:] for s in (path/'results.tsv').read_text().splitlines() if s.startswith('P\t') and s.split('\t')[3]=='timestral']
assert predictions(a)==predictions(b),'future electricity changed forecasts'

# Independently verify Keel fixed-point metrics against Python FP64 formulas.
points=[s.split('\t') for s in (a/'results.tsv').read_text().splitlines() if s.startswith('P\t') and s.split('\t')[3]=='timestral']
levels=[.01,.05]+[i/100 for i in range(10,95,5)]+[.95,.99]
ae=[];pb=[]
for p in points:
    if p[4]=='NA':continue
    y=int(p[4])/1000;qs=[int(v)/1000 for v in p[5:]];ae.append(abs(y-qs[10]))
    pb.append(sum(q*(y-x) if y>=x else (1-q)*(x-y) for q,x in zip(levels,qs))/21)
m=json.loads((a/'metrics.json').read_text())['metrics']['timestral']
assert abs(m['mae']-sum(ae)/len(ae))<1e-10
assert abs(m['pinball']-sum(pb)/len(pb))<1e-10
r,failed=run('wrong-checkpoint',original,checkpoint='0'*64)
assert r.returncode!=0
assert json.loads((failed/'run.json').read_text())['status']=='failed'
assert not any(s.startswith('P\t') for s in (failed/'results.tsv').read_text().splitlines())
summary={'status':'PASS','checks':['fragmented frame','checkpoint mismatch','invalid covariate lengths','future electricity leakage','Keel MAE/pinball units','failed runs retain diagnostics'],'artifacts':str(out)}
(root/'output/validation/integration.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
