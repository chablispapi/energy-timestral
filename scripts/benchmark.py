#!/usr/bin/env python3
"""Measure the resident Intel worker, including SSH transport; no CPU speed claim."""
import json,socket,struct,time,statistics,argparse
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--request',default='tests/fixtures/sine.json');p.add_argument('--output',default='output/validation/benchmark.json');p.add_argument('--repetitions',type=int,default=5);a=p.parse_args()
r=json.load(open(a.request));body=f"ENERGY1\tbenchmark\t{r['horizon']}\t{r.get('max_context',672)}\t1\t1\t0\n"
for role,key in [('TARGET','context'),('PAST','past'),('KNOWN','known')]:
    for row in r.get(key,[]):body+=role+''.join('\t'+('NA' if v is None else str(round(v*1000))) for v in row)+'\n'
body+='END\n';body=body.encode();samples=[]
def recv(s,n):
    b=b''
    while len(b)<n:
        part=s.recv(n-len(b))
        if not part:raise RuntimeError('worker disconnected')
        b+=part
    return b
with socket.create_connection(('127.0.0.1',7419),timeout=120) as s:
    for i in range(a.repetitions+1):
        start=time.perf_counter();s.sendall(struct.pack('!I',len(body))+body);length=struct.unpack('!I',recv(s,4))[0];reply=recv(s,length).decode();wall=(time.perf_counter()-start)*1000;h=reply.splitlines()[0].split('\t');assert h[2]=='OK',reply
        samples.append(dict(worker_ms=int(h[3]),wall_ms=wall,device=h[4],checkpoint=h[5],warmup=i==0))
result=dict(samples=samples,median_worker_ms=statistics.median(x['worker_ms'] for x in samples[1:]),median_wall_ms=statistics.median(x['wall_ms'] for x in samples[1:]),note='GPU execution verified; CPU speedup has not been measured')
Path(a.output).parent.mkdir(parents=True,exist_ok=True);Path(a.output).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
