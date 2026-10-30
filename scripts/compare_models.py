#!/usr/bin/env python3
"""Sequential development comparison. Restores the 5M service even after failure.

Requires the SSH tunnel, the deployed worker and a real-data development TOML.
Downloads weights (up to ~1.5 GB for all four), then keeps one model resident.
"""
import argparse,json,re,subprocess,time,tomllib
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('config');p.add_argument('--sizes',nargs='+',choices=['5m','28m','123m','209m'],default=['5m','28m','123m','209m']);a=p.parse_args()
root=Path(__file__).resolve().parents[1];source=(root/a.config).read_text();config=tomllib.loads(source)
assert config.get('phase','development')=='development','model selection must use development data'
host='chablis@192.168.0.167';out=root/'output/model-comparison';out.mkdir(parents=True,exist_ok=True)
def ssh(command,**kw):return subprocess.run(['ssh','-o','BatchMode=yes',host,command],check=True,**kw)
def field(text,name,value):
    line=f'{name} = {json.dumps(value)}'
    return re.sub(rf'^{name}\s*=.*$',line,text,flags=re.M) if re.search(rf'^{name}\s*=',text,re.M) else text+'\n'+line+'\n'
summary={}
try:
    ssh('systemctl --user stop energy-timestral')
    for size in a.sizes:
        unit=f'energy-timestral-compare-{size}'
        try:
            subprocess.run([str(root/'energy'),'checkpoint','--size',size,'--output',f'models/{size}'],cwd=root,check=True)
            pin=json.loads((root/f'models/{size}/checkpoint.json').read_text())
            tar=subprocess.Popen(['tar','-cf','-',f'models/{size}'],cwd=root,stdout=subprocess.PIPE)
            ssh('tar -xf - -C ~/Projects/energy-timestral',stdin=tar.stdout);tar.stdout.close();assert tar.wait()==0
            ssh(f'systemd-run --user --collect --unit={unit} --working-directory=/home/chablis/Projects/energy-timestral /home/chablis/Projects/energy-timestral/target/release/energy-timestral serve --intel --model models/{size}')
            ready=False
            for _ in range(90):
                log=ssh(f'journalctl --user -u {unit} --no-pager -n 5',capture_output=True,text=True).stdout
                if 'ready 127.0.0.1:7419' in log:ready=True;break
                time.sleep(1)
            assert ready,'worker did not become ready; inspect its journal for memory/driver failure'
            study=field(field(source,'checkpoint',pin['fingerprint']),'output',f'output/model-comparison/{size}')
            path=out/f'{size}.toml';path.write_text(study)
            subprocess.run([str(root/'energy'),'backtest','--config',str(path)],cwd=root,check=True)
            summary[size]={'status':'complete','checkpoint':pin,'metrics':json.loads((out/size/'metrics.json').read_text())}
        except Exception as e:
            summary[size]={'status':'failed','error':str(e),'note':'not an accuracy result; inspect worker journal for resource or input failure'}
        finally:
            subprocess.run(['ssh',host,f'systemctl --user stop {unit}'],check=False)
            (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
finally:
    ssh('systemctl --user start energy-timestral || systemd-run --user --unit=energy-timestral --working-directory=/home/chablis/Projects/energy-timestral /home/chablis/Projects/energy-timestral/target/release/energy-timestral serve --intel --model models/5m')
