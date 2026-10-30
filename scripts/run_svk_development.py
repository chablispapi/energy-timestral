#!/usr/bin/env python3
"""Run the four pre-July Svk development ablations; preserve all diagnostics."""
import argparse
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('features', nargs='*', metavar='FEATURE')
    args = parser.parse_args()
    if any(f not in ('history', 'calendar', 'weather', 'calendar-weather') for f in args.features):
        parser.error('features must be history, calendar, weather or calendar-weather')
    summary_path = ROOT / 'output/validation/svk-development-comparison.json'
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    for features in args.features or ('history', 'calendar', 'weather', 'calendar-weather'):
        log = ROOT / f'output/validation/svk-{features}-development.log'
        if log.exists():
            import time
            log.rename(log.with_name(log.stem+f'-attempt-{time.time_ns()}'+log.suffix))
        with log.open('w') as stream:
            result = subprocess.run(['target/debug/energy-timestral', 'backtest', '--config',
                f'config/svk-load-{features}.toml'], cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
        print(f'{features}: exit {result.returncode}; {log}', flush=True)
        if result.returncode:
            print(log.read_text()[-3000:], flush=True)
            summary[features] = {'status': 'failed', 'log': str(log.relative_to(ROOT))}
        else:
            run = ROOT / f'output/svk-load-{features}-development'
            assert 'live=0' in log.read_text()
            summary[features] = json.loads((run / 'metrics.json').read_text())['metrics']
        (ROOT / 'output/validation/svk-development-comparison.json').write_text(json.dumps(summary, indent=2)+'\n')
    if any(v.get('status') == 'failed' for v in summary.values()):
        raise SystemExit(1)

if __name__ == '__main__':
    main()
