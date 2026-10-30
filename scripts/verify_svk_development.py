#!/usr/bin/env python3
"""Acceptance audit of the imported source and completed development runs."""
import csv
import hashlib
import json
from datetime import datetime
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
def stamp(s):return int(datetime.fromisoformat(s.replace('Z','+00:00')).timestamp())
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def check_metric_units(directory, metrics):
    # Independent floating-point reconstruction from forecast rows and canonical
    # observations, rather than trusting Keel's aggregate metric records.
    observations={int(c[0]):int(c[1])/1000 for line in (ROOT/'data/svk_2026/load.tsv').read_text().splitlines() if (c:=line.split('\t'))[1]!='NA'}
    scales={}
    totals={}
    levels=[.01,.05]+[i/100 for i in range(10,95,5)]+[.95,.99]
    for line in (directory/'results.tsv').read_text().splitlines():
        c=line.split('\t')
        if c[0]!='P' or c[4]=='NA':continue
        origin=int(c[1]);y=int(c[4])/1000;qs=[int(v)/1000 for v in c[5:]]
        if origin not in scales:
            diffs=[abs(observations[t]-observations[t-604800]) for t in range(origin-672*3600,origin,3600) if t in observations and t-604800 in observations]
            scales[origin]=sum(diffs)/len(diffs)
        t=totals.setdefault(c[3],dict(n=0,mae=0,mase=0,pinball=0,coverage80=0,coverage90=0,width80=0,width90=0))
        t['n']+=1;t['mae']+=abs(y-qs[10]);t['mase']+=abs(y-qs[10])/scales[origin]
        t['pinball']+=sum(q*(y-p) if y>=p else (1-q)*(p-y) for q,p in zip(levels,qs))/21
        t['coverage80']+=qs[2]<=y<=qs[18];t['coverage90']+=qs[1]<=y<=qs[19]
        t['width80']+=qs[18]-qs[2];t['width90']+=qs[19]-qs[1]
    for name,t in totals.items():
        assert t['n']==metrics[name]['n']
        for key,value in t.items():
            if key!='n':assert abs(value/t['n']-metrics[name][key])<1e-8,(directory,name,key)

def main():
    assert json.loads((ROOT/'output/validation/svk-future.json').read_text())['status']=='PASS'
    assert json.loads((ROOT/'output/validation/integration.json').read_text())['status']=='PASS'
    assert 'test result: ok' in (ROOT/'output/validation/svk-core-checks.log').read_text()
    manifest=json.loads((ROOT/'data/prepared/svk_2026/manifest.json').read_text())
    assert digest(ROOT/'timvarden-2026-01-08.xls')==manifest['source_sha256']
    pin=json.loads((ROOT/'reference/PIN.json').read_text())['fingerprint']
    start=stamp('2025-12-31T23:00:00Z');stop=stamp('2026-08-31T23:00:00Z')
    grids=[]
    for kind in ('load','wind','solar'):
        p=ROOT/f'data/svk_2026/{kind}.tsv'
        rows=[line.split('\t') for line in p.read_text().splitlines()]
        assert len(rows)==5928
        assert all(r[1]!='NA' for r in rows[:5832])
        assert all(r[1]=='NA' for r in rows[5832:])
        grid=[int(r[0]) for r in rows];grids.append(grid)
        assert grid==list(range(start,stop+96*3600,3600))
        m=json.loads(p.with_suffix('.json').read_text())
        assert m['canonical_sha256']==digest(p)
        assert m['source_provenance']['manifest']==manifest
        assert m['source_provenance']['manifest_sha256']==digest(ROOT/'data/prepared/svk_2026/manifest.json')
        assert m['source_provenance']['csv_sha256']==digest(ROOT/f'data/prepared/svk_2026/{kind}.csv')==manifest['outputs'][f'{kind}.csv']
        with (ROOT/f'data/prepared/svk_2026/{kind}.csv').open() as f:
            source=list(csv.DictReader(f))
        assert all(abs(float(a['value'])-int(b[1])/1000)<=.000500001 for a,b in zip(source,rows))
    assert grids[0]==grids[1]==grids[2]
    summary={}
    weather_selections={}
    expected_origins=list(range(stamp('2026-03-08T00:00:00Z'),stamp('2026-07-01T00:00:00Z'),86400))
    for features in ('history','calendar','weather','calendar-weather'):
        d=ROOT/f'output/svk-load-{features}-development'
        run=json.loads((d/'run.json').read_text())
        assert run['status']=='complete' and run['jobs']==len(expected_origins)==115
        assert run['source_provenance']['target']['internal_missing_hours']==0
        assert 'live=0' in (ROOT/f'output/validation/svk-{features}-development.log').read_text()
        jobs=[line.split('\t') for line in (d/'jobs.tsv').read_text().splitlines()]
        assert [int(j[0]) for j in jobs]==expected_origins
        n=0;workers=0
        for line in (d/'results.tsv').read_text().splitlines():
            c=line.split('\t')
            if c[0]=='W':
                assert 'Intel(R)' in line and pin in line and '\tOK\t' in line;workers+=1
            if c[0]=='P':
                assert int(c[2])<stamp('2026-07-01T00:00:00Z')
                q=list(map(int,c[5:]));assert len(q)==21 and q==sorted(q)
                if c[3]=='timestral':n+=1
        assert workers==115 and n==5496
        metrics=json.loads((d/'metrics.json').read_text())['metrics']
        assert all(m['n']==5496 and m['missing_targets']==0 and m['missing']==0 and m['unavailable_predictions']==0 for m in metrics.values())
        check_metric_units(d,metrics)
        assert 'registered SE2 withdrawal' in (d/'report.html').read_text()
        assert run['config']['checkpoint']==pin and run['config']['context']==672
        if 'weather' in features:
            choices=run['source_provenance']['weather']
            assert len(choices)==115
            assert all(c['selected_initialization']<=c['origin']-8*3600 for c in choices)
            assert all(c['preferred_initialization']-c['selected_initialization'] in (0,86400) for c in choices)
            weather_selections[features]=[(c['origin'],c['selected_initialization']) for c in choices]
        summary[features]=metrics
    assert weather_selections['weather']==weather_selections['calendar-weather']
    for origin in expected_origins:
        a=(ROOT/f'output/svk-load-weather-development/covariates-{origin}.tsv').read_text().splitlines()
        b=(ROOT/f'output/svk-load-calendar-weather-development/covariates-{origin}.tsv').read_text().splitlines()
        assert len(a)==len(b)
        assert all(x.split('\t')[0]==y.split('\t')[0] and x.split('\t')[-6:]==y.split('\t')[-6:] for x,y in zip(a,b))
    (ROOT/'output/validation/svk-acceptance.json').write_text(json.dumps({'status':'PASS','observations_per_series':5832,'future_calendar_rows':96,'independent_source_checks':len(manifest['independent_checks']),'origins':115,'forecast_hours_per_variant':5496,'metrics':summary},indent=2)+'\n')
    baseline=summary['history']
    for metrics in summary.values():
        for name in ('yesterday','last_week'):assert metrics[name]==baseline[name]
    text='''# Svk SE2 development results

Daily UTC origins: March 8–June 30, 2026 (115 origins), 672 hours of context.
Each variant scores 5,496 forecast/actual pairs, including overlapping 48-hour
forecasts; the final origin is clipped at July 1. No July–September scoring was
performed. Load is registered SE2 withdrawal including network losses and
storage charging. These are retrospective settlement observations, with no
proof of publication-time availability. January and February provide context
and trailing baseline calibration. Weather variants use the same archived
values and initialization choices. When the preferred archive run is unavailable,
the preceding daily cycle is used only if it covers the complete horizon; all
choices are recorded under source_provenance.weather in run metadata.

MAE, pinball and widths are in MW; MASE is dimensionless. Baseline intervals use
past-only residuals. The same baselines apply to all four variants.

| Forecast | MAE | MASE | Pinball | 80% coverage | 90% coverage | 80% width | 90% width |
|---|---:|---:|---:|---:|---:|---:|---:|
'''
    rows=[(features,metrics['timestral']) for features,metrics in summary.items()]
    rows.extend((name,baseline[name]) for name in ('yesterday','last_week'))
    for name,m in rows:
        text+=f"| {name} | {m['mae']:.3f} | {m['mase']:.3f} | {m['pinball']:.3f} | {100*m['coverage80']:.2f}% | {100*m['coverage90']:.2f}% | {m['width80']:.3f} | {m['width90']:.3f} |\n"
    text+='\nAll variants have zero missing target observations. Detailed horizon errors,\nquantile calibration, source provenance and paired weekly-block bootstrap\ncomparisons are retained in each run’s metrics.json and report.html.\n\nAcceptance: 5,832 aligned source observations plus 96 missing calendar rows per\nseries, unchanged source workbook and CSVs, pinned Intel GPU checkpoint, sorted\n21-quantile forecasts, Keel live=0, all development targets before July 1.\nSee output/validation/svk-future.json and integration.json for real-data cutoff\nisolation and ancillary-load/generation leakage checks; test_svk_study.py\nverifies that the full holdout fails coverage preflight.\n'
    (ROOT/'docs/svk-development-results.md').write_text(text)
    print('Svk development acceptance PASS')

if __name__=='__main__':main()
