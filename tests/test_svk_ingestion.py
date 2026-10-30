"""CLI regression checks: no network or GPU needed."""
import hashlib
import json
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BIN = ROOT / 'target/debug/energy-timestral'
PREP = ROOT / 'data/prepared/svk_2026'

class Ingestion(unittest.TestCase):
    def ingest(self, csv, manifest, output, kind='load', start='2025-12-31T23:00:00Z'):
        return subprocess.run([str(BIN), 'ingest', '--kind', kind, '--csv', str(csv),
            '--source-manifest', str(manifest), '--from', start,
            '--to', '2026-08-31T23:00:00Z', '--output', str(output)], capture_output=True, text=True)

    def test_prepared_imports_and_hashes(self):
        with tempfile.TemporaryDirectory() as tmp:
            grids=[]
            for kind in ('load','wind','solar'):
                out=Path(tmp)/f'{kind}.tsv'
                r=self.ingest(PREP/f'{kind}.csv',PREP/'manifest.json',out,kind)
                self.assertEqual(r.returncode,0,r.stderr)
                rows=[line.split('\t') for line in out.read_text().splitlines()]
                self.assertEqual(len(rows),5832+96)
                self.assertTrue(all(r[1]!='NA' for r in rows[:5832]))
                self.assertTrue(all(r[1]=='NA' for r in rows[5832:]))
                grids.append([r[0] for r in rows])
                meta=json.loads(out.with_suffix('.json').read_text())
                self.assertEqual(meta['canonical_sha256'],hashlib.sha256(out.read_bytes()).hexdigest())
                self.assertEqual(meta['source_provenance']['series_kind'],kind)
            self.assertEqual(grids[0],grids[1]);self.assertEqual(grids[1],grids[2])

    def test_tampering_and_wrong_kind(self):
        with tempfile.TemporaryDirectory() as tmp:
            csv=Path(tmp)/'load.csv'; csv.write_bytes((PREP/'load.csv').read_bytes()+b'\n')
            r=self.ingest(csv,PREP/'manifest.json',Path(tmp)/'out.tsv')
            self.assertIn('SHA256',r.stderr); self.assertNotEqual(r.returncode,0)
            r=self.ingest(PREP/'wind.csv',PREP/'manifest.json',Path(tmp)/'out.tsv')
            self.assertIn('kind',r.stderr); self.assertNotEqual(r.returncode,0)

    def test_invalid_manifest_and_coverage(self):
        with tempfile.TemporaryDirectory() as tmp:
            m=json.loads((PREP/'manifest.json').read_text())
            for key,value in [('output_unit','MWh'),('interval_seconds',1800),('rows',1),('end_exclusive_utc','2026-09-01T00:00:00Z')]:
                bad=dict(m);bad[key]=value
                manifest=Path(tmp)/'manifest.json';manifest.write_text(json.dumps(bad))
                r=self.ingest(PREP/'load.csv',manifest,Path(tmp)/'out.tsv')
                self.assertNotEqual(r.returncode,0,key)
            r=self.ingest(PREP/'load.csv',PREP/'manifest.json',Path(tmp)/'out.tsv',start='2026-01-01')
            self.assertNotEqual(r.returncode,0)

    def test_offsets_and_native_interval_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            csv=Path(tmp)/'load.csv'
            original=(PREP/'load.csv').read_text()
            manifest=Path(tmp)/'manifest.json'
            for timestamp,duration,expected in [('2026-01-01T00:00:00+01:00','3600',0),('2025-12-31T23:00:00Z','1800',1)]:
                lines=original.splitlines();cells=lines[1].split(',');cells[:2]=[timestamp,duration];lines[1]=','.join(cells)
                csv.write_text('\n'.join(lines)+'\n')
                m=json.loads((PREP/'manifest.json').read_text());m['outputs']['load.csv']=hashlib.sha256(csv.read_bytes()).hexdigest()
                manifest.write_text(json.dumps(m))
                r=self.ingest(csv,manifest,Path(tmp)/'out.tsv')
                self.assertEqual(r.returncode,expected,r.stderr)
                if not expected:self.assertEqual((Path(tmp)/'out.tsv').read_text().split('\t')[0],'1767222000')

    def test_legacy_csv_coverage_ignores_missing_edges(self):
        with tempfile.TemporaryDirectory() as tmp:
            csv=Path(tmp)/'legacy.csv';csv.write_text('timestamp,duration_seconds,value\n0,3600,NA\n3600,3600,0\n7200,3600,NA\n')
            out=Path(tmp)/'out.tsv'
            r=subprocess.run([str(BIN),'ingest','--csv',str(csv),'--from','1970-01-01','--to','1970-01-01T03:00:00Z','--output',str(out)],capture_output=True,text=True)
            self.assertEqual(r.returncode,0,r.stderr)
            meta=json.loads(out.with_suffix('.json').read_text())
            self.assertEqual(meta['observed_from'],3600)
            self.assertEqual(meta['observed_to_exclusive'],7200)
            self.assertNotIn('source_provenance',meta)

    def test_source_manifest_requires_csv(self):
        r=subprocess.run([str(BIN),'ingest','--source-manifest',str(PREP/'manifest.json'),'--from','2026-01-01','--to','2026-02-01','--output','unused.tsv'],capture_output=True,text=True)
        self.assertNotEqual(r.returncode,0)
        self.assertIn('--csv',r.stderr)

    def test_missing_and_zero_remain_distinct(self):
        with tempfile.TemporaryDirectory() as tmp:
            csv=Path(tmp)/'solar.csv'
            lines=(PREP/'solar.csv').read_text().splitlines()
            cells=lines[1].split(',');cells[2]='NA';lines[1]=','.join(cells)
            cells=lines[2].split(',');cells[2]='0';lines[2]=','.join(cells)
            csv.write_text('\n'.join(lines)+'\n')
            m=json.loads((PREP/'manifest.json').read_text());m['outputs']['solar.csv']=hashlib.sha256(csv.read_bytes()).hexdigest()
            manifest=Path(tmp)/'manifest.json';manifest.write_text(json.dumps(m))
            out=Path(tmp)/'out.tsv';r=self.ingest(csv,manifest,out,'solar')
            self.assertEqual(r.returncode,0,r.stderr)
            rows=out.read_text().splitlines()
            self.assertEqual(rows[0].split('\t')[1],'NA');self.assertEqual(rows[1].split('\t')[1],'0')
            self.assertEqual(json.loads(out.with_suffix('.json').read_text())['missing_hours'],1)

if __name__=='__main__':unittest.main()
