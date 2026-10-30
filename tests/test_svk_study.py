"""Preflight checks must fail before reserving runs or contacting services."""
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
BIN=ROOT/'target/debug/energy-timestral'

class Coverage(unittest.TestCase):
    def test_final_fails_coverage_before_frozen_selection_or_weather(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg=Path(tmp)/'final.toml';out=Path(tmp)/'results'
            cfg.write_text(f'''data = "data/svk_2026/load.tsv"
output = "{out}"
from = "2026-07-01"
to = "2026-10-01"
phase = "final"
features = "calendar-weather"
frozen_selection = "{tmp}/does-not-exist"
''')
            r=subprocess.run([str(BIN),'backtest','--config',str(cfg)],cwd=ROOT,capture_output=True,text=True)
            self.assertNotEqual(r.returncode,0)
            self.assertIn('exceeds observed source coverage',r.stderr)
            self.assertFalse(out.exists())

    def test_calendar_padding_cannot_extend_scoring(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg=Path(tmp)/'padding.toml';out=Path(tmp)/'results'
            cfg.write_text(f'''data = "data/svk_2026/load.tsv"
output = "{out}"
from = "2026-08-31"
to = "2026-09-01"
phase = "final"
''')
            r=subprocess.run([str(BIN),'backtest','--config',str(cfg)],cwd=ROOT,capture_output=True,text=True)
            self.assertIn('exceeds observed source coverage',r.stderr)
            self.assertFalse(out.exists())

    def test_source_workbook_unchanged(self):
        import hashlib
        m=json.loads((ROOT/'data/prepared/svk_2026/manifest.json').read_text())
        self.assertEqual(hashlib.sha256((ROOT/'timvarden-2026-01-08.xls').read_bytes()).hexdigest(),m['source_sha256'])

if __name__=='__main__':unittest.main()
