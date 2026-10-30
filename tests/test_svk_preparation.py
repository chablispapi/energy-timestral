"""Focused proof for source preparation; no workbook edits or GPU calls."""
from datetime import datetime, timezone
from decimal import Decimal
import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("svk", ROOT / "scripts/prepare_svk_workbook.py")
svk = importlib.util.module_from_spec(spec)
spec.loader.exec_module(svk)


class PreparationTests(unittest.TestCase):
    def test_fixed_standard_time_in_summer(self):
        ts, _ = svk.parse_time("1.7.2026 0:00")
        self.assertEqual(ts, datetime(2026, 6, 30, 23, tzinfo=timezone.utc))

    def test_mixed_excel_and_text_dates(self):
        excel, kind = svk.parse_time(46143.0)  # 1 May 2026
        text, other = svk.parse_time("1.5.2026 0:00")
        self.assertEqual(excel, text)
        self.assertEqual((kind, other), ("excel_date", "text_date"))

    def test_industry_storage_and_losses_are_counted_once(self):
        values = {name: Decimal(0) for name, _ in svk.FIELDS.values()}
        for k, n in zip(svk.METERED + svk.PROFILED, (1, 2, 3, 4, 5, 6, 7)):
            values[k] = Decimal(n)
        values["storage_generation"] = Decimal(999)
        result = svk.derive(values)
        self.assertEqual(result["metered_total"], Decimal(15))
        self.assertEqual(result["profiled_total"], Decimal(13))
        self.assertEqual(result["load"], Decimal(28))

    def test_zero_offshore_is_observed_not_missing(self):
        values = {name: Decimal(0) for name, _ in svk.FIELDS.values()}
        values["wind_onshore"] = Decimal(123)
        self.assertEqual(svk.derive(values)["wind"], Decimal(123))
        values["wind_offshore"] = None
        self.assertIsNone(svk.derive(values)["wind"])

    def test_missing_load_component_does_not_become_zero(self):
        values = {name: Decimal(1) for name, _ in svk.FIELDS.values()}
        values["profiled_losses"] = None
        self.assertIsNone(svk.derive(values)["load"])

    def test_invalid_measurements_rejected(self):
        self.assertIsNone(svk.numeric(""))
        self.assertEqual(svk.numeric(0), Decimal(0))
        for v in (float("nan"), float("inf"), "invalid"):
            with self.assertRaises(ValueError):
                svk.numeric(v)

    def test_real_source_matches_independent_winter_and_summer_records(self):
        records, counts, columns = svk.read_source(ROOT / "timvarden-2026-01-08.xls")
        self.assertEqual(len(records), 5832)
        self.assertEqual(counts["text_date"], 744)
        self.assertEqual(counts["excel_date"], 5088)
        self.assertEqual(columns["metered_industry_gt50"]["column"], "G")
        self.assertEqual(len(svk.reconcile(records, ROOT / "tests/fixtures/svk")), 98)


if __name__ == "__main__":
    unittest.main()
