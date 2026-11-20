# -*- coding: utf-8 -*-
"""CLI 端到端：mock 冒烟、产物完整性、契约失败返回 2。"""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


class TestCLI(unittest.TestCase):
    def test_mock_smoke(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "p4-smoke"
            proc = subprocess.run(
                [sys.executable, "-m", "src.evaluation", "--mock", "--output-dir", str(out)],
                cwd=str(ROOT), capture_output=True, text=True)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            for name in ("comparison.csv", "aggregate.csv", "comparison.md",
                         "environment.json", "reproduction.json"):
                self.assertTrue((out / name).is_file(), name)
            rows = (out / "comparison.csv").read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(rows), 7)  # 表头 + 2 方法 x 3 seed

            agg = (out / "aggregate.csv").read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(agg), 3)  # 表头 + 2 组

    def test_tampered_predictions_exit_2(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            proc = subprocess.run(
                [sys.executable, "-m", "src.evaluation", "--mock",
                 "--output-dir", str(tmp_path / "smoke")],
                cwd=str(ROOT), capture_output=True, text=True)
            self.assertEqual(proc.returncode, 0, proc.stderr)

            run_dir = ROOT / "fixtures" / "runs" / "baseline-s42-test"
            data_dir = ROOT / "fixtures" / "data" / "mock-corgi-v1"
            pred = run_dir / "predictions.jsonl"
            backup = pred.read_bytes()
            try:
                rows = [json.loads(x) for x in backup.decode("utf-8").splitlines() if x.strip()]
                rows[0]["score"] = float("nan")
                with pred.open("w", encoding="utf-8", newline="\n") as fh:
                    for row in rows:
                        fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                proc = subprocess.run(
                    [sys.executable, "-m", "src.evaluation", "evaluate",
                     "--data-dir", str(data_dir), "--split", "test",
                     "--run-dir", str(run_dir)],
                    cwd=str(ROOT), capture_output=True, text=True)
                self.assertEqual(proc.returncode, 2, proc.stdout)
                self.assertIn("契约错误", proc.stderr)
            finally:
                pred.write_bytes(backup)

    def test_compare_rejects_mock_by_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            subprocess.run([sys.executable, "-m", "src.evaluation", "--mock",
                            "--output-dir", str(tmp_path / "smoke")],
                           cwd=str(ROOT), capture_output=True, text=True)
            data_dir = ROOT / "fixtures" / "data" / "mock-corgi-v1"
            runs = [str(ROOT / "fixtures" / "runs" / f"{m}-s42-test")
                    for m in ("baseline", "improved")]
            proc = subprocess.run(
                [sys.executable, "-m", "src.evaluation", "compare",
                 "--run-dirs", *runs, "--data-dir", str(data_dir),
                 "--output-dir", str(tmp_path / "cmp")],
                cwd=str(ROOT), capture_output=True, text=True)
            self.assertEqual(proc.returncode, 2, proc.stdout)
            self.assertIn("mock", proc.stderr)


if __name__ == "__main__":
    unittest.main()
