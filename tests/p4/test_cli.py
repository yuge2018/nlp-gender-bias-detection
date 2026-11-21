import csv
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from src.common.contracts import load_dataset, sha256, write_json
from src.common.model_api import MockPredictor

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "fixtures" / "dataset"


def run_cli(*args):
    return subprocess.run([sys.executable, "-m", "src.evaluation", *args],
                          cwd=str(ROOT), capture_output=True, text=True)


class CLITests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.base = Path(self._tmp.name)

    def test_mock_smoke(self):
        out = self.base / "p4-smoke"
        done = run_cli("--mock", "--output-dir", str(out))
        self.assertEqual(done.returncode, 0, done.stderr)
        for name in ("comparison.csv", "aggregate.csv", "comparison.md",
                     "environment.json", "reproduction.json"):
            self.assertTrue((out / name).is_file(), name)
        with (out / "comparison.csv").open(encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 2)
        self.assertEqual({row["is_mock"] for row in rows}, {"True"})

    def test_mock_requires_empty_output_dir(self):
        out = self.base / "busy"
        out.mkdir()
        (out / "keep.txt").write_text("x", encoding="utf-8")
        done = run_cli("--mock", "--output-dir", str(out))
        self.assertEqual(done.returncode, 2)
        self.assertIn("非空", done.stderr)

    def test_evaluate_tampered_predictions_exit_2(self):
        data = self.base / "dataset"
        shutil.copytree(DATA, data)
        _, rows = load_dataset(data)
        inputs = rows["test.jsonl"]
        scores = MockPredictor().predict_proba([row["text"] for row in inputs])
        run_dir = self.base / "run"
        run_dir.mkdir()
        predictions = [{"id": row["id"], "label": int(score >= 0.5), "score": score}
                       for row, score in zip(inputs, scores)]
        predictions[0]["score"] = float("nan")
        with (run_dir / "predictions.jsonl").open("w", encoding="utf-8", newline="\n") as stream:
            for row in predictions:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        write_json(run_dir / "run_metadata.json", {
            "schema_version": "1.0", "method": "mock", "split": "test",
            "dataset_manifest_sha256": sha256(data / "manifest.json"),
            "is_mock": True, "seed": 42, "threshold": 0.5, "code_commit": "MOCK",
            "model_name": "mock", "model_revision": "mock-v1",
            "config": {}, "environment": {},
        })
        done = run_cli("evaluate", "--data-dir", str(data), "--split", "test",
                       "--run-dir", str(run_dir))
        self.assertEqual(done.returncode, 2, done.stdout)
        self.assertIn("契约错误", done.stderr)

    def test_compare_rejects_mock_by_default(self):
        out = self.base / "smoke"
        self.assertEqual(run_cli("--mock", "--output-dir", str(out)).returncode, 0)
        runs = [str(out / "mock-s42-test"), str(out / "mock-s43-test")]
        done = run_cli("compare", "--run-dirs", *runs,
                       "--output-dir", str(self.base / "results"))
        self.assertEqual(done.returncode, 2)
        self.assertIn("mock", done.stderr)


if __name__ == "__main__":
    unittest.main()
