"""端到端检查第二标注分包工具：拆分、校验、合并、复核。"""
import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT / "tools" / "p5_review"
CHALLENGE = ROOT / "data" / "challenge" / "p5-gender-challenge-v1"
COLUMNS = ["id", "text", "label", "targets", "phenomena", "notes"]


def call(script, *args):
    return subprocess.run(
        [sys.executable, str(TOOLS / script), *map(str, args)],
        cwd=str(ROOT), capture_output=True, text=True, encoding="utf-8", errors="replace",
    )


def read_jsonl(path):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def read_csv(path):
    with open(path, encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def split(out_dir, parts=6):
    return call("split_chunks.py", "--challenge-dir", CHALLENGE, "--output-dir", out_dir, "--parts", parts)


def fake_returned(chunks_dir, returned_dir):
    labels = {row["id"]: row["label"] for row in read_jsonl(CHALLENGE / "challenge_labels.jsonl")}
    annotations = {row["id"]: row for row in read_jsonl(CHALLENGE / "challenge_annotations.jsonl")}
    returned_dir.mkdir(parents=True, exist_ok=True)
    for path in sorted(Path(chunks_dir).glob("chunk-*.csv")):
        rows = read_csv(path)
        for row in rows:
            ann = annotations[row["id"]]
            row.update({
                "label": str(labels[row["id"]]),
                "targets": ";".join(ann["targets"]),
                "phenomena": ";".join(ann["phenomena"]),
                "notes": "自测：按原标注复刻",
            })
        with open(returned_dir / path.name, "w", encoding="utf-8-sig", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=COLUMNS, lineterminator="\r\n")
            writer.writeheader()
            writer.writerows(rows)


class ReviewPackageTest(unittest.TestCase):
    def test_split_covers_every_row_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "chunks"
            proc = split(out)
            self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
            ids = []
            for path in sorted(out.glob("chunk-*.csv")):
                rows = read_csv(path)
                self.assertTrue(16 <= len(rows) <= 17, f"{path.name}: {len(rows)} rows")
                self.assertTrue(all(row["label"] == "" and row["notes"] == "" for row in rows))
                ids.extend(row["id"] for row in rows)
            self.assertEqual(len(ids), 100)
            self.assertEqual(len(set(ids)), 100)
            self.assertEqual(len(read_csv(out / "assignment.csv")), 6)

    def test_validate_rejects_bad_label_and_missing_notes(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "chunks"
            self.assertEqual(split(out).returncode, 0)
            returned = Path(tmp) / "returned"
            fake_returned(out, returned)
            path = returned / "chunk-1.csv"
            rows = read_csv(path)
            rows[0]["label"] = "2"
            rows[1]["notes"] = ""
            with open(path, "w", encoding="utf-8-sig", newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=COLUMNS, lineterminator="\r\n")
                writer.writeheader()
                writer.writerows(rows)
            proc = call("validate_returned.py", "--chunk", out / "chunk-1.csv", "--returned", path)
            self.assertEqual(proc.returncode, 1, proc.stdout + proc.stderr)
            self.assertIn("label must be 0 or 1", proc.stdout)
            self.assertIn("notes is required", proc.stdout)

    def test_merge_then_review_gives_perfect_kappa(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "chunks"
            returned = Path(tmp) / "returned"
            pass2 = Path(tmp) / "pass2"
            self.assertEqual(split(out).returncode, 0)
            fake_returned(out, returned)
            proc = call("merge_annotations.py", "--challenge-dir", CHALLENGE, "--chunks-dir", out, "--returned-dir", returned, "--output-dir", pass2)
            self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
            review = Path(tmp) / "review.json"
            proc = subprocess.run(
                [sys.executable, "-m", "src.fairness", "review", "--first", str(CHALLENGE), "--second", str(pass2), "--output", str(review)],
                cwd=str(ROOT), capture_output=True, text=True, encoding="utf-8", errors="replace",
            )
            self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
            result = json.loads(review.read_text(encoding="utf-8"))
            self.assertEqual(result["label_agreement"], 1.0)
            self.assertEqual(result["label_kappa"], 1.0)
            self.assertEqual(result["n"], 100)
            self.assertEqual(len(result["second"]["reviewers"]), 6)


if __name__ == "__main__":
    unittest.main()
