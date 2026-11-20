"""P5 功能与失败输入测试：挑战集完整性、分组指标、案例、CLI 契约。"""
import json
import tempfile
import unittest
from pathlib import Path

from src.common.contracts import write_json, write_jsonl
from src.fairness import build as challenge_build
from src.fairness.cli import mock_command, report_command, review_command
from src.fairness.data import align_run, check_models_consistent, load_challenge, load_run
from src.fairness.metrics import build_subgroup_metrics, group_metrics


def _mock_run(challenge, out, name, scores, is_mock=True, threshold=0.5):
    run = Path(out) / name
    run.mkdir(parents=True, exist_ok=True)
    predictions = [
        {"id": row["id"], "label": int(score >= threshold), "score": float(score)}
        for row, score in zip(challenge, scores)
    ]
    write_jsonl(run / "predictions.jsonl", predictions)
    write_json(run / "run_metadata.json", {
        "schema_version": "1.0",
        "method": name,
        "split": "challenge",
        "dataset_manifest_sha256": "0" * 64,
        "is_mock": is_mock,
        "seed": 42,
        "threshold": threshold,
        "code_commit": "test",
        "model_name": name,
        "model_revision": "test",
        "config": {},
        "environment": {},
    })
    return run


class ChallengeSetTest(unittest.TestCase):
    def test_build_and_coverage(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "challenge"
            result = challenge_build.build_challenge(out)
            data = load_challenge(out)
            self.assertEqual(result["rows"], 100)
            self.assertFalse(result["composition"]["multi_target"] is None)
            self.assertEqual(data["coverage"]["problems"], [])
            for target in ("male", "female"):
                counts = data["coverage"]["by_target"][target]
                self.assertGreaterEqual(counts["0"] + counts["1"], 20)
                self.assertGreaterEqual(counts["0"], 5)
                self.assertGreaterEqual(counts["1"], 5)
            self.assertIn("double independent review is not complete", " ".join(data["coverage"]["warnings"]))
            self.assertEqual(len({row["id"] for row in data["challenge"]}), 100)

    def test_tampered_challenge_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "challenge"
            challenge_build.build_challenge(out)
            path = out / "challenge.jsonl"
            path.write_text(path.read_text(encoding="utf-8") + '{"id":"extra","text":"x"}\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "sha256 mismatch"):
                load_challenge(out)

    def test_annotation_schema_is_enforced(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "challenge"
            challenge_build.build_challenge(out)
            rows = [json.loads(line) for line in (out / "challenge_annotations.jsonl").read_text(encoding="utf-8").splitlines()]
            rows[0]["targets"] = ["alien"]
            body = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
            (out / "challenge_annotations.jsonl").write_text(body, encoding="utf-8")
            challenge_build.seal_challenge(out)
            with self.assertRaisesRegex(ValueError, "unknown targets"):
                load_challenge(out)


class MetricsTest(unittest.TestCase):
    def test_hand_computed_group_metrics(self):
        gold = [1, 1, 0, 0, 0, 1]
        pred = [1, 0, 1, 0, 0, 1]
        result = group_metrics(gold, pred)
        self.assertEqual((result["tp"], result["fn"], result["fp"], result["tn"]), (2, 1, 1, 2))
        self.assertAlmostEqual(result["fpr"], 1 / 3)
        self.assertAlmostEqual(result["fnr"], 1 / 3)
        self.assertAlmostEqual(result["positive_f1"], 2 * 2 / (4 + 1 + 1))
        self.assertAlmostEqual(result["negative_f1"], 2 * 2 / (4 + 1 + 1))
        self.assertAlmostEqual(result["macro_f1"], result["positive_f1"])

    def test_zero_denominators_are_null(self):
        result = group_metrics([1, 1], [1, 1])
        self.assertIsNone(result["fpr"])
        self.assertIsNone(result["negative_f1"])
        self.assertEqual(result["fnr"], 0.0)
        self.assertEqual(result["macro_f1_defined_classes"], 1.0)

    def test_subgroup_table_uses_multi_target_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "challenge"
            challenge_build.build_challenge(out)
            data = load_challenge(out)
            run = _mock_run(data["challenge"], Path(tmp), "m", [0.9] * len(data["challenge"]))
            aligned = align_run(data["challenge"], load_run(run))
            table = build_subgroup_metrics(data["challenge"], data["labels"], data["annotations"], [{"name": "m", "predictions": aligned}])
            male = next(entry for entry in table if entry["dimension"] == "target" and entry["group"] == "male")
            self.assertEqual(male["n"], data["coverage"]["by_target"]["male"]["0"] + data["coverage"]["by_target"]["male"]["1"])
            self.assertEqual(male["positive_n"] + male["negative_n"], male["n"])


class RunValidationTest(unittest.TestCase):
    def test_label_must_match_threshold(self):
        with tempfile.TemporaryDirectory() as tmp:
            command = Path(tmp)
            challenge = [{"id": "p5-001", "text": "x"}]
            run = _mock_run(challenge, command, "bad", [0.9])
            rows = [json.loads(line) for line in (run / "predictions.jsonl").read_text(encoding="utf-8").splitlines()]
            rows[0]["label"] = 0
            write_jsonl(run / "predictions.jsonl", rows)
            with self.assertRaisesRegex(ValueError, "label contradicts"):
                load_run(run)

    def test_score_outside_unit_interval_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            command = Path(tmp)
            run = _mock_run([{"id": "p5-001", "text": "x"}], command, "bad", [0.5])
            rows = [json.loads(line) for line in (run / "predictions.jsonl").read_text(encoding="utf-8").splitlines()]
            rows[0]["score"] = 1.4
            write_jsonl(run / "predictions.jsonl", rows)
            with self.assertRaisesRegex(ValueError, "score"):
                load_run(run)

    def test_id_mismatch_and_mock_mixing_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "challenge"
            challenge_build.build_challenge(out)
            data = load_challenge(out)
            run = _mock_run(data["challenge"][:5], Path(tmp), "short", [0.5] * 5)
            with self.assertRaisesRegex(ValueError, "predictions for"):
                align_run(data["challenge"], load_run(run))
            real = _mock_run(data["challenge"], Path(tmp), "real", [0.5] * 100, is_mock=False)
            fake = _mock_run(data["challenge"], Path(tmp), "fake", [0.5] * 100, is_mock=True)
            with self.assertRaisesRegex(ValueError, "cannot be mixed"):
                check_models_consistent([load_run(real), load_run(fake)])


class ReportTest(unittest.TestCase):
    def test_report_writes_grouped_metrics_and_cases(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "challenge"
            challenge_build.build_challenge(out)
            data = load_challenge(out)
            labels = [row["label"] for row in data["labels"]]
            scores = [0.8 if label else 0.2 for label in labels]
            run = _mock_run(data["challenge"], Path(tmp), "perfect", scores)
            report = report_command(out, [f"perfect={run}"], Path(tmp) / "report", max_cases=8)
            self.assertEqual(report["cases"], 8)
            written = json.loads((Path(tmp) / "report" / "subgroup_metrics.json").read_text(encoding="utf-8"))
            self.assertTrue(written["is_mock"])
            self.assertEqual(len(written["models"]), 1)
            all_group = next(entry for entry in written["groups"] if entry["dimension"] == "target" and entry["group"] == "all")
            self.assertEqual(all_group["results"]["perfect"]["macro_f1"], 1.0)
            cases = [json.loads(line) for line in (Path(tmp) / "report" / "error_cases.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertTrue(all({"id", "text", "gold", "targets", "phenomena", "case_type", "models", "explanation"} <= set(case) for case in cases))

    def test_require_real_rejects_mock(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "challenge"
            challenge_build.build_challenge(out)
            data = load_challenge(out)
            run = _mock_run(data["challenge"], Path(tmp), "m", [0.5] * 100)
            with self.assertRaisesRegex(ValueError, "mock predictions"):
                report_command(out, [f"m={run}"], Path(tmp) / "report", require_real=True)

    def test_mock_cli_wiring(self):
        with tempfile.TemporaryDirectory() as tmp:
            status = mock_command(Path(tmp) / "p5-smoke")
            self.assertTrue(status["is_mock"])
            files = {path.name for path in (Path(tmp) / "p5-smoke").iterdir()}
            self.assertIn("subgroup_metrics.json", files)
            self.assertIn("status.json", files)
            self.assertIn("log.txt", files)

    def test_config_freezes_targets_and_rules(self):
        from src.fairness.config import load_config

        config = load_config()
        self.assertEqual(config["targets"], ["male", "female", "other_explicit", "unspecified"])
        self.assertFalse(config["use_challenge_for_tuning"])
        self.assertTrue(config["multi_target"])
        self.assertEqual(config["review"]["independent_reviewers_required"], 2)
        self.assertIsNone(config["zero_denominator"])

    def test_two_reviewer_disagreement_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            first = Path(tmp) / "a"
            second = Path(tmp) / "b"
            challenge_build.build_challenge(first)
            challenge_build.build_challenge(second)
            rows = [json.loads(line) for line in (second / "challenge_labels.jsonl").read_text(encoding="utf-8").splitlines()]
            rows[0]["label"] = 1 - rows[0]["label"]
            write_jsonl(second / "challenge_labels.jsonl", rows)
            annotations = [json.loads(line) for line in (second / "challenge_annotations.jsonl").read_text(encoding="utf-8").splitlines()]
            annotations[0]["phenomena"] = ["neutral_description"]
            write_jsonl(second / "challenge_annotations.jsonl", annotations)
            for base, name in ((first, "reviewer-a"), (second, "reviewer-b")):
                rows = [json.loads(line) for line in (base / "challenge_annotations.jsonl").read_text(encoding="utf-8").splitlines()]
                for row in rows:
                    row["reviewers"] = [name]
                write_jsonl(base / "challenge_annotations.jsonl", rows)
                challenge_build.seal_challenge(base)
            result = review_command(first, second, Path(tmp) / "review.json")
            self.assertEqual(result["disagreements"], 1)
            self.assertAlmostEqual(result["label_agreement"], 0.99)
            self.assertIsNotNone(result["label_kappa"])


if __name__ == "__main__":
    unittest.main()
