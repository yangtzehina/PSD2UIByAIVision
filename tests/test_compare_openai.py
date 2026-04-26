import json
import os
import tempfile
import unittest
from pathlib import Path

from uiir.compare_openai import CompareOptions, IterateOptions, _write_experiment_manifest, review_run, run_compare_openai, run_iterate_openai
from uiir.provider import LLMProviderConfig


class CompareOpenAITests(unittest.TestCase):
    def test_compare_openai_skips_without_api_key(self):
        old_key = os.environ.pop("OPENAI_API_KEY", None)
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                report = run_compare_openai(root, root / "out", CompareOptions(limit=2))

                self.assertEqual(report["status"], "skipped")
                self.assertTrue((root / "out" / "comparison.json").exists())
        finally:
            if old_key is not None:
                os.environ["OPENAI_API_KEY"] = old_key

    def test_compare_openai_skips_with_custom_provider_key_env(self):
        old_key = os.environ.pop("THIRD_PARTY_API_KEY", None)
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                report = run_compare_openai(
                    root,
                    root / "out",
                    CompareOptions(
                        limit=2,
                        provider_name="third-party",
                        api_key_env="THIRD_PARTY_API_KEY",
                        base_url="https://gateway.example.test/v1",
                        api_mode="chat-completions",
                    ),
                )

                self.assertEqual(report["status"], "skipped")
                self.assertEqual(report["reason"], "THIRD_PARTY_API_KEY is not set")
                self.assertEqual(report["provider"]["provider_name"], "third-party")
                self.assertEqual(report["provider"]["api_key_env"], "THIRD_PARTY_API_KEY")
                self.assertEqual(report["provider"]["base_url"], "https://gateway.example.test/v1")
                self.assertEqual(report["provider"]["api_mode"], "chat-completions")
                self.assertFalse(report["provider"]["api_key_present"])
        finally:
            if old_key is not None:
                os.environ["THIRD_PARTY_API_KEY"] = old_key

    def test_review_run_writes_findings(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "comparison.json").write_text(
                json.dumps(
                    {
                        "status": "ok",
                        "items": [
                            {
                                "name": "sample",
                                "unknown_delta": 2,
                                "invalid_parent_hints": 1,
                                "pixel_similarity_delta": -0.05,
                                "type_changes": [{"candidate_id": "c1", "before": "Unknown", "after": "Button"}],
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            review = review_run(root)

            self.assertEqual(review["finding_count"], 4)
            self.assertTrue((root / "review.md").exists())

    def test_iterate_openai_skips_without_api_key(self):
        old_key = os.environ.pop("OPENAI_API_KEY", None)
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                report = run_iterate_openai(root, root / "out", IterateOptions(limit=2))

                self.assertEqual(report["status"], "skipped")
                self.assertTrue((root / "out" / "leaderboard.json").exists())
                self.assertTrue((root / "out" / "leaderboard.md").exists())
        finally:
            if old_key is not None:
                os.environ["OPENAI_API_KEY"] = old_key

    def test_experiment_manifest_redacts_base_url_and_token(self):
        old_key = os.environ.get("PROXY_API_KEY")
        os.environ["PROXY_API_KEY"] = "placeholder-value"
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                provider = LLMProviderConfig(
                    provider_name="third-party",
                    api_key_env="PROXY_API_KEY",
                    base_url="https://gateway.example.test/v1",
                    api_mode="chat-completions",
                ).normalized()
                _write_experiment_manifest(
                    root,
                    root,
                    {"items": [{"source": "sample.psd"}]},
                    IterateOptions(provider_name="third-party", api_key_env="PROXY_API_KEY", base_url="https://gateway.example.test/v1"),
                    "semantic_v2",
                    "strict",
                    1.25,
                    provider,
                )

                raw = (root / "experiment_manifest.json").read_text(encoding="utf-8")
                manifest = json.loads(raw)
                self.assertTrue(manifest["api_key_present"])
                self.assertTrue(manifest["base_url_present"])
                self.assertNotIn("placeholder-value", raw)
                self.assertNotIn("https://gateway.example.test/v1", raw)
        finally:
            if old_key is None:
                os.environ.pop("PROXY_API_KEY", None)
            else:
                os.environ["PROXY_API_KEY"] = old_key


if __name__ == "__main__":
    unittest.main()
