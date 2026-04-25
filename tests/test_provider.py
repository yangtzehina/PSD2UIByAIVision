import os
import unittest

from uiir.provider import LLMProviderConfig, missing_api_key_reason, provider_summary, resolve_base_url


class ProviderTests(unittest.TestCase):
    def test_provider_summary_uses_custom_env_without_exposing_key(self):
        old_key = os.environ.get("THIRD_PARTY_API_KEY")
        old_base = os.environ.get("UIIR_OPENAI_BASE_URL")
        os.environ["THIRD_PARTY_API_KEY"] = "secret-value"
        os.environ["UIIR_OPENAI_BASE_URL"] = "https://gateway.example.test/v1"
        try:
            summary = provider_summary(
                LLMProviderConfig(
                    provider_name="third-party",
                    api_key_env="THIRD_PARTY_API_KEY",
                    api_mode="chat_completions",
                )
            )

            self.assertEqual(summary["provider_name"], "third-party")
            self.assertEqual(summary["api_key_env"], "THIRD_PARTY_API_KEY")
            self.assertEqual(summary["api_mode"], "chat-completions")
            self.assertTrue(summary["api_key_present"])
            self.assertEqual(summary["base_url"], "https://gateway.example.test/v1")
            self.assertNotIn("secret-value", str(summary))
        finally:
            _restore_env("THIRD_PARTY_API_KEY", old_key)
            _restore_env("UIIR_OPENAI_BASE_URL", old_base)

    def test_explicit_base_url_wins_over_env(self):
        old_base = os.environ.get("OPENAI_BASE_URL")
        os.environ["OPENAI_BASE_URL"] = "https://env.example.test/v1"
        try:
            base_url = resolve_base_url(
                LLMProviderConfig(
                    provider_name="proxy",
                    api_key_env="PROXY_API_KEY",
                    base_url="https://cli.example.test/v1",
                )
            )

            self.assertEqual(base_url, "https://cli.example.test/v1")
        finally:
            _restore_env("OPENAI_BASE_URL", old_base)

    def test_missing_key_reason_uses_selected_env_name(self):
        reason = missing_api_key_reason(LLMProviderConfig(api_key_env="OPENROUTER_API_KEY"))

        self.assertEqual(reason, "OPENROUTER_API_KEY is not set")

    def test_unknown_api_mode_is_rejected(self):
        with self.assertRaises(ValueError):
            LLMProviderConfig(api_mode="legacy").normalized()


def _restore_env(name: str, old_value: str | None) -> None:
    if old_value is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = old_value


if __name__ == "__main__":
    unittest.main()
