from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any


DEFAULT_PROVIDER_NAME = "openai"
DEFAULT_API_KEY_ENV = "OPENAI_API_KEY"
DEFAULT_API_MODE = "responses"
SUPPORTED_API_MODES = {"responses", "chat-completions"}
BASE_URL_ENV_VARS = ("UIIR_OPENAI_BASE_URL", "OPENAI_BASE_URL")


@dataclass(frozen=True)
class LLMProviderConfig:
    provider_name: str = DEFAULT_PROVIDER_NAME
    api_key_env: str = DEFAULT_API_KEY_ENV
    base_url: str | None = None
    api_mode: str = DEFAULT_API_MODE

    def normalized(self) -> "LLMProviderConfig":
        provider_name = (self.provider_name or DEFAULT_PROVIDER_NAME).strip() or DEFAULT_PROVIDER_NAME
        api_key_env = (self.api_key_env or DEFAULT_API_KEY_ENV).strip() or DEFAULT_API_KEY_ENV
        base_url = (self.base_url or "").strip() or None
        api_mode = (self.api_mode or DEFAULT_API_MODE).strip().lower().replace("_", "-") or DEFAULT_API_MODE
        if api_mode not in SUPPORTED_API_MODES:
            raise ValueError(f"Unsupported API mode {api_mode!r}; expected one of {sorted(SUPPORTED_API_MODES)}")
        return LLMProviderConfig(provider_name=provider_name, api_key_env=api_key_env, base_url=base_url, api_mode=api_mode)


def resolve_api_key(config: LLMProviderConfig) -> str | None:
    normalized = config.normalized()
    return os.getenv(normalized.api_key_env)


def resolve_base_url(config: LLMProviderConfig) -> str | None:
    normalized = config.normalized()
    if normalized.base_url:
        return normalized.base_url
    for env_name in BASE_URL_ENV_VARS:
        value = (os.getenv(env_name) or "").strip()
        if value:
            return value
    return None


def missing_api_key_reason(config: LLMProviderConfig) -> str:
    normalized = config.normalized()
    return f"{normalized.api_key_env} is not set"


def provider_summary(config: LLMProviderConfig) -> dict[str, Any]:
    normalized = config.normalized()
    return {
        "provider_name": normalized.provider_name,
        "api_key_env": normalized.api_key_env,
        "api_key_present": bool(resolve_api_key(normalized)),
        "base_url": resolve_base_url(normalized),
        "api_mode": normalized.api_mode,
    }


def create_openai_compatible_client(config: LLMProviderConfig) -> Any:
    normalized = config.normalized()
    api_key = resolve_api_key(normalized)
    if not api_key:
        raise RuntimeError(f"{missing_api_key_reason(normalized)} when --use-openai is set.")
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("openai package is required for --use-openai.") from exc

    kwargs: dict[str, Any] = {"api_key": api_key}
    base_url = resolve_base_url(normalized)
    if base_url:
        kwargs["base_url"] = base_url
    return OpenAI(**kwargs)
