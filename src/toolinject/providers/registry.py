"""Provider registry for dynamic provider loading."""

from typing import Any, Type

from toolinject.core.config import Settings
from toolinject.providers.base import ModelProvider


# Registry of provider classes
_PROVIDERS: dict[str, Type[ModelProvider]] = {}


def register_provider(name: str, provider_class: Type[ModelProvider]) -> None:
    """Register a provider class."""
    _PROVIDERS[name] = provider_class


def get_provider(
    provider_name: str,
    settings: Settings,
    **kwargs: Any,
) -> ModelProvider:
    """Get a provider instance by name."""
    # Lazy import to avoid circular dependencies
    if not _PROVIDERS:
        _register_default_providers()
    
    if provider_name not in _PROVIDERS:
        raise ValueError(f"Unknown provider: {provider_name}. Available: {list(_PROVIDERS.keys())}")
    
    provider_class = _PROVIDERS[provider_name]
    
    # Get API key from settings
    api_key = ""
    if provider_name == "openai":
        api_key = settings.providers.openai_api_key
    elif provider_name == "anthropic":
        api_key = settings.providers.anthropic_api_key
    elif provider_name == "mistral":
        api_key = settings.providers.mistral_api_key
    elif provider_name == "stub":
        api_key = "stub-key"
    
    if not api_key and provider_name != "stub":
        raise ValueError(f"No API key found for provider: {provider_name}")
    
    return provider_class(api_key=api_key, **kwargs)


def _register_default_providers() -> None:
    """Register built-in providers."""
    from toolinject.providers.stub import StubProvider
    from toolinject.providers.openai_provider import OpenAIProvider
    from toolinject.providers.anthropic_provider import AnthropicProvider
    from toolinject.providers.mistral_provider import MistralProvider
    
    register_provider("stub", StubProvider)
    register_provider("openai", OpenAIProvider)
    register_provider("anthropic", AnthropicProvider)
    register_provider("mistral", MistralProvider)


def list_providers() -> list[str]:
    """List available provider names."""
    if not _PROVIDERS:
        _register_default_providers()
    return list(_PROVIDERS.keys())
