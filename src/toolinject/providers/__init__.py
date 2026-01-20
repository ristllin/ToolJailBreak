"""Model provider adapters."""

from toolinject.providers.base import ModelProvider
from toolinject.providers.registry import get_provider, register_provider

__all__ = ["ModelProvider", "get_provider", "register_provider"]
