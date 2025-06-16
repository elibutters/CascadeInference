"""
The strategies module provides different methods for checking agreement
between multiple LLM responses.
"""

from .base import AgreementStrategy
from .strict import StrictAgreement
from .semantic import SemanticAgreement
from .remote_semantic import RemoteSemanticAgreement

__all__ = [
    "AgreementStrategy",
    "StrictAgreement",
    "SemanticAgreement",
    "RemoteSemanticAgreement",
] 