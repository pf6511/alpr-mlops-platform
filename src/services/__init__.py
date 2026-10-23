"""
Couche métier ALPR : les cas d'usage, indépendants de toute interface.

    from src.services import build_services
    services = build_services()
    outcome = services.capture.process(image_bgr)

Dépendances autorisées : services → src.inference, src.data, src.models.
Jamais l'inverse, et jamais vers src.ui ou Gradio.
"""

from .capture import AccessDecision, ALPRService, CaptureOutcome, decide_access
from .container import Services, build_services
from .history import HistoryService
from .residents import ResidentService
from .system import SystemService
from .validation import REJECT, VALIDATE_DECLARED, VALIDATE_PREDICTED, ValidationService

__all__ = [
    "ALPRService",
    "AccessDecision",
    "CaptureOutcome",
    "HistoryService",
    "REJECT",
    "ResidentService",
    "Services",
    "SystemService",
    "VALIDATE_DECLARED",
    "VALIDATE_PREDICTED",
    "ValidationService",
    "build_services",
    "decide_access",
]
