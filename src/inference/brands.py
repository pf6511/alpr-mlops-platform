"""Règles de comparaison des marques, partagées par tous les points d'entrée."""

from __future__ import annotations

from typing import Optional

# Variantes d'écriture ramenées au nom de classe du modèle
BRAND_ALIASES = {
    "vw": "volkswagen",
    "mercedes-benz": "mercedes",
    "mercedes benz": "mercedes",
    "citroën": "citroen",
    "škoda": "skoda",
}


def normalize_brand(brand: str) -> str:
    """Nom de marque en minuscules, variantes ramenées au nom canonique."""
    key = brand.strip().lower()
    return BRAND_ALIASES.get(key, key)


def is_brand_mismatch(predicted: Optional[str], declared: Optional[str]) -> bool:
    """Vrai si une marque a été prédite ET déclarée, et qu'elles diffèrent."""
    if not predicted or not declared:
        return False
    return normalize_brand(predicted) != normalize_brand(declared)
