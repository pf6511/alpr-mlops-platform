"""Règles de comparaison des marques, partagées par tous les points d'entrée."""

from __future__ import annotations

from typing import Optional, Sequence

# Variantes d'écriture ramenées à une forme commune (comparaison des marques).
# Attention : cette forme n'est pas toujours le nom de la classe du modèle
# (« mercedes » contre « mercedes benz ») : voir to_model_class().
BRAND_ALIASES = {
    "vw": "volkswagen",
    "mercedes-benz": "mercedes",
    "mercedes benz": "mercedes",
    "land rover": "landrover",
    "land-rover": "landrover",
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


def to_model_class(label: str, classes: Sequence[str]) -> Optional[str]:
    """
    Classe du modèle correspondant à une étiquette, ou None si la marque est inconnue.

    Les deux côtés sont comparés sous leur forme normalisée : « mercedes »,
    « Mercedes-Benz » et « mercedes benz » donnent toutes la classe « mercedes benz »
    (nom exact du modèle). Utilisée par l'entraînement et par le compteur du
    réentraînement, pour qu'ils retiennent les mêmes images.
    """
    wanted = normalize_brand(label)
    for model_class in classes:
        if normalize_brand(model_class) == wanted:
            return model_class
    return None
