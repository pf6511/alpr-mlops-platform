"""
MLOps : registre de modèles et cycle de vie (seul paquet qui parle à MLflow).

    registry.py   versions, alias (champion / challenger), téléchargement sûr
    bundle.py     format du paquet de modèle (poids sans code + classes + manifeste)
    bootstrap.py  python -m src.mlops.bootstrap   le modèle local devient champion
    promote.py    python -m src.mlops.promote     challenger contre champion, même test
"""
