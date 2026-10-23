"""
Conservation des données personnelles (RGPD, art. 5.1.e : limitation de la conservation).

    python -m src.data.retention            # affiche la politique et la règle en place
    python -m src.data.retention --apply    # applique la politique de configs/config.yaml

Les images d'incohérences (bucket « captures », préfixe mismatches/) montrent le
véhicule et sa plaque. Elles ne servent qu'à la validation par un opérateur :
au-delà de `retention.captures_days`, elles sont supprimées.

  - Stockage S3 (MinIO, AWS S3) : une règle de cycle de vie est posée sur le
    bucket. C'est le stockage lui-même qui supprime, même application arrêtée.
    API S3 standard : la même commande vaut pour MinIO et pour AWS.
  - Stockage local : pas de cycle de vie ; --apply supprime les fichiers échus.

Commande d'administration, lancée à l'installation (ou quand la durée change),
pas au démarrage de l'application : sur AWS, la clé de l'application n'a pas
à pouvoir modifier la configuration des buckets (moindre privilège).
Idempotente : la configuration de cycle de vie du bucket est remplacée, pas complétée.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from configs.settings import RetentionSettings, S3Settings, get_settings  # noqa: E402

RULE_ID = "rgpd-expiration-captures"


# ═══════════════════════════════════════════════════════════════════════════════
# POLITIQUE
# ═══════════════════════════════════════════════════════════════════════════════

def lifecycle_configuration(policy: RetentionSettings) -> Dict[str, Any]:
    """Règle S3 : expiration des objets du préfixe après N jours."""
    if policy.captures_days < 1:
        raise ValueError("captures_days doit valoir au moins 1 jour")
    return {
        "Rules": [{
            "ID": RULE_ID,
            "Status": "Enabled",
            "Filter": {"Prefix": policy.captures_prefix},
            "Expiration": {"Days": int(policy.captures_days)},
        }]
    }


# ═══════════════════════════════════════════════════════════════════════════════
# STOCKAGE S3 (MinIO, AWS S3)
# ═══════════════════════════════════════════════════════════════════════════════

def s3_client(s3: S3Settings):
    import boto3

    return boto3.client(
        "s3",
        endpoint_url=s3.endpoint or None,          # vide = AWS S3 standard
        aws_access_key_id=s3.access_key or None,
        aws_secret_access_key=s3.secret_key or None,
        region_name=s3.region,
    )


def current_rules(client, bucket: str) -> List[Dict[str, Any]]:
    """Règles de cycle de vie en place sur le bucket ([] si aucune)."""
    try:
        return client.get_bucket_lifecycle_configuration(Bucket=bucket).get("Rules", [])
    except Exception as exc:  # noqa: BLE001
        code = getattr(exc, "response", {}).get("Error", {}).get("Code", "")
        if code in ("NoSuchLifecycleConfiguration", "NoSuchLifecycle"):
            return []
        raise


def apply_lifecycle(client, bucket: str, policy: RetentionSettings) -> List[Dict[str, Any]]:
    """Pose la règle (remplace la configuration du bucket) et renvoie les règles relues."""
    client.put_bucket_lifecycle_configuration(
        Bucket=bucket, LifecycleConfiguration=lifecycle_configuration(policy)
    )
    return current_rules(client, bucket)


# ═══════════════════════════════════════════════════════════════════════════════
# STOCKAGE LOCAL
# ═══════════════════════════════════════════════════════════════════════════════

def expired_local_files(captures_dir: Path, policy: RetentionSettings,
                        now: Optional[float] = None) -> List[Path]:
    """Fichiers du préfixe plus anciens que la durée de conservation."""
    root = Path(captures_dir) / policy.captures_prefix
    if not root.exists():
        return []
    limit = (now if now is not None else time.time()) - policy.captures_days * 86400
    return sorted(p for p in root.rglob("*") if p.is_file() and p.stat().st_mtime < limit)


def purge_local(captures_dir: Path, policy: RetentionSettings, now: Optional[float] = None) -> List[Path]:
    files = expired_local_files(captures_dir, policy, now)
    for path in files:
        path.unlink()
    return files


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def describe(rules: List[Dict[str, Any]]) -> str:
    if not rules:
        return "aucune règle (conservation illimitée)"
    lines = []
    for r in rules:
        prefix = r.get("Filter", {}).get("Prefix", r.get("Prefix", ""))
        days = r.get("Expiration", {}).get("Days", "?")
        lines.append(f"[{r.get('Status')}] {r.get('ID')} : {prefix or '(tout le bucket)'} supprimé après {days} jours")
    return "\n  ".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Conservation des données personnelles (RGPD)")
    parser.add_argument("--apply", action="store_true", help="Appliquer la politique (sinon : afficher)")
    args = parser.parse_args(argv)

    settings = get_settings()
    policy, s3 = settings.retention, settings.s3
    print(f"Politique : {s3.bucket_captures}/{policy.captures_prefix} conservé {policy.captures_days} jours")

    if s3.mode == "s3":
        client, bucket = s3_client(s3), s3.bucket_captures
        target = s3.endpoint or "AWS S3"
        if args.apply:
            print(f"✅ Règle appliquée sur {bucket} ({target}) :\n  {describe(apply_lifecycle(client, bucket, policy))}")
        else:
            print(f"Règles en place sur {bucket} ({target}) :\n  {describe(current_rules(client, bucket))}")
            print("(--apply pour appliquer la politique)")
        return 0

    captures_dir = Path(s3.local_path) / "captures"
    if args.apply:
        removed = purge_local(captures_dir, policy)
        print(f"✅ Stockage local : {len(removed)} fichier(s) échu(s) supprimé(s) dans {captures_dir}")
    else:
        expired = expired_local_files(captures_dir, policy)
        print(f"Stockage local : {len(expired)} fichier(s) échu(s) dans {captures_dir} (--apply pour supprimer)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
