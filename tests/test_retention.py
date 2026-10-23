"""Conservation des données personnelles (RGPD) : règle de cycle de vie et purge locale."""

import os

import pytest

from configs.settings import RetentionSettings, get_settings
from src.data import retention

POLICY = RetentionSettings(captures_days=30, captures_prefix="mismatches/")


class NoLifecycle(Exception):
    response = {"Error": {"Code": "NoSuchLifecycleConfiguration"}}


class FakeS3:
    """Imite les deux appels de cycle de vie de boto3."""

    def __init__(self):
        self.config = None

    def get_bucket_lifecycle_configuration(self, Bucket):
        if self.config is None:
            raise NoLifecycle()
        return self.config

    def put_bucket_lifecycle_configuration(self, Bucket, LifecycleConfiguration):
        self.config = LifecycleConfiguration          # remplace : pas d'empilement


def test_policy_comes_from_config():
    policy = get_settings().retention
    assert policy.captures_days == 30 and policy.captures_prefix == "mismatches/"


def test_lifecycle_rule_targets_mismatch_images_only():
    rule = retention.lifecycle_configuration(POLICY)["Rules"][0]
    assert rule["Filter"] == {"Prefix": "mismatches/"}
    assert rule["Expiration"] == {"Days": 30} and rule["Status"] == "Enabled"
    with pytest.raises(ValueError):
        retention.lifecycle_configuration(RetentionSettings(captures_days=0))


def test_apply_is_idempotent():
    s3 = FakeS3()
    assert retention.current_rules(s3, "alpr-captures") == []           # aucune règle au départ
    retention.apply_lifecycle(s3, "alpr-captures", POLICY)
    rules = retention.apply_lifecycle(s3, "alpr-captures", POLICY)      # deuxième passage
    assert len(rules) == 1 and rules[0]["ID"] == retention.RULE_ID
    assert "30 jours" in retention.describe(rules)


def test_local_purge_removes_only_expired_mismatch_images(tmp_path):
    now = 1_000_000_000
    old = tmp_path / "mismatches" / "20250101" / "AB123CD_120000.jpg"
    recent = tmp_path / "mismatches" / "20250301" / "EF456GH_090000.jpg"
    other = tmp_path / "autre" / "fichier.jpg"                          # hors du préfixe
    for path, age_days in ((old, 31), (recent, 2), (other, 365)):
        path.parent.mkdir(parents=True)
        path.write_bytes(b"jpg")
        os.utime(path, (now - age_days * 86400,) * 2)

    assert retention.purge_local(tmp_path, POLICY, now=now) == [old]
    assert not old.exists() and recent.exists() and other.exists()
