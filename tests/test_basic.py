"""
ALPR Engine - Tests
Tests unitaires de base pour valider la structure.
"""

import pytest
import sys
from pathlib import Path

# Ajouter le projet au path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))


class TestImports:
    """Test que tous les imports fonctionnent."""
    
    def test_import_configs(self):
        """Test import configs."""
        from configs.settings import get_settings
        settings = get_settings()
        assert settings is not None
        assert hasattr(settings, 'env')
        assert hasattr(settings, 'database')
    
    def test_import_data(self):
        """Test import data layer."""
        from src.data.database import DatabaseManager
        from src.data.storage import get_storage
        # Just test imports work
        assert DatabaseManager is not None
        assert get_storage is not None
    
    def test_import_models(self):
        """Test import models."""
        from src.models.pipeline import ALPRPipeline
        from src.models.mismatch_detector import MismatchDetector
        from src.models.mlflow_client import MLflowManager
        assert ALPRPipeline is not None
        assert MismatchDetector is not None
        assert MLflowManager is not None
    
    def test_import_utils(self):
        """Test import utils."""
        from src.utils.validation_ui import create_stats_grid
        assert create_stats_grid is not None
    
    def test_import_pipelines(self):
        """Test import pipelines."""
        from pipelines.training_pipeline import TrainingPipeline
        from pipelines.inference_pipeline import InferencePipeline
        assert TrainingPipeline is not None
        assert InferencePipeline is not None


class TestSettings:
    """Test configuration."""
    
    def test_settings_singleton(self):
        """Test que get_settings retourne un singleton."""
        from configs.settings import get_settings
        s1 = get_settings()
        s2 = get_settings()
        assert s1 is s2
    
    def test_settings_defaults(self):
        """Test valeurs par défaut."""
        from configs.settings import get_settings
        settings = get_settings()
        
        assert settings.database.mode in ['sqlite', 'postgres']
        assert settings.s3.mode in ['local', 's3']
        assert settings.gradio.server_port == 7860
    
    def test_brand_classes(self):
        """Test que les classes de marques sont définies."""
        from configs.settings import get_settings
        settings = get_settings()
        
        assert len(settings.models.brand_classes) >= 20
        assert 'renault' in settings.models.brand_classes
        assert 'peugeot' in settings.models.brand_classes


class TestDatabase:
    """Test database manager."""
    
    def test_database_init(self):
        """Test initialisation DB."""
        from src.data.database import DatabaseManager
        
        db = DatabaseManager()
        assert db is not None
    
    def test_database_whitelist(self):
        """Test récupération whitelist."""
        from src.data.database import DatabaseManager
        
        db = DatabaseManager()
        whitelist = db.get_whitelist()
        assert isinstance(whitelist, list)


class TestMismatchDetector:
    """Test mismatch detector."""
    
    def test_detector_singleton(self):
        """Test singleton."""
        from src.models.mismatch_detector import get_mismatch_detector
        
        d1 = get_mismatch_detector()
        d2 = get_mismatch_detector()
        assert d1 is d2
    
    def test_brand_variants(self):
        """Test variantes de marques."""
        from src.models.mismatch_detector import MismatchDetector
        
        detector = MismatchDetector()
        
        # Test normalisation
        assert detector._normalize_brand("VW") == "volkswagen"
        assert detector._normalize_brand("Mercedes-Benz") == "mercedes"
        assert detector._normalize_brand("RENAULT") == "renault"


class TestValidationUI:
    """Test composants UI."""
    
    def test_stats_card(self):
        """Test création carte stats."""
        from src.utils.validation_ui import create_stats_card
        
        html = create_stats_card("Test", "42", "#3b82f6", "📊")
        assert "42" in html
        assert "Test" in html
    
    def test_stats_grid(self):
        """Test création grille stats."""
        from src.utils.validation_ui import create_stats_grid
        
        stats = {
            'total_detected': 10,
            'pending': 5,
            'validated': 3,
            'rejected': 2
        }
        
        html = create_stats_grid(stats)
        assert "10" in html
        assert "5" in html


# ═══════════════════════════════════════════════════════════════════════════════
# RUN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
