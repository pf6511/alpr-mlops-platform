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
        assert MLflowManager is not None
        assert MismatchDetector is not None
    
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




# ═══════════════════════════════════════════════════════════════════════════════
# RUN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    pytest.main([__file__, "-v"])