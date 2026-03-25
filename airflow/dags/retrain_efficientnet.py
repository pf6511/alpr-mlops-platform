"""
ALPR Engine - DAG Retraining EfficientNet
Retraining automatique du classificateur de marques.

Trigger: Quotidien ou quand dataset_size >= MIN_DATASET_SIZE
"""

import os
import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any

# Airflow imports
from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.empty import EmptyOperator
from airflow.utils.trigger_rule import TriggerRule

# Ajouter le projet au path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

DEFAULT_ARGS = {
    'owner': 'alpr-engine',
    'depends_on_past': False,
    'start_date': datetime(2025, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Paramètres d'entraînement
TRAINING_CONFIG = {
    'model_name': 'efficientnet_b4',
    'batch_size': 32,
    'epochs': 20,
    'learning_rate': 0.001,
    'weight_decay': 1e-4,
    'early_stopping_patience': 5,
    'min_accuracy_improvement': 0.01,  # 1% minimum pour promotion
}


# ═══════════════════════════════════════════════════════════════════════════════
# TASKS
# ═══════════════════════════════════════════════════════════════════════════════

def check_dataset_ready(**context) -> str:
    """
    Vérifie si le dataset est prêt pour le retraining.
    
    Returns:
        'prepare_dataset' si prêt, 'skip_training' sinon
    """
    from configs.settings import get_settings
    from src.models.mismatch_detector import get_mismatch_detector
    
    settings = get_settings()
    detector = get_mismatch_detector()
    
    stats = detector.get_dataset_stats()
    min_size = settings.airflow.min_dataset_size
    
    total_images = stats.get('total', 0)
    
    print(f"📊 Dataset stats: {total_images} images (minimum: {min_size})")
    print(f"   Par marque: {stats.get('by_label', {})}")
    
    if total_images >= min_size:
        print("✅ Dataset prêt pour retraining")
        return 'prepare_dataset'
    else:
        print(f"⏳ Dataset insuffisant ({total_images}/{min_size})")
        return 'skip_training'


def prepare_dataset(**context) -> Dict:
    """
    Prépare le dataset pour l'entraînement.
    
    Returns:
        Dict avec chemins train/val/test
    """
    from configs.settings import get_settings
    from src.data.storage import get_storage
    
    import tempfile
    import shutil
    from sklearn.model_selection import train_test_split
    
    settings = get_settings()
    storage = get_storage()
    
    print("📦 Préparation du dataset...")
    
    # Créer un dossier temporaire
    work_dir = Path(tempfile.mkdtemp(prefix="alpr_training_"))
    train_dir = work_dir / "train"
    val_dir = work_dir / "val"
    test_dir = work_dir / "test"
    
    # Récupérer les images labellisées
    files = storage.list_files("labeled/", bucket_type='dataset', max_keys=10000)
    
    print(f"   {len(files)} images trouvées")
    
    # Organiser par classe
    by_class = {}
    for f in files:
        # labeled/renault/xxx.jpg -> renault
        parts = f.key.split('/')
        if len(parts) >= 2:
            class_name = parts[1]
            if class_name not in by_class:
                by_class[class_name] = []
            by_class[class_name].append(f)
    
    print(f"   {len(by_class)} classes: {list(by_class.keys())}")
    
    # Split train/val/test (70/15/15)
    stats = {'train': 0, 'val': 0, 'test': 0}
    
    for class_name, class_files in by_class.items():
        # Créer les dossiers
        (train_dir / class_name).mkdir(parents=True, exist_ok=True)
        (val_dir / class_name).mkdir(parents=True, exist_ok=True)
        (test_dir / class_name).mkdir(parents=True, exist_ok=True)
        
        # Split
        if len(class_files) < 3:
            # Pas assez d'images, tout en train
            train_files = class_files
            val_files = []
            test_files = []
        else:
            train_files, temp_files = train_test_split(
                class_files, test_size=0.3, random_state=42
            )
            if len(temp_files) >= 2:
                val_files, test_files = train_test_split(
                    temp_files, test_size=0.5, random_state=42
                )
            else:
                val_files = temp_files
                test_files = []
        
        # Télécharger et organiser
        for f in train_files:
            dest = train_dir / class_name / Path(f.key).name
            storage.download_file(f.key, dest, bucket_type='dataset')
            stats['train'] += 1
        
        for f in val_files:
            dest = val_dir / class_name / Path(f.key).name
            storage.download_file(f.key, dest, bucket_type='dataset')
            stats['val'] += 1
        
        for f in test_files:
            dest = test_dir / class_name / Path(f.key).name
            storage.download_file(f.key, dest, bucket_type='dataset')
            stats['test'] += 1
    
    print(f"   Train: {stats['train']}, Val: {stats['val']}, Test: {stats['test']}")
    
    # Sauvegarder dans XCom
    result = {
        'work_dir': str(work_dir),
        'train_dir': str(train_dir),
        'val_dir': str(val_dir),
        'test_dir': str(test_dir),
        'num_classes': len(by_class),
        'class_names': list(by_class.keys()),
        'stats': stats
    }
    
    context['ti'].xcom_push(key='dataset_info', value=result)
    
    return result


def train_model(**context) -> Dict:
    """
    Entraîne le modèle EfficientNet.
    
    Returns:
        Dict avec métriques et chemin du modèle
    """
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms
    
    try:
        import timm
        TIMM_AVAILABLE = True
    except ImportError:
        TIMM_AVAILABLE = False
    
    from src.models.mlflow_client import get_mlflow_manager, MLflowManager
    
    # Récupérer les infos dataset
    ti = context['ti']
    dataset_info = ti.xcom_pull(key='dataset_info', task_ids='prepare_dataset')
    
    train_dir = Path(dataset_info['train_dir'])
    val_dir = Path(dataset_info['val_dir'])
    num_classes = dataset_info['num_classes']
    class_names = dataset_info['class_names']
    
    print(f"🚀 Entraînement EfficientNet B4")
    print(f"   Classes: {num_classes}")
    print(f"   Train: {train_dir}")
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"   Device: {device}")
    
    # Transforms
    train_transform = transforms.Compose([
        transforms.Resize((380, 380)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((380, 380)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Datasets
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=val_transform)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=TRAINING_CONFIG['batch_size'],
        shuffle=True, 
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=TRAINING_CONFIG['batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Modèle
    if TIMM_AVAILABLE:
        model = timm.create_model('efficientnet_b4', pretrained=True, num_classes=num_classes)
    else:
        from torchvision.models import efficientnet_b4, EfficientNet_B4_Weights
        model = efficientnet_b4(weights=EfficientNet_B4_Weights.IMAGENET1K_V1)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    
    model = model.to(device)
    
    # Loss & optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=TRAINING_CONFIG['learning_rate'],
        weight_decay=TRAINING_CONFIG['weight_decay']
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=2
    )
    
    # MLflow
    mlflow_manager = get_mlflow_manager()
    run_id = mlflow_manager.start_run(
        MLflowManager.EXPERIMENT_BRAND_CLASSIFICATION,
        run_name=f"retrain_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        tags={'trigger': 'airflow', 'num_classes': str(num_classes)}
    )
    
    mlflow_manager.log_params({
        'model': TRAINING_CONFIG['model_name'],
        'batch_size': TRAINING_CONFIG['batch_size'],
        'epochs': TRAINING_CONFIG['epochs'],
        'learning_rate': TRAINING_CONFIG['learning_rate'],
        'num_classes': num_classes,
        'train_size': len(train_dataset),
        'val_size': len(val_dataset)
    })
    
    # Training loop
    best_accuracy = 0.0
    best_model_state = None
    patience_counter = 0
    
    for epoch in range(TRAINING_CONFIG['epochs']):
        # Train
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
        
        train_accuracy = train_correct / train_total
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        val_accuracy = val_correct / val_total
        
        print(f"   Epoch {epoch+1}/{TRAINING_CONFIG['epochs']}: "
              f"Train Acc: {train_accuracy:.4f}, Val Acc: {val_accuracy:.4f}")
        
        # Log metrics
        mlflow_manager.log_metrics({
            'train_loss': train_loss / len(train_loader),
            'train_accuracy': train_accuracy,
            'val_loss': val_loss / len(val_loader),
            'val_accuracy': val_accuracy,
            'learning_rate': optimizer.param_groups[0]['lr']
        }, step=epoch)
        
        # Scheduler
        scheduler.step(val_accuracy)
        
        # Best model
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= TRAINING_CONFIG['early_stopping_patience']:
            print(f"   Early stopping at epoch {epoch+1}")
            break
    
    # Sauvegarder le meilleur modèle
    model.load_state_dict(best_model_state)
    
    model_path = Path(dataset_info['work_dir']) / "best_model.pth"
    torch.save(best_model_state, model_path)
    
    # Log model to MLflow
    mlflow_manager.log_metrics({'best_accuracy': best_accuracy})
    mlflow_manager.log_artifact(str(model_path))
    
    # Register model
    model_uri = mlflow_manager.log_model(
        model,
        "efficientnet_brand_classifier",
        registered_name=MLflowManager.MODEL_EFFICIENTNET
    )
    
    mlflow_manager.end_run()
    
    result = {
        'run_id': run_id,
        'model_path': str(model_path),
        'model_uri': model_uri,
        'best_accuracy': best_accuracy,
        'num_classes': num_classes,
        'class_names': class_names
    }
    
    ti.xcom_push(key='training_result', value=result)
    
    print(f"✅ Entraînement terminé: accuracy={best_accuracy:.4f}")
    
    return result


def evaluate_and_compare(**context) -> str:
    """
    Évalue le nouveau modèle et compare avec la production.
    
    Returns:
        'promote_model' si meilleur, 'skip_promotion' sinon
    """
    from src.models.mlflow_client import get_mlflow_manager, MLflowManager
    
    ti = context['ti']
    training_result = ti.xcom_pull(key='training_result', task_ids='train_model')
    
    new_accuracy = training_result['best_accuracy']
    
    print(f"📊 Évaluation du nouveau modèle")
    print(f"   Nouvelle accuracy: {new_accuracy:.4f}")
    
    # Comparer avec le modèle en production
    mlflow_manager = get_mlflow_manager()
    production_model = mlflow_manager.get_production_model(MLflowManager.MODEL_EFFICIENTNET)
    
    if production_model is None:
        print("   Pas de modèle en production - promotion automatique")
        return 'promote_model'
    
    prod_accuracy = production_model.accuracy
    print(f"   Accuracy production: {prod_accuracy:.4f}")
    
    improvement = new_accuracy - prod_accuracy
    min_improvement = TRAINING_CONFIG['min_accuracy_improvement']
    
    print(f"   Amélioration: {improvement:+.4f} (minimum requis: {min_improvement})")
    
    if improvement >= min_improvement:
        print("✅ Nouveau modèle meilleur - promotion")
        return 'promote_model'
    else:
        print("⏭️ Amélioration insuffisante - pas de promotion")
        return 'skip_promotion'


def promote_model(**context):
    """Promeut le nouveau modèle en production."""
    from src.models.mlflow_client import get_mlflow_manager, MLflowManager
    
    ti = context['ti']
    training_result = ti.xcom_pull(key='training_result', task_ids='train_model')
    
    mlflow_manager = get_mlflow_manager()
    
    # Récupérer la dernière version
    versions = mlflow_manager.get_model_versions(MLflowManager.MODEL_EFFICIENTNET)
    
    if not versions:
        print("⚠️ Aucune version trouvée")
        return
    
    # Trier par version (plus récente en premier)
    latest = sorted(versions, key=lambda v: int(v.version), reverse=True)[0]
    
    print(f"🚀 Promotion du modèle v{latest.version}")
    
    # Promouvoir en staging d'abord
    success, msg = mlflow_manager.promote_to_staging(
        MLflowManager.MODEL_EFFICIENTNET,
        latest.version
    )
    print(f"   Staging: {msg}")
    
    # Puis en production
    success, msg = mlflow_manager.promote_to_production(
        MLflowManager.MODEL_EFFICIENTNET,
        latest.version,
        archive_current=True
    )
    print(f"   Production: {msg}")
    
    print("✅ Promotion terminée")


def cleanup(**context):
    """Nettoie les fichiers temporaires."""
    import shutil
    
    ti = context['ti']
    dataset_info = ti.xcom_pull(key='dataset_info', task_ids='prepare_dataset')
    
    if dataset_info and 'work_dir' in dataset_info:
        work_dir = Path(dataset_info['work_dir'])
        if work_dir.exists():
            shutil.rmtree(work_dir)
            print(f"🧹 Nettoyage: {work_dir}")


# ═══════════════════════════════════════════════════════════════════════════════
# DAG DEFINITION
# ═══════════════════════════════════════════════════════════════════════════════

with DAG(
    dag_id='alpr_retrain_efficientnet',
    default_args=DEFAULT_ARGS,
    description='Retraining automatique du classificateur de marques EfficientNet',
    schedule_interval='@daily',  # Ou None pour trigger manuel uniquement
    catchup=False,
    tags=['alpr', 'ml', 'retraining'],
) as dag:
    
    # Start
    start = EmptyOperator(task_id='start')
    
    # Check if dataset is ready
    check_dataset = BranchPythonOperator(
        task_id='check_dataset',
        python_callable=check_dataset_ready,
        provide_context=True
    )
    
    # Skip training
    skip_training = EmptyOperator(task_id='skip_training')
    
    # Prepare dataset
    prepare = PythonOperator(
        task_id='prepare_dataset',
        python_callable=prepare_dataset,
        provide_context=True
    )
    
    # Train model
    train = PythonOperator(
        task_id='train_model',
        python_callable=train_model,
        provide_context=True
    )
    
    # Evaluate and decide
    evaluate = BranchPythonOperator(
        task_id='evaluate_and_compare',
        python_callable=evaluate_and_compare,
        provide_context=True
    )
    
    # Promote model
    promote = PythonOperator(
        task_id='promote_model',
        python_callable=promote_model,
        provide_context=True
    )
    
    # Skip promotion
    skip_promotion = EmptyOperator(task_id='skip_promotion')
    
    # Cleanup
    cleanup_task = PythonOperator(
        task_id='cleanup',
        python_callable=cleanup,
        provide_context=True,
        trigger_rule=TriggerRule.ALL_DONE
    )
    
    # End
    end = EmptyOperator(
        task_id='end',
        trigger_rule=TriggerRule.ALL_DONE
    )
    
    # Dependencies
    start >> check_dataset
    
    check_dataset >> skip_training >> end
    check_dataset >> prepare >> train >> evaluate
    
    evaluate >> promote >> cleanup_task >> end
    evaluate >> skip_promotion >> cleanup_task >> end
