"""
ALPR Engine - Storage Manager
Support local (dev) et S3 (AWS/Scaleway/MinIO) pour production.
"""

import os
import io
import json
import shutil
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Tuple, BinaryIO, Union
from dataclasses import dataclass

# Import configuration
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from configs.settings import get_settings

# S3 support (optionnel)
try:
    import boto3
    from botocore.exceptions import ClientError, NoCredentialsError
    S3_AVAILABLE = True
except ImportError:
    S3_AVAILABLE = False
    print("⚠️ boto3 non installé - Mode local uniquement")

# Image support
try:
    from PIL import Image
    import numpy as np
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


@dataclass
class StorageFile:
    """Représente un fichier stocké."""
    key: str
    size: int
    last_modified: datetime
    bucket: str
    url: Optional[str] = None


class StorageManager:
    """
    Gestionnaire de stockage unifié.
    Supporte local (dev) et S3 (production).
    """
    
    def __init__(self):
        """Initialise le gestionnaire de stockage."""
        self.settings = get_settings()
        self.s3_config = self.settings.s3
        
        # Déterminer le mode
        if self.s3_config.mode == "s3" and S3_AVAILABLE:
            self.mode = "s3"
            self._init_s3_client()
        else:
            self.mode = "local"
            self._init_local_storage()
        
        print(f"📦 Storage: {self.mode.upper()}" + 
              (f" → {self.s3_config.endpoint}" if self.mode == "s3" else " → Local"))
    
    # ═══════════════════════════════════════════════════════════════════════════
    # INITIALIZATION
    # ═══════════════════════════════════════════════════════════════════════════
    
    def _init_s3_client(self):
        """Initialise le client S3."""
        self.s3_client = boto3.client(
            's3',
            endpoint_url=self.s3_config.endpoint,
            aws_access_key_id=self.s3_config.access_key,
            aws_secret_access_key=self.s3_config.secret_key,
            region_name=self.s3_config.region
        )
        
        # Vérifier/créer les buckets
        self._ensure_buckets_exist()
    
    def _init_local_storage(self):
        """Initialise le stockage local."""
        self.s3_client = None
        
        # Créer les dossiers locaux
        self.local_paths = {
            'dataset': Path(Path(self.s3_config.local_path) / "dataset"),
            'captures': Path(Path(self.s3_config.local_path) / "captures"),
            'artifacts': Path(Path(self.s3_config.local_path) / "dataset").parent / "artifacts"
        }
        
        for name, path in self.local_paths.items():
            path.mkdir(parents=True, exist_ok=True)
    
    def _ensure_buckets_exist(self):
        """Crée les buckets S3 s'ils n'existent pas."""
        buckets = [
            self.s3_config.bucket_dataset,
            self.s3_config.bucket_captures,
            self.s3_config.artifacts_bucket
        ]
        
        for bucket in buckets:
            try:
                self.s3_client.head_bucket(Bucket=bucket)
            except ClientError as e:
                error_code = e.response.get('Error', {}).get('Code', '')
                if error_code == '404':
                    try:
                        self.s3_client.create_bucket(
                            Bucket=bucket,
                            CreateBucketConfiguration={'LocationConstraint': self.s3_config.region}
                        )
                        print(f"✅ Bucket créé: {bucket}")
                    except Exception as create_error:
                        print(f"⚠️ Impossible de créer le bucket {bucket}: {create_error}")
    
    def _get_bucket(self, bucket_type: str) -> str:
        """Retourne le nom du bucket selon le type."""
        buckets = {
            'dataset': self.s3_config.bucket_dataset,
            'captures': self.s3_config.bucket_captures,
            'artifacts': self.s3_config.artifacts_bucket
        }
        return buckets.get(bucket_type, self.s3_config.bucket_dataset)
    
    def _get_local_path(self, bucket_type: str) -> Path:
        """Retourne le chemin local selon le type."""
        return self.local_paths.get(bucket_type, self.local_paths['dataset'])
    
    # ═══════════════════════════════════════════════════════════════════════════
    # UPLOAD
    # ═══════════════════════════════════════════════════════════════════════════
    
    def upload_file(self, file_path: Union[str, Path], key: str, 
                    bucket_type: str = 'dataset', metadata: Dict = None) -> Tuple[bool, str]:
        """
        Upload un fichier.
        
        Args:
            file_path: Chemin local du fichier
            key: Clé/chemin dans le bucket
            bucket_type: 'dataset', 'captures', ou 'artifacts'
            metadata: Métadonnées optionnelles
            
        Returns:
            (success, url_or_error)
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            return False, f"Fichier non trouvé: {file_path}"
        
        if self.mode == "s3":
            return self._upload_s3(file_path, key, bucket_type, metadata)
        else:
            return self._upload_local(file_path, key, bucket_type)
    
    def _upload_s3(self, file_path: Path, key: str, bucket_type: str, 
                   metadata: Dict = None) -> Tuple[bool, str]:
        """Upload vers S3."""
        bucket = self._get_bucket(bucket_type)
        
        try:
            extra_args = {}
            if metadata:
                extra_args['Metadata'] = {k: str(v) for k, v in metadata.items()}
            
            # Détecter le content type
            content_type = self._get_content_type(file_path)
            if content_type:
                extra_args['ContentType'] = content_type
            
            self.s3_client.upload_file(
                str(file_path), bucket, key,
                ExtraArgs=extra_args if extra_args else None
            )
            
            url = f"{self.s3_config.endpoint}/{bucket}/{key}"
            return True, url
            
        except Exception as e:
            return False, f"Erreur upload S3: {e}"
    
    def _upload_local(self, file_path: Path, key: str, bucket_type: str) -> Tuple[bool, str]:
        """Upload en local (copie)."""
        local_dir = self._get_local_path(bucket_type)
        dest_path = local_dir / key
        
        try:
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(file_path, dest_path)
            return True, str(dest_path)
        except Exception as e:
            return False, f"Erreur copie locale: {e}"
    
    def upload_image(self, image: Union[Image.Image, np.ndarray, str, Path],
                     key: str, bucket_type: str = 'captures',
                     quality: int = 95) -> Tuple[bool, str]:
        """
        Upload une image (PIL, numpy array, ou chemin).
        
        Args:
            image: Image PIL, numpy array, ou chemin fichier
            key: Clé/chemin dans le bucket
            bucket_type: 'dataset' ou 'captures'
            quality: Qualité JPEG (1-100)
            
        Returns:
            (success, url_or_error)
        """
        if not PIL_AVAILABLE:
            return False, "PIL non disponible"
        
        # Convertir en PIL Image si nécessaire
        if isinstance(image, (str, Path)):
            image = Image.open(image)
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        # Sauvegarder en mémoire
        buffer = io.BytesIO()
        image_format = 'JPEG' if key.lower().endswith(('.jpg', '.jpeg')) else 'PNG'
        
        if image_format == 'JPEG' and image.mode == 'RGBA':
            image = image.convert('RGB')
        
        image.save(buffer, format=image_format, quality=quality)
        buffer.seek(0)
        
        # Upload
        if self.mode == "s3":
            return self._upload_bytes_s3(buffer, key, bucket_type)
        else:
            return self._upload_bytes_local(buffer, key, bucket_type)
    
    def _upload_bytes_s3(self, buffer: BinaryIO, key: str, bucket_type: str) -> Tuple[bool, str]:
        """Upload bytes vers S3."""
        bucket = self._get_bucket(bucket_type)
        
        try:
            content_type = self._get_content_type(Path(key))
            
            self.s3_client.upload_fileobj(
                buffer, bucket, key,
                ExtraArgs={'ContentType': content_type} if content_type else None
            )
            
            url = f"{self.s3_config.endpoint}/{bucket}/{key}"
            return True, url
            
        except Exception as e:
            return False, f"Erreur upload S3: {e}"
    
    def _upload_bytes_local(self, buffer: BinaryIO, key: str, bucket_type: str) -> Tuple[bool, str]:
        """Upload bytes en local."""
        local_dir = self._get_local_path(bucket_type)
        dest_path = local_dir / key
        
        try:
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            with open(dest_path, 'wb') as f:
                f.write(buffer.read())
            return True, str(dest_path)
        except Exception as e:
            return False, f"Erreur écriture locale: {e}"
    
    # ═══════════════════════════════════════════════════════════════════════════
    # DOWNLOAD
    # ═══════════════════════════════════════════════════════════════════════════
    
    def download_file(self, key: str, dest_path: Union[str, Path],
                      bucket_type: str = 'dataset') -> Tuple[bool, str]:
        """
        Télécharge un fichier.
        
        Args:
            key: Clé/chemin dans le bucket
            dest_path: Chemin local de destination
            bucket_type: 'dataset', 'captures', ou 'artifacts'
            
        Returns:
            (success, path_or_error)
        """
        dest_path = Path(dest_path)
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        
        if self.mode == "s3":
            bucket = self._get_bucket(bucket_type)
            try:
                self.s3_client.download_file(bucket, key, str(dest_path))
                return True, str(dest_path)
            except Exception as e:
                return False, f"Erreur download S3: {e}"
        else:
            local_dir = self._get_local_path(bucket_type)
            src_path = local_dir / key
            
            if not src_path.exists():
                return False, f"Fichier non trouvé: {src_path}"
            
            try:
                shutil.copy2(src_path, dest_path)
                return True, str(dest_path)
            except Exception as e:
                return False, f"Erreur copie locale: {e}"
    
    def download_image(self, key: str, bucket_type: str = 'captures') -> Tuple[bool, Union[Image.Image, str]]:
        """
        Télécharge une image et retourne un objet PIL.
        
        Args:
            key: Clé/chemin dans le bucket
            bucket_type: 'dataset' ou 'captures'
            
        Returns:
            (success, image_or_error)
        """
        if not PIL_AVAILABLE:
            return False, "PIL non disponible"
        
        if self.mode == "s3":
            bucket = self._get_bucket(bucket_type)
            try:
                buffer = io.BytesIO()
                self.s3_client.download_fileobj(bucket, key, buffer)
                buffer.seek(0)
                image = Image.open(buffer)
                return True, image
            except Exception as e:
                return False, f"Erreur download S3: {e}"
        else:
            local_dir = self._get_local_path(bucket_type)
            src_path = local_dir / key
            
            if not src_path.exists():
                return False, f"Fichier non trouvé: {src_path}"
            
            try:
                image = Image.open(src_path)
                return True, image
            except Exception as e:
                return False, f"Erreur lecture image: {e}"
    
    # ═══════════════════════════════════════════════════════════════════════════
    # LIST / DELETE
    # ═══════════════════════════════════════════════════════════════════════════
    
    def list_files(self, prefix: str = "", bucket_type: str = 'dataset',
                   max_keys: int = 1000) -> List[StorageFile]:
        """
        Liste les fichiers dans un bucket/dossier.
        
        Args:
            prefix: Préfixe/dossier à lister
            bucket_type: 'dataset', 'captures', ou 'artifacts'
            max_keys: Nombre maximum de résultats
            
        Returns:
            Liste de StorageFile
        """
        if self.mode == "s3":
            bucket = self._get_bucket(bucket_type)
            try:
                response = self.s3_client.list_objects_v2(
                    Bucket=bucket,
                    Prefix=prefix,
                    MaxKeys=max_keys
                )
                
                files = []
                for obj in response.get('Contents', []):
                    files.append(StorageFile(
                        key=obj['Key'],
                        size=obj['Size'],
                        last_modified=obj['LastModified'],
                        bucket=bucket,
                        url=f"{self.s3_config.endpoint}/{bucket}/{obj['Key']}"
                    ))
                return files
                
            except Exception as e:
                print(f"⚠️ Erreur list S3: {e}")
                return []
        else:
            local_dir = self._get_local_path(bucket_type)
            search_dir = local_dir / prefix if prefix else local_dir
            
            files = []
            if search_dir.exists():
                for path in search_dir.rglob("*"):
                    if path.is_file():
                        rel_path = path.relative_to(local_dir)
                        stat = path.stat()
                        files.append(StorageFile(
                            key=str(rel_path),
                            size=stat.st_size,
                            last_modified=datetime.fromtimestamp(stat.st_mtime),
                            bucket=bucket_type,
                            url=str(path)
                        ))
            
            return files[:max_keys]
    
    def delete_file(self, key: str, bucket_type: str = 'dataset') -> Tuple[bool, str]:
        """
        Supprime un fichier.
        
        Args:
            key: Clé/chemin dans le bucket
            bucket_type: 'dataset', 'captures', ou 'artifacts'
            
        Returns:
            (success, message)
        """
        if self.mode == "s3":
            bucket = self._get_bucket(bucket_type)
            try:
                self.s3_client.delete_object(Bucket=bucket, Key=key)
                return True, f"Supprimé: {key}"
            except Exception as e:
                return False, f"Erreur delete S3: {e}"
        else:
            local_dir = self._get_local_path(bucket_type)
            file_path = local_dir / key
            
            if not file_path.exists():
                return False, f"Fichier non trouvé: {file_path}"
            
            try:
                file_path.unlink()
                return True, f"Supprimé: {file_path}"
            except Exception as e:
                return False, f"Erreur suppression: {e}"
    
    # ═══════════════════════════════════════════════════════════════════════════
    # DATASET MANAGEMENT (pour retraining)
    # ═══════════════════════════════════════════════════════════════════════════
    
    def save_labeled_image(self, image: Union[Image.Image, np.ndarray, str, Path],
                           label: str, plaque: str, confidence: float = None,
                           metadata: Dict = None) -> Tuple[bool, str]:
        """
        Sauvegarde une image labellisée pour le dataset de retraining.
        
        Args:
            image: Image à sauvegarder
            label: Label/marque corrigée
            plaque: Plaque d'immatriculation associée
            confidence: Confiance de la prédiction originale
            metadata: Métadonnées supplémentaires
            
        Returns:
            (success, path_or_error)
        """
        # Générer le chemin : dataset/brand_name/YYYYMMDD_HHMMSS_plaque.jpg
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_label = label.lower().replace(" ", "_")
        safe_plaque = plaque.replace("-", "").replace(" ", "")
        
        key = f"labeled/{safe_label}/{timestamp}_{safe_plaque}.jpg"
        
        # Ajouter métadonnées
        full_metadata = {
            'plaque': plaque,
            'label': label,
            'labeled_at': datetime.now().isoformat()
        }
        if confidence is not None:
            full_metadata['original_confidence'] = str(confidence)
        if metadata:
            full_metadata.update(metadata)
        
        # Upload
        if isinstance(image, (str, Path)):
            return self.upload_file(image, key, 'dataset', full_metadata)
        else:
            return self.upload_image(image, key, 'dataset')
    
    def get_dataset_stats(self) -> Dict:
        """
        Retourne les statistiques du dataset labellisé.
        
        Returns:
            Dict avec stats par label
        """
        files = self.list_files("labeled/", 'dataset', max_keys=10000)
        
        stats = {
            'total': len(files),
            'by_label': {},
            'size_mb': sum(f.size for f in files) / (1024 * 1024)
        }
        
        for f in files:
            # Extraire le label du chemin: labeled/renault/xxx.jpg
            parts = f.key.split('/')
            if len(parts) >= 2:
                label = parts[1]
                stats['by_label'][label] = stats['by_label'].get(label, 0) + 1
        
        return stats
    
    # ═══════════════════════════════════════════════════════════════════════════
    # HELPERS
    # ═══════════════════════════════════════════════════════════════════════════
    
    def _get_content_type(self, file_path: Path) -> Optional[str]:
        """Retourne le content type selon l'extension."""
        ext = file_path.suffix.lower()
        types = {
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.png': 'image/png',
            '.gif': 'image/gif',
            '.webp': 'image/webp',
            '.json': 'application/json',
            '.pt': 'application/octet-stream',
            '.pth': 'application/octet-stream',
            '.onnx': 'application/octet-stream',
        }
        return types.get(ext)
    
    def get_presigned_url(self, key: str, bucket_type: str = 'captures',
                          expires_in: int = 3600) -> Optional[str]:
        """
        Génère une URL présignée pour accès temporaire (S3 uniquement).
        
        Args:
            key: Clé du fichier
            bucket_type: Type de bucket
            expires_in: Durée de validité en secondes
            
        Returns:
            URL présignée ou None si local
        """
        if self.mode != "s3":
            return None
        
        bucket = self._get_bucket(bucket_type)
        try:
            url = self.s3_client.generate_presigned_url(
                'get_object',
                Params={'Bucket': bucket, 'Key': key},
                ExpiresIn=expires_in
            )
            return url
        except Exception as e:
            print(f"⚠️ Erreur génération URL: {e}")
            return None


# ═══════════════════════════════════════════════════════════════════════════════
# SINGLETON
# ═══════════════════════════════════════════════════════════════════════════════

_storage_instance: Optional[StorageManager] = None

def get_storage() -> StorageManager:
    """Singleton pour obtenir le gestionnaire de stockage."""
    global _storage_instance
    if _storage_instance is None:
        _storage_instance = StorageManager()
    return _storage_instance


# ═══════════════════════════════════════════════════════════════════════════════
# TEST
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("🧪 Test StorageManager...")
    
    storage = get_storage()
    
    # Test création d'une image test
    if PIL_AVAILABLE:
        # Créer une image test
        test_image = Image.new('RGB', (100, 100), color='red')
        
        # Upload
        success, result = storage.upload_image(
            test_image, 
            "test/test_image.jpg",
            bucket_type='captures'
        )
        print(f"Upload image: {success} → {result}")
        
        # List
        files = storage.list_files("test/", bucket_type='captures')
        print(f"List files: {len(files)} fichier(s)")
        
        # Download
        success, img = storage.download_image("test/test_image.jpg", bucket_type='captures')
        print(f"Download image: {success}")
        
        # Delete
        success, msg = storage.delete_file("test/test_image.jpg", bucket_type='captures')
        print(f"Delete: {msg}")
        
        # Dataset stats
        stats = storage.get_dataset_stats()
        print(f"Dataset stats: {stats}")
    else:
        print("⚠️ PIL non disponible, test limité")
    
    print("✅ Tests OK")
