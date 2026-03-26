"""
ALPR Engine - Ray Serve Deployment Utilities
Scripts de déploiement, monitoring et tests.
"""

import os
import sys
import time
import json
import base64
import argparse
from pathlib import Path
from typing import Dict, Optional

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

try:
    import ray
    from ray import serve
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False


# ═══════════════════════════════════════════════════════════════════════════════
# DEPLOYMENT COMMANDS
# ═══════════════════════════════════════════════════════════════════════════════

def deploy(host: str = "0.0.0.0", port: int = 8000, detach: bool = False):
    """Déploie l'application ALPR sur Ray Serve."""
    if not RAY_AVAILABLE:
        print("❌ Ray non installé: pip install 'ray[serve]'")
        return False
    
    from src.serving.ray_serve import start_server
    
    print(f"🚀 Déploiement ALPR Engine")
    print(f"   http://{host}:{port}")
    
    try:
        start_server(host=host, port=port)
        if not detach:
            print("\n[Ctrl+C pour arrêter]")
            while True:
                time.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 Arrêt...")
        from src.serving.ray_serve import stop_server
        stop_server()
    
    return True


def undeploy():
    """Arrête l'application Ray Serve."""
    if not RAY_AVAILABLE:
        print("❌ Ray non installé")
        return
    
    try:
        from serve.ray_serve import stop_server
        stop_server()
        print("✅ Application arrêtée")
    except Exception as e:
        print(f"⚠️ Erreur: {e}")


def status():
    """Affiche le statut des déploiements."""
    if not RAY_AVAILABLE:
        print("❌ Ray non installé")
        return
    
    try:
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)
        
        deployments = serve.list_deployments()
        
        print("📊 Statut Ray Serve:")
        print("-" * 40)
        
        if not deployments:
            print("   Aucun déploiement actif")
            return
        
        for name, dep in deployments.items():
            print(f"   {name}: {dep.num_replicas} replica(s)")
        
    except Exception as e:
        print(f"⚠️ Erreur: {e}")


def scale(deployment_name: str, num_replicas: int):
    """Scale un deployment."""
    if not RAY_AVAILABLE:
        print("❌ Ray non installé")
        return
    
    try:
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)
        
        serve.get_deployment(deployment_name).options(
            num_replicas=num_replicas
        ).deploy()
        
        print(f"✅ {deployment_name} → {num_replicas} replicas")
        
    except Exception as e:
        print(f"⚠️ Erreur: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# CLIENT
# ═══════════════════════════════════════════════════════════════════════════════

class ALPRClient:
    """Client pour l'API ALPR Ray Serve."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip("/")
    
    def health(self) -> Dict:
        """Health check."""
        if not REQUESTS_AVAILABLE:
            return {"error": "requests non installé"}
        try:
            resp = requests.get(f"{self.base_url}/health", timeout=5)
            return resp.json()
        except Exception as e:
            return {"error": str(e)}
    
    def process(self, image_path: str, declared_brand: str = None,
                return_images: bool = False) -> Dict:
        """Pipeline complet."""
        if not REQUESTS_AVAILABLE:
            return {"error": "requests non installé"}
        
        with open(image_path, "rb") as f:
            image_b64 = base64.b64encode(f.read()).decode('utf-8')
        
        payload = {"image": image_b64, "return_images": return_images}
        if declared_brand:
            payload["declared_brand"] = declared_brand
        
        try:
            resp = requests.post(f"{self.base_url}/", json=payload, timeout=30)
            return resp.json()
        except Exception as e:
            return {"error": str(e)}


def test_endpoint(base_url: str = "http://localhost:8000"):
    """Test les endpoints."""
    print(f"🧪 Test: {base_url}")
    
    client = ALPRClient(base_url)
    result = client.health()
    
    if "error" in result:
        print(f"❌ {result['error']}")
        return False
    
    print(f"✅ Status: {result.get('status', 'unknown')}")
    return True


def benchmark(image_path: str, num_requests: int = 10, 
              base_url: str = "http://localhost:8000"):
    """Benchmark de performance."""
    if not os.path.exists(image_path):
        print(f"❌ Image non trouvée: {image_path}")
        return
    
    client = ALPRClient(base_url)
    
    print(f"🏃 Benchmark: {num_requests} requêtes")
    
    times = []
    for i in range(num_requests):
        start = time.time()
        result = client.process(image_path)
        elapsed = time.time() - start
        
        if "error" not in result:
            times.append(elapsed)
            plates = len(result.get("plates", []))
            brand = result.get("brand", {}).get("name", "N/A")
            print(f"   {i+1}: {elapsed:.3f}s - {plates} plaque(s), {brand}")
        else:
            print(f"   {i+1}: ❌ {result['error']}")
    
    if times:
        print(f"\n📊 Moyenne: {sum(times)/len(times):.3f}s")
        print(f"   Throughput: {len(times)/sum(times):.2f} req/s")


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="ALPR Ray Serve")
    sub = parser.add_subparsers(dest="cmd")
    
    # Deploy
    p = sub.add_parser("deploy")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--detach", action="store_true")
    
    # Autres commandes
    sub.add_parser("undeploy")
    sub.add_parser("status")
    
    p = sub.add_parser("scale")
    p.add_argument("deployment")
    p.add_argument("replicas", type=int)
    
    p = sub.add_parser("test")
    p.add_argument("--url", default="http://localhost:8000")
    
    p = sub.add_parser("benchmark")
    p.add_argument("image")
    p.add_argument("--requests", type=int, default=10)
    p.add_argument("--url", default="http://localhost:8000")
    
    args = parser.parse_args()
    
    if args.cmd == "deploy":
        deploy(args.host, args.port, args.detach)
    elif args.cmd == "undeploy":
        undeploy()
    elif args.cmd == "status":
        status()
    elif args.cmd == "scale":
        scale(args.deployment, args.replicas)
    elif args.cmd == "test":
        test_endpoint(args.url)
    elif args.cmd == "benchmark":
        benchmark(args.image, args.requests, args.url)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
