import argparse
import json
from pathlib import Path
from datetime import datetime
import mlflow


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset-uri", required=True)
    parser.add_argument("--dataset-version", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--imgsz", type=int, default=224)
    parser.add_argument("--mlflow-tracking-uri", required=True)
    parser.add_argument("--mlflow-experiment", required=True)

    return parser.parse_args()


def simulate_training():
    """
    Placeholder → à remplacer plus tard par EfficientNet réel
    """
    return {
        "accuracy": 0.87,
        "loss": 0.45,
        "f1_score": 0.85,
    }


def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    mlflow.set_tracking_uri(args.mlflow_tracking_uri)
    mlflow.set_experiment(args.mlflow_experiment)

    with mlflow.start_run(
        run_name=f"brand_train_{args.dataset_version}"
    ) as run:

        run_id = run.info.run_id

        # Params
        mlflow.log_params({
            "dataset_uri": args.dataset_uri,
            "dataset_version": args.dataset_version,
            "epochs": args.epochs,
            "imgsz": args.imgsz,
            "task": "vehicle_brand_classification"
        })

        # Training (simulation)
        metrics = simulate_training()

        mlflow.log_metrics(metrics)

        # 🔥 IMPORTANT pour le DAG de promotion
        mlflow.set_tags({
            "candidate": "true",
            "candidate_status": "READY_FOR_PROMOTION",
            "model_name": "vehicle_brand_model"
        })

        # Sauvegarde fichiers pour Airflow
        metrics_path = output_dir / "metrics.json"
        run_info_path = output_dir / "run_info.json"

        run_info = {
            "run_id": run_id,
            "dataset_version": args.dataset_version,
            "model_name": "vehicle_brand_model",
            "status": "READY_FOR_PROMOTION",
            "created_at": datetime.utcnow().isoformat()
        }

        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)

        with open(run_info_path, "w") as f:
            json.dump(run_info, f, indent=2)

        mlflow.log_artifact(str(metrics_path))
        mlflow.log_artifact(str(run_info_path))

        print("===== BRAND TRAINING DONE =====")
        print(metrics)
        print(run_info)


if __name__ == "__main__":
    main()