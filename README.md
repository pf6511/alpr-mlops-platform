alpr-mlops-platform/
│
├── src/
│   ├── data/              # ingestion, preprocessing
│   ├── features/          # feature engineering
│   ├── models/            # training, inference
│   ├── utils/             # fonctions communes
│
├── pipelines/
│   ├── training_pipeline.py
│   ├── inference_pipeline.py
│
├── airflow/
│   └── dags/
│
├── mlflow/
│   └── tracking/
│
├── configs/
│   └── config.yaml
│
├── tests/
│
├── notebooks/             # eda
│
├── docker/
│   └── Dockerfile
│
├── k8s/                    # Kubernetes
│   ├── deployment.yaml
│   ├── service.yaml
│
├── helm/                   # charts Helm
│   └── alpr-chart/
│
├── docs/   
│
├── .gitignore
├── requirements.txt
├── README.md