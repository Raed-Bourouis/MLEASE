import mlflow

mlflow.set_experiment("quick-test")

with mlflow.start_run():
    mlflow.log_param("test_param", 123)
    mlflow.log_metric("test_metric", 0.98)
    print("MLflow test run successful.")