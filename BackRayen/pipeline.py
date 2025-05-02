import prefect
from prefect import Flow, task, Parameter
import pandas as pd
import mlflow
from EDA import run_eda
from Preprocessing import TimeSeriesPreprocessor
from ModelSelection import analyze_timeseries
from Sarima import train_sarima
from Prophet import train_prophet
from Xgboost import train_xgboost
from Lstm import train_lstm

# Définir les tâches Prefect
@task
def run_eda_task(data_path: str, output_report_path: str):
    with mlflow.start_run(run_name="EDA"):
        df = pd.read_csv(data_path, parse_dates=True, index_col=0)
        run_eda(df, output_report_path)
    return output_report_path

@task
def preprocess_task(data_path: str, eda_report_path: str, output_processed_path: str):
    with mlflow.start_run(run_name="Preprocessing"):
        preprocessor = TimeSeriesPreprocessor(eda_report_path)
        df = pd.read_csv(data_path, parse_dates=True, index_col=0)
        df_processed = preprocessor.preprocess(df)
        df_processed.to_csv(output_processed_path)
    return output_processed_path

@task
def model_selection_task(processed_data_path: str):
    with mlflow.start_run(run_name="ModelSelection"):
        df = pd.read_csv(processed_data_path, parse_dates=True, index_col=0)
        target_col = df.columns[-1]
        result = analyze_timeseries(df, target_col=target_col)
        mlflow.log_param("recommended_approach", result["recommended_approach"])
        if result["recommended_approach"] == "ensemble":
            recommended_models = result["ensemble_details"]["models"]
        else:
            recommended_models = [result["recommended_model"]]
        mlflow.log_param("recommended_models", recommended_models)
    return recommended_models

@task
def train_models_task(processed_data_path: str, selected_models: list):
    with mlflow.start_run(run_name="Training"):
        df = pd.read_csv(processed_data_path, parse_dates=True, index_col=0)

        
        results = {}
        if not selected_models:
            selected_models = ["SARIMA", "Prophet"]

        for model_name in selected_models:
            if model_name == "SARIMA":
                for target_col in df.columns:
                    results["SARIMA"] = train_sarima(df, target_col)
            elif model_name == "Prophet":
                for target_col in df.columns:
                    results["Prophet"] = train_prophet(df, target_col)
            elif model_name == "XGBoost":
                for target_col in df.columns:
                    results["XGBoost"] = train_xgboost(df, target_col)
            elif model_name == "LSTM":
                for target_col in df.columns:
                    results["LSTM"] = train_lstm(df, target_col)
        return results

# Définir le workflow Prefect
with Flow("TimeSeriesPipeline") as flow:
    # Paramètres configurables
    data_path = Parameter("data_path", default="../datasets/Miles_Traveled.csv")
    eda_report_path = Parameter("eda_report_path", default="mlops_eda_report.json")
    processed_data_path = Parameter("processed_data_path", default="preprocessed_timeseries.csv")
    selected_models = Parameter("selected_models", default=None)  # Si None, on utilise ModelSelection

    # Étape 1 : EDA
    eda_output = run_eda_task(data_path, eda_report_path)

    # Étape 2 : Preprocessing
    processed_output = preprocess_task(data_path, eda_output, processed_data_path)

    # Étape 3 : ModelSelection (optionnelle)
    recommended_models = model_selection_task(processed_output)

    # Étape 4 : Entraînement des modèles
    models_to_train = selected_models if selected_models is not None else recommended_models
    train_results = train_models_task(processed_output, recommended_models)

# Exécuter le pipeline
if __name__ == "__main__":
    flow.run()