import pandas as pd
import mlflow
import os

def combine_ensemble_predictions(models_list, output_path="../datasets/predictions/ensemble_predictions.csv"):
    """
    Combine predictions from selected models using simple average.
    Automatically supports both monovariate and multivariate forecasts.

    Args:
        models_list (list): List of model names, e.g., ["Prophet", "Xgboost"]
        output_path (str): Path where to save the ensemble predictions.
    """
    print(f"🔵 Starting combination of models: {models_list}")

    prediction_dataframes = []

    model_to_file = {
        "prophet": "../datasets/predictions/predictions_prophet.csv",
        "xgboost": "../datasets/predictions/predictions_xgboost.csv",
        "sarima": "../datasets/predictions/predictions_sarima.csv",
        "lstm": "../datasets/predictions/predictions_lstm.csv"
    }

    for model in models_list:
        model = model.lower()
        
        if model not in model_to_file:
            raise ValueError(f"Unknown model {model} for ensembling.")

        prediction_file = model_to_file[model]

        # Check if file exists
        if not os.path.isfile(prediction_file):
            raise FileNotFoundError(f"❌ Prediction file not found: {prediction_file}")

        # Load predictions
        df = pd.read_csv(prediction_file)

        # Detect prediction columns automatically
        pred_columns = [col for col in df.columns if col.startswith("yhat")]

        if not pred_columns:
            raise ValueError(f"❌ No prediction columns ('yhat') found in {prediction_file}")

        print(f"✅ Loaded predictions from {model}: {pred_columns}")
        prediction_dataframes.append(df[pred_columns])

    if not prediction_dataframes:
        raise ValueError("❌ No valid prediction files found to combine.")

    # --- Align all predictions by columns ---
    all_predictions = pd.concat(prediction_dataframes, axis=1)

    # Group by column names: for example, 'yhat_sales', 'yhat_traffic', etc.
    grouped = all_predictions.groupby(level=0, axis=1)

    # Average across models for each variable
    ensemble_predictions = grouped.mean()

    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    ensemble_predictions.to_csv(output_path, index=False)
    print(f"✅ Ensemble predictions saved to {output_path}")

    # Log to MLflow
    with mlflow.start_run(run_name="Ensemble_Combination"):
        mlflow.log_param("Ensemble Models", "+".join(models_list))
        mlflow.log_artifact(output_path)

    print("✅ Ensemble predictions logged to MLflow.")

# Standalone usage
# if __name__ == "__main__":
#     example_models = ["Prophet", "Xgboost"] 
#     combine_ensemble_predictions(example_models)
