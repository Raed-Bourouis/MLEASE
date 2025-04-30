import sys
from pathlib import Path
import os

project_root = Path(__file__).resolve().parent.parent  # Go up two levels from app.py
sys.path.append(str(project_root))

from flask import Flask, request, jsonify
import Backend.EDA  # example function
import pandas as pd
import mlflow

app = Flask(__name__)

mlflow.set_tracking_uri("http://localhost:5000")


@app.route("/")
def home():
    return "MLEASE API is running!"


@app.route("/upload-csv")
def upload_csv():
    if "file" not in request.files:
        return jsonify({"status": "error", "message": "No file part in request"}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"status": "error", "message": "No selected file"}), 400

    if file and file.filename.endswith(".csv"):
        try:
            # Read the CSV file into a pandas DataFrame
            with mlflow.start_run():

                df = pd.read_csv(file)

                # Optional: process DataFrame
                # result = your_pipeline_function(df)
                temp_path = os.path.join("temp", file.filename)
                os.makedirs("temp", exist_ok=True)
                file.seek(0)
                file.save(temp_path)
                mlflow.log_artifact(temp_path, artifact_path="uploaded_data")
                os.remove(temp_path)

            return jsonify(
                {
                    "status": "success",
                    "columns": df.columns.tolist(),
                    "preview": df.head(5).to_dict(orient="records"),
                }
            )
        except Exception as e:
            return jsonify({"status": "error", "message": str(e)}), 500
    else:
        return jsonify({"status": "error", "message": "Invalid file type"}), 400


@app.route("/forecast", methods=["POST"])
def forecast():
    data = request.get_json()
    try:
        result = Backend.EDA.generate_mlops_report(
            data
        )  # this function should run your pipeline
        return jsonify({"status": "success", "result": result})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, port=8000)
