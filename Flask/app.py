import sys
from pathlib import Path
import os

project_root = Path(__file__).resolve().parent.parent  # Go up two levels from app.py
sys.path.append(str(project_root))

from flask import Flask, request, jsonify

import pandas as pd
import mlflow

from flask_cors import CORS
import sqlite3
import os
import hashlib
import secrets
import jwt
from datetime import datetime, timedelta
from werkzeug.utils import secure_filename

from BackRayen.pipeline import *


app = Flask(__name__)

CORS(app)  # Enable CORS for all routes
app.config["SECRET_KEY"] = secrets.token_hex(16)  # Generate a random secret key

# Database setup
DB_PATH = "users.db"

mlflow.set_tracking_uri("http://localhost:5000")


def init_db():
    """Initialize the database with users table if it doesn't exist"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Create users table
    cursor.execute(
        """
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        email TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """
    )

    conn.commit()
    conn.close()


def hash_password(password):
    """Hash a password with SHA-256"""
    return hashlib.sha256(password.encode()).hexdigest()


def generate_token(user_id):
    """Generate a JWT token for authentication"""
    expiration = datetime.utcnow() + timedelta(days=1)  # Token valid for 1 day
    payload = {"user_id": user_id, "exp": expiration}
    return jwt.encode(payload, app.config["SECRET_KEY"], algorithm="HS256")


def verify_token(token):
    """Verify a JWT token"""
    try:
        payload = jwt.decode(token, app.config["SECRET_KEY"], algorithms=["HS256"])
        return payload["user_id"]
    except jwt.ExpiredSignatureError:
        return None
    except jwt.InvalidTokenError:
        return None


@app.route("/api/signup", methods=["POST"])
def signup():
    """Register a new user"""
    data = request.json

    # Validate required fields
    if not all(k in data for k in ["username", "email", "password"]):
        return jsonify({"error": "Missing required fields"}), 400

    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        # Check if username or email already exists
        cursor.execute(
            "SELECT id FROM users WHERE username = ? OR email = ?",
            (data["username"], data["email"]),
        )
        if cursor.fetchone():
            return jsonify({"error": "Username or email already exists"}), 409

        # Hash the password and insert the new user
        password_hash = hash_password(data["password"])
        cursor.execute(
            "INSERT INTO users (username, email, password_hash) VALUES (?, ?, ?)",
            (data["username"], data["email"], password_hash),
        )

        conn.commit()
        user_id = cursor.lastrowid
        conn.close()

        # Generate a token for the new user
        token = generate_token(user_id)

        return (
            jsonify(
                {
                    "message": "User registered successfully",
                    "token": token,
                    "user": {
                        "id": user_id,
                        "username": data["username"],
                        "email": data["email"],
                    },
                }
            ),
            201,
        )

    except sqlite3.Error as e:
        return jsonify({"error": f"Database error: {str(e)}"}), 500


@app.route("/api/signin", methods=["POST"])
def signin():
    """Authenticate a user"""
    data = request.json

    # Validate required fields
    if not all(k in data for k in ["username", "password"]):
        return jsonify({"error": "Missing required fields"}), 400

    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        # Find user by username
        cursor.execute(
            "SELECT id, username, email, password_hash FROM users WHERE username = ?",
            (data["username"],),
        )
        user = cursor.fetchone()
        conn.close()

        # Check if user exists and password is correct
        if not user or user[3] != hash_password(data["password"]):
            return jsonify({"error": "Invalid username or password"}), 401

        # Generate a token for the authenticated user
        token = generate_token(user[0])

        return (
            jsonify(
                {
                    "message": "Authentication successful",
                    "token": token,
                    "user": {"id": user[0], "username": user[1], "email": user[2]},
                }
            ),
            200,
        )

    except sqlite3.Error as e:
        return jsonify({"error": f"Database error: {str(e)}"}), 500


@app.route("/api/protected", methods=["GET"])
def protected():
    """Example of a protected route that requires authentication"""
    auth_header = request.headers.get("Authorization")

    if not auth_header or not auth_header.startswith("Bearer "):
        return jsonify({"error": "No token provided"}), 401

    token = auth_header.split(" ")[1]
    user_id = verify_token(token)

    if not user_id:
        return jsonify({"error": "Invalid or expired token"}), 401

    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        cursor.execute("SELECT username, email FROM users WHERE id = ?", (user_id,))
        user = cursor.fetchone()
        conn.close()

        if not user:
            return jsonify({"error": "User not found"}), 404

        return (
            jsonify(
                {
                    "message": "You have access to this protected resource",
                    "user": {"username": user[0], "email": user[1]},
                }
            ),
            200,
        )

    except sqlite3.Error as e:
        return jsonify({"error": f"Database error: {str(e)}"}), 500


@app.route("/")
def home():
    return "MLEASE API is running!"


@app.route("/upload-csv", methods=["POST"])
def upload_csv():
    if "file" not in request.files:
        return jsonify({"status": "error", "message": "No file part in request"}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"status": "error", "message": "No selected file"}), 400

    if file and file.filename.endswith(".csv"):
        try:
            # Save dataset to local storage
            filename = secure_filename(file.filename)
            experiment_name = filename.removesuffix(".csv")
            mlflow.set_experiment(experiment_name=experiment_name)

            # log dataset to mlflow as artefact
            with mlflow.start_run(run_name="Shared Resources") as run:
                global run_id
                df = pd.read_csv(file)
                temp_path = os.path.join("temp/datasets/", filename)
                os.makedirs("temp/datasets/", exist_ok=True)
                print("temp path: ", temp_path)
                print("filename: ", filename)
                file.seek(0)
                file.save(temp_path)
                mlflow.log_artifact(temp_path, artifact_path="uploaded_data")
                run_id = run.info.run_id
                # os.remove(temp_path)

            return jsonify(
                {
                    "status": "success",
                    "Dataset_id": run_id,
                    "columns": df.columns.tolist(),
                    "preview": df.head(5).to_dict(orient="records"),
                },
                201,
            )
        except Exception as e:
            return jsonify({"status": "error", "message": str(e)}), 500
    else:
        return jsonify({"status": "error", "message": "Invalid file type"}), 400


@app.route("/run-eda", methods=["POST"])
def run_eda():
    try:
        # Extract parameters from the request (JSON body)
        data = request.get_json()

        data_path = data.get("data_path")  # Path to the dataset (CSV)
        output_report_path = data.get("output_report_path", "mlops_eda_report.json")

        if not data_path or not os.path.exists(data_path):
            return (
                jsonify(
                    {"status": "error", "message": "Valid 'data_path' must be provided"}
                ),
                400,
            )

        # Call the EDA task
        eda_report_path = run_eda_task.run(
            data_path=data_path, output_report_path=output_report_path
        )

        return (
            jsonify(
                {
                    "status": "success",
                    "message": "EDA task completed",
                    "eda_report_path": eda_report_path,
                }
            ),
            200,
        )

    except Exception as e:
        return (
            jsonify(
                {"status": "error", "message": f"EDA task failed to execute: {str(e)}"}
            ),
            500,
        )




@app.route("/run-preprocessing", methods=["POST"])
def run_preprocessing():
    try:
        data = request.get_json()

        data_path = data.get("data_path")
        eda_report_path = data.get("eda_report_path")
        output_processed_path = data.get(
            "output_processed_path", "processed/preprocessed.csv"
        )

        if not data_path or not os.path.exists(data_path):
            return (
                jsonify(
                    {"status": "error", "message": "Valid 'data_path' must be provided"}
                ),
                400,
            )
        if not eda_report_path or not os.path.exists(eda_report_path):
            return (
                jsonify(
                    {
                        "status": "error",
                        "message": "Valid 'eda_report_path' must be provided",
                    }
                ),
                400,
            )

        # Run the preprocessing task
        processed_path = preprocess_task.run(
            data_path, eda_report_path, output_processed_path
        )

        return (
            jsonify({"status": "success", "processed_data_path": processed_path}),
            200,
        )

    except Exception as e:
        return (
            jsonify({"status": "error", "message": f"Preprocessing failed: {str(e)}"}),
            500,
        )



@app.route("/run-model-selection", methods=["POST"])
def run_model_selection():
    try:
        data = request.get_json()
        processed_data_path = data.get("processed_data_path")

        if not processed_data_path or not os.path.exists(processed_data_path):
            return (
                jsonify(
                    {
                        "status": "error",
                        "message": "Valid 'processed_data_path' must be provided",
                    }
                ),
                400,
            )

        # Run model selection task
        recommended_models = model_selection_task.run(
            processed_data_path=processed_data_path
        )

        return (
            jsonify({"status": "success", "recommended_models": recommended_models}),
            200,
        )

    except Exception as e:
        return (
            jsonify(
                {"status": "error", "message": f"Model selection task failed: {str(e)}"}
            ),
            500,
        )



@app.route("/run-training", methods=["POST"])
def run_training():
    try:
        data = request.get_json()

        processed_data_path = data.get("processed_data_path")
        selected_models = data.get("selected_models", [])  # Defaults to SARIMA + Prophet in task

        if not processed_data_path or not os.path.exists(processed_data_path):
            return jsonify({"status": "error", "message": "Valid 'processed_data_path' must be provided"}), 400

        # Run the training task
        training_results = train_models_task.run(processed_data_path=processed_data_path, selected_models=selected_models)

        return jsonify({
            "status": "success",
            "message": "Training completed",
            "results": training_results
        }), 200

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Training task failed: {str(e)}"
        }), 500

@app.route("/run-pipeline", methods=["POST"])
def run_pipeline():
    try:
        # Extract parameters from the request (JSON body)
        data = request.get_json()

        data_path = data.get("data_path")  # Required
        eda_report_path = data.get("eda_report_path", "mlops_eda_report.json")
        processed_data_path = data.get(
            "processed_data_path", "preprocessed_timeseries.csv"
        )
        selected_models = data.get("selected_models")  # Can be None

        if not data_path or not os.path.exists(data_path):
            return (
                jsonify(
                    {"status": "error", "message": "Valid 'data_path' must be provided"}
                ),
                400,
            )

        # Run the pipeline with parameters
        flow_state = mlease_pipeline.run(
            parameters={
                "data_path": data_path,
                "eda_report_path": eda_report_path,
                "processed_data_path": processed_data_path,
            }
        )

        return (
            jsonify(
                {
                    "status": "success",
                    "message": "Pipeline executed",
                    "flow_state": str(flow_state),
                }
            ),
            200,
        )

    except Exception as e:
        return (
            jsonify(
                {"status": "error", "message": f"Pipeline failed to execute: {str(e)}"}
            ),
            500,
        )


# @app.route("/forecast", methods=["POST"])
# def forecast():
#     data = request.get_json()
#     try:
#         # result = Backend.EDA.generate_mlops_report(data)  # this function should run your pipeline
#         result = "true"
#         return jsonify({"status": "success", "result": result})
#     except Exception as e:
#         return jsonify({"status": "error", "message": str(e)}), 500


# Initialize database when the app starts
init_db()

if __name__ == "__main__":
    app.run(debug=True, port=8000)
