import datetime as dt
import json
import subprocess
import sys
from pathlib import Path

from flask import Flask, jsonify, request, render_template

API_TOKEN = "PREDICT"

BASE_DIR = Path(__file__).resolve().parent
FORECAST_FILE = BASE_DIR / "data" / "forecast.json"
UPDATE_SCRIPT = BASE_DIR / "run_pipeline.py"

PYTHON_PATH = BASE_DIR / "venv" / "bin" / "python"


PAGE = "test.html"

app = Flask(__name__)


class InvalidUsage(Exception):
    status_code = 400

    def __init__(self, message, status_code=None, payload=None):
        super().__init__()
        self.message = message
        if status_code is not None:
            self.status_code = status_code
        self.payload = payload

    def to_dict(self):
        rv = dict(self.payload or {})
        rv["message"] = self.message
        return rv


@app.errorhandler(InvalidUsage)
def handle_invalid_usage(error):
    response = jsonify(error.to_dict())
    response.status_code = error.status_code
    return response


def load_forecast():
    print("--> Reading forecast from:", FORECAST_FILE.resolve())

    if not FORECAST_FILE.exists():
        raise InvalidUsage("--> forecast.json not found!", status_code=404)

    with open(FORECAST_FILE, "r", encoding="utf-8") as f:
        return json.load(f)



def run_update_forecast():

    if not UPDATE_SCRIPT.exists():
        raise InvalidUsage("--> run_pipeline.py not found!", status_code=500)

    if not PYTHON_PATH.exists():
        raise InvalidUsage("--> venv python not found!", status_code=500)

    try:
        result = subprocess.run(
            [str(PYTHON_PATH), str(UPDATE_SCRIPT)],
            capture_output=True,
            text=True,
            check=True,
            cwd=str(BASE_DIR)
        )
        return result.stdout

    except subprocess.CalledProcessError as e:
        raise InvalidUsage(
            f"--> forecast update failed --> {e.stderr or e.stdout}",
            status_code=500
        )


#-----------------------------------show html page endpoint--------------------------------------------
@app.route("/")
def home_page():
    return render_template(PAGE)


#-----------------------------------get endpoint--------------------------------------------

@app.route("/content/api/v1/forecast/get", methods=["POST"])
def get_forecast():
    start_dt = dt.datetime.now()
    json_data = request.get_json()

    if not json_data:
        raise InvalidUsage("--> JSON body is required", status_code=400)

    if json_data.get("token") is None:
        raise InvalidUsage("--> token is required", status_code=400)


    token = json_data.get("token")
    if token != API_TOKEN:
        raise InvalidUsage("--> wrong API token", status_code=403)

    region = json_data.get("region", "all")

    forecast_data = load_forecast()
    end_dt = dt.datetime.now()

    if region == "all":
        return {
            "requester_name": "Markovska Taisia",
            "timestamp": end_dt.isoformat() + "Z",
            "last_model_train_time": forecast_data.get("last_model_train_time"),
            "last_prediction_time": forecast_data.get("last_prediction_time"),
            "model_name": forecast_data.get("model_name"),
            "regions_forecast": forecast_data.get("regions_forecast", {})
        }

    region_data = forecast_data.get("regions_forecast", {}).get(region)

    if region_data is None:
        raise InvalidUsage(f"--> region '{region}' not found", status_code=404)

    return {
        "requester_name": "Markovska Taisia",
        "timestamp": end_dt.isoformat() + "Z",
        "last_model_train_time": forecast_data.get("last_model_train_time"),
        "last_prediction_time": forecast_data.get("last_prediction_time"),
        "model_name": forecast_data.get("model_name"),
        "regions_forecast": {
            region: region_data
        }
    }


#-----------------------------------update endpoint--------------------------------------------

@app.route("/content/api/v1/forecast/update", methods=["POST"])
def update_forecast():
    start_dt = dt.datetime.now()
    json_data = request.get_json()

    if not json_data:
        raise InvalidUsage("--> JSON body is required", status_code=400)

    if json_data.get("token") is None:
        raise InvalidUsage("--> token is required", status_code=400)


    token = json_data.get("token")
    if token != API_TOKEN:
        raise InvalidUsage("--> wrong API token", status_code=403)

    stdout = run_update_forecast()
    forecast_data = load_forecast()
    end_dt = dt.datetime.now()

    return {
        "requester_name": "Markovska Taisia",
        "timestamp": end_dt.isoformat() + "Z",
        "message": "forecast updated successfully",
        "script_output": stdout,
        "last_prediction_time": forecast_data.get("last_prediction_time")
    }


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)