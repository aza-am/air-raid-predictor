import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
import random
import pickle
import pandas as pd
from xgboost import XGBClassifier


BASE_DIR = Path(__file__).resolve().parent

OUTPUT_PATH = BASE_DIR / "data" / "forecast.json"
#MODEL_PATH = BASE_DIR / "models" / "6__xgboost__v2.json"

MODEL_PATH = BASE_DIR / "models" / "6__xgboost__v2.pkl"

DATASET_PATH = BASE_DIR / "data" / "latest_input.csv"
REGIONS_MAP_PATH = BASE_DIR / "data" / "regions_map_final.json"

TEST_DATA_PATH = BASE_DIR / "data" / "test_data.csv"

def load_dataset():
    df = pd.read_csv(DATASET_PATH)

    print(df.columns.tolist())
    return df

def load_region_map():
    with open(REGIONS_MAP_PATH, "r", encoding="utf-8") as f:
        return json.load(f)



def load_model():
    with open(MODEL_PATH, "rb") as f:
        data = pickle.load(f)

    model = data["model"]
    threshold = data["threshold"]
    regional_thresholds = data["regional_thresholds"]


    if hasattr(model, "feature_names_in_"):
        print("Model expects columns:")
        print(model.feature_names_in_)
        print("Number of columns:", len(model.feature_names_in_))

    return model, threshold, regional_thresholds



def build_input_data(df, model):
    
    expected_cols = list(model.feature_names_in_)

    meta_cols = []
    for col in ["region_id", "hour", "datetime"]:
        if col in df.columns:
            meta_cols.append(col)

    df_input = df.copy()

    for col in expected_cols:
        if col not in df_input.columns:
            df_input[col] = 0

    X_new = df_input[expected_cols].copy()
    meta_df = df_input[meta_cols].copy()

    return X_new, meta_df



def predict(model, X_new, meta_df, regional_thresholds, default_threshold):
    probs = model.predict_proba(X_new)[:, 1]

    predictions = []

    for i in range(len(meta_df)):
        region_id = int(meta_df.iloc[i]["region_id"])

        threshold = regional_thresholds.get(region_id, regional_thresholds.get(str(region_id), default_threshold))

        pred = probs[i] >= threshold

        item = {
            "region_id": region_id,
            "prediction": bool(pred),
            "probability": round(float(probs[i]), 4),
            "threshold": round(float(threshold), 4)
        }

        hour_val = int(meta_df.iloc[i]["hour"])
        item["hour_str"] = f"{hour_val:02d}:00"

        predictions.append(item)

    return predictions



def format_forecast(predictions, region_map):
    result = {
        "last_model_train_time": "unknown",
        "last_prediction_time": datetime.now(timezone.utc).isoformat(),
        "model_name": "6__xgboost__v1",
        "regions_forecast": {}
    }

    for item in predictions:
        region_id = str(item["region_id"])
        region = region_map.get(region_id, region_id)
        hour_str = item["hour_str"]

        if region not in result["regions_forecast"]:
            result["regions_forecast"][region] = {}

        result["regions_forecast"][region][hour_str] = {
            "alarm": item["prediction"],
            "probability": item["probability"]
        }

    return result



def save_forecast(data, path=OUTPUT_PATH):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)



def main():
    print("--> Loading model --> \n")
    model, threshold, regional_thresholds = load_model()

    print("--> Loading dataset --> \n")
    df = load_dataset()

    print("--> Building input data --> \n")
    X_new, meta_df = build_input_data(df, model)

    print("X_new shape:", X_new.shape)

    print("--> Running forecast --> \n")
    predictions = predict(
        model,
        X_new,
        meta_df,
        regional_thresholds,
        threshold
    )

    print("--> Formatting result --> \n")
    region_map = load_region_map()
    forecast_data = format_forecast(predictions, region_map)

    print("--> Saving forecast.json --> \n")
    save_forecast(forecast_data)

    print(" --> All Done\n")


if __name__ == "__main__":
    main()
