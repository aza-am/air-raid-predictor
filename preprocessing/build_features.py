import json
import pandas as pd
from pathlib import Path
from datetime import datetime


BASE_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = BASE_DIR / "data" / "raw"
DATA_DIR = BASE_DIR / "data"
TEST_DATA_PATH = DATA_DIR / "test_data.csv"

WEATHER_PATH = RAW_DIR / "latest_weather.json"
OUTPUT_PATH = DATA_DIR / "latest_input.csv"


WEATHER_RENAME = {
    "temp": "hour_temp",
    "feelslike": "hour_feelslike",
    "humidity": "hour_humidity",
    "dew": "hour_dew",
    "precip": "hour_precip",
    "precipprob": "hour_precipprob",
    "snow": "hour_snow",
    "snowdepth": "hour_snowdepth",
    "windgust": "hour_windgust",
    "windspeed": "hour_windspeed",
    "winddir": "hour_winddir",
    "pressure": "hour_pressure",
    "cloudcover": "hour_cloudcover",
    "solarradiation": "hour_solarradiation",
    "solarenergy": "hour_solarenergy",
    "uvindex": "hour_uvindex",
}


def load_weather():
    with open(WEATHER_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def build_weather_features(weather_data):
    rows = []

    for region_id, hours in weather_data.items():
        for hour_data in hours[:24]:
            dt = datetime.fromtimestamp(hour_data["datetimeEpoch"])

            row = {
                "region_id": int(region_id),
                "hour": dt.hour,
                "month": dt.month,
                "is_weekend": int(dt.weekday() >= 5),
                "is_night": int(dt.hour < 6 or dt.hour >= 22),
                "region_id_norm": int(region_id) / 25,
            }

            for api_col, model_col in WEATHER_RENAME.items():
                row[model_col] = hour_data.get(api_col, 0) or 0

            rows.append(row)

    return pd.DataFrame(rows)



def add_last_known_features(df):
    old_df = pd.read_csv(TEST_DATA_PATH)

    feature_cols = [
        col for col in old_df.columns
        if (
            col.startswith("isw_tfidf_")
            or col.startswith("tfidf_")
            or col in [
                "alarm_24h_ago",
                "alarms_lag_24h",
                "alarms_lag_48h",
                "alarms_lag_72h",
                "alarms_lag_168h",
                "alarms_rolling_24h_mean",
            ]
        )
    ]

    result_blocks = []

    for region_id in df["region_id"].unique():
        new_region = df[df["region_id"] == region_id].copy()
        old_region = old_df[old_df["region_id"] == region_id].copy()

        if old_region.empty:
            for col in feature_cols:
                new_region[col] = 0
            result_blocks.append(new_region)
            continue

        old_tail = old_region.tail(24).reset_index(drop=True)
        new_region = new_region.reset_index(drop=True)

        for col in feature_cols:
            if col in old_tail.columns:
                if len(old_tail) >= 24:
                    values = old_tail[col].values[:len(new_region)]
                    new_region[col] = values * (1 + (new_region["hour"] / 100))
                    
                else:
                    new_region[col] = old_tail.iloc[-1][col]
            else:
                new_region[col] = 0

        result_blocks.append(new_region)

    result = pd.concat(result_blocks, axis=0, ignore_index=True)

    return result.copy()


def main():
    weather_data = load_weather()

    df = build_weather_features(weather_data)

    df = add_last_known_features(df)

    print(df[[
        "region_id",
        "hour",
        "alarm_24h_ago",
        "alarms_lag_24h",
        "alarms_lag_48h",
        "alarms_rolling_24h_mean"
    ]].head(30))
    
    print(df.groupby("region_id")[[
        "alarms_lag_24h",
        "alarms_lag_48h",
        "alarms_rolling_24h_mean"
    ]].first())

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    print("--> latest_input.csv saved")
    print("Shape:", df.shape)
    print(df.head())


if __name__ == "__main__":
    main()
