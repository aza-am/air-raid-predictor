import json
import pandas as pd
from pathlib import Path
from datetime import datetime

import re
import pickle
import stanza

BASE_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = BASE_DIR / "data" / "raw"
DATA_DIR = BASE_DIR / "data"
TEST_DATA_PATH = DATA_DIR / "test_data.csv"

WEATHER_PATH = RAW_DIR / "latest_weather.json"
OUTPUT_PATH = DATA_DIR / "latest_input.csv"

TELEGRAM_PATH = RAW_DIR / "latest_telegram.csv"
TELEGRAM_VECTORIZER_PATH = BASE_DIR / "models" / "vectorizer_tg.pkl"

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
        new_region = df[df["region_id"] == region_id].copy().reset_index(drop=True)
        old_region = old_df[old_df["region_id"] == region_id].copy()

        if old_region.empty:
            for col in feature_cols:
                new_region[col] = 0
            result_blocks.append(new_region)
            continue

        old_tail = old_region.tail(24).reset_index(drop=True)

        if len(old_tail) < 24:
            last_row = old_tail.iloc[-1:]
            while len(old_tail) < 24:
                old_tail = pd.concat([old_tail, last_row], ignore_index=True)

        for col in feature_cols:
            if col in old_tail.columns:
                new_region[col] = old_tail[col].values[:len(new_region)]
            else:
                new_region[col] = 0

        result_blocks.append(new_region)

    return pd.concat(result_blocks, axis=0, ignore_index=True).copy()



ua_stopwords = [
    'а', 'б', 'в', 'г', 'ґ', 'д', 'е', 'є', 'ж', 'з', 'и', 'і', 'ї', 'й',
    'к', 'л', 'м', 'н', 'о', 'п', 'р', 'с', 'т', 'у', 'ф', 'х', 'ц', 'ч',
    'ш', 'щ', 'ь', 'ю', 'я', 'на', 'зі', 'до', 'за', 'під', 'над', 'перед',
    'при', 'про', 'без', 'через', 'для', 'задля', 'від', 'відтак', 'по',
    'понад', 'попід', 'та', 'але', 'алеж', 'чи', 'або', 'що', 'щоб', 'як', 'коли',
    'якщо', 'хоч', 'хоча', 'бо', 'тому що', 'немов', 'немовби', 'немовбито',
    'не', 'ні', 'таки', 'же', 'ось', 'от', 'ачей', 'мабуть', 'навряд',
    'невже', 'хіба', 'тільки', 'лиш', 'лише', 'ти', 'він', 'вона', 'воно',
    'ми', 'ви', 'вони', 'це', 'то', 'той', 'те', 'такі', 'цей', 'ця', 'ці',
    'свій', 'мій', 'твій', 'наш', 'ваш', 'їхній', 'хто', 'який', 'чий',
    'дехто', 'дещо', 'було', 'буду', 'був', 'була', 'були', 'щоб', 'вам',
    'вас', 'весь', 'все', 'всіх', 'завжди', 'навіть', 'адже', 'вздовж',
    'замість', 'поза', 'вниз', 'внизу', 'всередині', 'навколо', 'да',
    'давай', 'більш', 'бути',
]


def clean_telegram_text(text):
    text = re.sub(r'[^\w\s\u0400-\u04FF]', ' ', str(text))
    text = re.sub(r'\s+', ' ', text).strip()
    text = text.lower()
    text = re.sub(r'\d+', ' ', text)
    text = re.sub(r'http\S+|www\S+|t\.me\S+', ' ', text)
    return text


nlp_uk = stanza.Pipeline("uk", processors="tokenize,lemma", verbose=False)


def lemmatize_uk(text):
    doc = nlp_uk(text)
    return " ".join([
        word.lemma
        for sent in doc.sentences
        for word in sent.words
    ])


def build_telegram_features():
    if not TELEGRAM_PATH.exists():
        print("--> Telegram file not found. Telegram features skipped.")
        return pd.DataFrame()

    if not TELEGRAM_VECTORIZER_PATH.exists():
        print("--> Telegram vectorizer not found. Telegram features skipped.")
        return pd.DataFrame()

    df_telegram = pd.read_csv(TELEGRAM_PATH)

    if df_telegram.empty or "Text" not in df_telegram.columns:
        print("--> Telegram file is empty or has no Text column.")
        return pd.DataFrame()

    df_telegram["text_clean"] = df_telegram["Text"].apply(clean_telegram_text)

    df_telegram["tokens"] = df_telegram["text_clean"].apply(
        lambda text: [
            t for t in text.split()
            if t not in ua_stopwords and len(t) > 2
        ]
    )

    df_telegram["tokens_str"] = df_telegram["tokens"].apply(lambda x: " ".join(x))

    print("--> Starting Telegram lemmatization")
    df_telegram["tokens_str"] = df_telegram["tokens_str"].apply(lemmatize_uk)
    print("--> Telegram lemmatization done")

    df_telegram["datetime"] = (
        pd.to_datetime(df_telegram["Date"])
        .dt.floor("h")
        + pd.Timedelta(hours=1)
    )

    df_telegram_hourly = (
        df_telegram
        .groupby("datetime")["tokens_str"]
        .apply(lambda x: " ".join(x))
        .reset_index()
    )

    with open(TELEGRAM_VECTORIZER_PATH, "rb") as f:
        vectorizer = pickle.load(f)


    tfidf_matrix = vectorizer.transform(df_telegram_hourly["tokens_str"])

    tfidf_df = pd.DataFrame(
        tfidf_matrix.toarray(),
        columns=[f"tfidf_{w}" for w in vectorizer.get_feature_names_out()]
    )

    tfidf_df["datetime"] = df_telegram_hourly["datetime"].values

    print("--> Telegram TF-IDF shape:", tfidf_df.shape)

    return tfidf_df

def apply_telegram_features(df, telegram_tfidf_df):
    if telegram_tfidf_df.empty:
        print("--> No Telegram TF-IDF features to apply.")
        return df

    tg_cols = [
        col for col in telegram_tfidf_df.columns
        if col.startswith("tfidf_")
    ]

    if not tg_cols:
        print("--> No tfidf_ columns in Telegram TF-IDF.")
        return df

    telegram_tfidf_df = telegram_tfidf_df.sort_values("datetime").tail(24).reset_index(drop=True)

    if len(telegram_tfidf_df) < 24:
        last_row = telegram_tfidf_df.iloc[-1:]
        while len(telegram_tfidf_df) < 24:
            telegram_tfidf_df = pd.concat([telegram_tfidf_df, last_row], ignore_index=True)

    result_blocks = []

    for region_id in df["region_id"].unique():
        region_df = df[df["region_id"] == region_id].copy().reset_index(drop=True)

        for col in tg_cols:
            region_df[col] = telegram_tfidf_df[col].values[:len(region_df)]

        result_blocks.append(region_df)

    result = pd.concat(result_blocks, axis=0, ignore_index=True)

    print("--> Telegram features applied to latest_input.csv")

    return result

def main():
    weather_data = load_weather()

    df = build_weather_features(weather_data)

    df = add_last_known_features(df)

    telegram_tfidf_df = build_telegram_features()

    df = apply_telegram_features(df, telegram_tfidf_df)
    tg_cols = [col for col in df.columns if col.startswith("tfidf_")]
    print("--> Telegram cols in final df:", len(tg_cols))
    print("--> Telegram non-zero sum:", df[tg_cols].sum().sum())



    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    print("--> latest_input.csv saved")
    print("Shape:", df.shape)
    print(df.head())


if __name__ == "__main__":
    main()
