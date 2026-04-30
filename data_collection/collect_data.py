from bs4 import BeautifulSoup
import requests
import json
import time
import csv
from datetime import datetime, timedelta, timezone
from pathlib import Path
from telethon.sync import TelegramClient
import os
from dotenv import load_dotenv

load_dotenv()


BASE_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = BASE_DIR / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)


#---------------------------------------------------ISW------------------------------------------------------------------

BASE = "https://understandingwar.org"

headers = {
    "User-Agent": "Mozilla/5.0"
}


# --------Building a link to an ISW ​​report by date------------
def build_isw_link(date_obj):
    month = date_obj.strftime("%B").lower()
    day = date_obj.day
    year = date_obj.year

    return (
        f"{BASE}/research/russia-ukraine/"
        f"russian-offensive-campaign-assessment-{month}-{day}-{year}/"
    )


def collect_isw():
    isw_file = RAW_DIR / "latest_isw.csv"

    today = datetime.now()
    links = []

    #-----generating links for the last 4 days----
    for i in range(4):
        date_obj = today - timedelta(days=i)
        link = build_isw_link(date_obj)
        links.append(link)

    print(f"--> Generated {len(links)} possible ISW links")

    collected_rows = []

    for link in links:
        print("Now looking on --> ", link)

        try:
            req = requests.get(link, headers=headers, timeout=20)
        except Exception as e:
            print("Request failed:", e)
            continue

        print("Status:", req.status_code)

        if req.status_code != 200:
            continue

        if "Cloudflare" in req.text or "Attention Required" in req.text:
            print("--> Blocked by Cloudflare")
            continue

        soup = BeautifulSoup(req.text, "html.parser")

        date_tag = soup.find("h6", class_="gb-text")
        date = date_tag.text.strip() if date_tag else ""

        title_tag = soup.find("h1")
        title = title_tag.get_text(strip=True) if title_tag else ""

        content = soup.find("div", class_="dynamic-entry-content")

        if content:
            for tag in content(["script", "style", "img", "figure", "noscript"]):
                tag.extract()

            text = content.get_text(separator=" ", strip=True)
        else:
            text = ""

        if not text:
            continue

        collected_rows.append([link, date, title, text])
        print("--> Saved in memory")

        break

        #---fallback protection---
    if not collected_rows:
        print("--> ISW blocked or no report found. Previous latest_isw.csv was not changed.")
        return

    with open(isw_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["url", "date", "title", "text"])
        writer.writerows(collected_rows)

    print(f"--> ISW file updated --> {isw_file}")


#---------------------------------------------------Weather------------------------------------------------------------------

API_KEY = os.getenv("WEATHER_API_KEY")

REGION_CITY_MAP = {
    "6": "Zhytomyr",
    "11": "Kropyvnytskyi",
    "4": "Dnipro",
    "18": "Sumy",
    "9": "Ivano-Frankivsk",
    "17": "Rivne",
    "24": "Chernivtsi",
    "3": "Lutsk",
    "15": "Odesa",
    "14": "Mykolaiv",
    "7": "Uzhhorod",
    "13": "Lviv",
    "5": "Donetsk",
    "19": "Ternopil",
    "21": "Kherson",
    "20": "Kharkiv",
    "2": "Vinnytsia",
    "23": "Cherkasy",
    "16": "Poltava",
    "8": "Zaporizhzhia",
    "10": "Kyiv",
    "22": "Khmelnytskyi",
    "25": "Chernihiv",
}


def collect_weather():
    all_weather = {}
    weather_file = RAW_DIR / "latest_weather.json"

    for region_id, city in REGION_CITY_MAP.items():
        print(f"--> Collecting weather for {city}")

        url = (
            f"https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/"
            f"timeline/{city}/next24hours"
            f"?unitGroup=metric&key={API_KEY}&contentType=json"
        )

        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            data = response.json()

            hours = data["days"][0]["hours"] + data["days"][1]["hours"]
            hourly_forecast = hours[:24]

            if hourly_forecast:
                all_weather[region_id] = hourly_forecast

        except Exception as e:
            print(f"--> Error for {city}: {e}")


        #---fallback protection---
    if not all_weather:
        print("--> Weather collection failed for all regions.")
        print("--> Previous latest_weather.json was not changed.")
        return

    with open(weather_file, "w", encoding="utf-8") as f:
        json.dump(all_weather, f, ensure_ascii=False, indent=2)

    print("--> Weather file ready and saved\n")



#---------------------------------------------------Telegram------------------------------------------------------------------

api_id = os.getenv("TG_API_ID")
api_hash = os.getenv("TG_API_HASH")

session_name = str(Path(__file__).resolve().parent / "my_session")
channel_username = "kpszsu"


def collect_telegram():
    telegram_file = RAW_DIR / "latest_telegram.csv"

    now_utc = datetime.now(timezone.utc)
    yesterday_utc = (now_utc - timedelta(days=1)).replace(
        hour=0,
        minute=0,
        second=0,
        microsecond=0
    )

    collected_rows = []

    try:
        with TelegramClient(session_name, api_id, api_hash) as client:
            print(f"--> Сollecting the latest posts from @{channel_username}")
            print(
                "--> Data collection period:",
                yesterday_utc.strftime("%Y-%m-%d %H:%M"),
                "-",
                now_utc.strftime("%Y-%m-%d %H:%M"),
                "(UTC)"
            )

            for msg in client.iter_messages(channel_username, offset_date=now_utc):
                if msg.date < yesterday_utc:
                    break

                if msg.text:
                    collected_rows.append([msg.date.isoformat(), msg.text])

    except Exception as e:
        print(f"--> Telegram collection failed: {e}")
        print("--> Previous latest_telegram.csv was not changed.")
        return


        #---fallback protection---
    if not collected_rows:
        print("--> No Telegram messages collected.")
        print("--> Previous latest_telegram.csv was not changed.")
        return

    with open(telegram_file, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Date", "Text"])
        writer.writerows(collected_rows)

    print(f"--> Telegram messages collected: {len(collected_rows)}")
    print(f"--> Telegram file updated --> {telegram_file}")


if __name__ == "__main__":
    collect_isw()
    collect_weather()
    collect_telegram()
