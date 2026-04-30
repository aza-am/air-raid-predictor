import requests
import json
import csv

API_KEY = "YOUR_API_KEY"
CITY = "Kyiv"

url = f"https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/{CITY}/next24hours?unitGroup=metric&key={API_KEY}&contentType=json"

response = requests.get(url)

data = response.json()

# with open("weather_forecast.json", "w") as f:
#     json.dump(data, f, indent=4)

# print("Weather forecast saved.")

# ця частина дає загально інформацію, а код нижче - погодинно, і більш зручно


hours = data["days"][0]["hours"] + data["days"][1]["hours"]
hourly_forecast = hours[:24]

with open("weather_forecast_24h.json", "w") as f:
    json.dump(hourly_forecast, f, indent=4)

with open("weather_forecast_next24h.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    headers = hourly_forecast[0].keys()
    writer.writerow(headers)
    for hour in hourly_forecast:
        writer.writerow(hour.values())

print("24-hour forecast saved!")
