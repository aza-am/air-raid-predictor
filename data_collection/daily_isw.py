from bs4 import BeautifulSoup
import requests
import time
import csv
from datetime import datetime, timedelta


BASE = "https://understandingwar.org"

headers = {"User-Agent": "Mozilla/5.0"}


today = datetime.now()
yesterday = today - timedelta(days=1)

today_str = today.strftime("%B %d, %Y").replace(" 0", " ")   #exact format of the dates on the website
yesterday_str = yesterday.strftime("%B %d, %Y").replace(" 0", " ")


file = f"war_reports_{today_str}.csv"


url = f"{BASE}/research/?_teams=russia-ukraine&_paged=1"

req = requests.get(url, headers=headers)
soup = BeautifulSoup(req.text, "html.parser")

articles = soup.select("h3.research-card-title a")
links = [a["href"] for a in articles]

print(f"--> {len(links)} links on this page")



with open(file, "w", newline="", encoding="utf-8") as f:
    

    writer = csv.writer(f)
    writer.writerow(["url", "date", "title", "text"])


    for link in links:
        print("Now looking on --> ", link)

        req = requests.get(link, headers=headers)
        soup = BeautifulSoup(req.text, "html.parser")

        date_tag = soup.find("h6", class_="gb-text")
        date = date_tag.text.strip() if date_tag else ""


        if date != today_str and date != yesterday_str:
            continue

        title = soup.find("h1").get_text(strip=True)   


        content = soup.find("div", class_="dynamic-entry-content")
        if content:
            
            for tag in content(["script", "style", "img", "figure", "noscript"]):
                tag.extract()

            text = content.get_text(separator=" ", strip=True)
        else:
            text = ""


        writer.writerow([link, date, title, text])
        f.flush()
        time.sleep(1)


    print(f"--> CSV file ready and saved --> {file}")
