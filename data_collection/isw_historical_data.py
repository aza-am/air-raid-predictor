from bs4 import BeautifulSoup
import requests
import time
import csv
from datetime import datetime

BASE = "https://understandingwar.org"

headers = {
    "User-Agent": "Mozilla/5.0"
}

START_DATE = datetime(2022, 2, 24)
END_DATE = datetime(2026, 3, 16)

#--------------------1)Collecting links from the website----------------------------------------------------------

links = []

for page in range(1, 65):

    url = f"{BASE}/research/?_teams=russia-ukraine&_paged={page}"
    print("Scanning page --> ", page)

    req = requests.get(url, headers=headers)
    soup = BeautifulSoup(req.text, "html.parser")

    articles = soup.select("h3.research-card-title a")
                            
    if not articles:
        break

    for a in articles:
        links.append(a["href"])

    time.sleep(0.5)
          

print("Total links:", len(links))


#-------------------------2)Collecting data from each link-------------------------------------------------------

with open("isw_data.csv", "w", newline="", encoding="utf-8") as f:

    writer = csv.writer(f)
    writer.writerow(["url", "date", "title", "text"])

    for link in links:

        print("Now looking on --> ", link)

        req = requests.get(link, headers=headers)
        if req.status_code != 200:
            print("Failed:", link)
            continue

        soup = BeautifulSoup(req.text, "html.parser")

        date_tag = soup.find("h6", class_="gb-text")
        date = date_tag.text.strip() if date_tag else ""

        try:
            date_obj = datetime.strptime(date, "%B %d, %Y")
        except:
            continue

        if not (START_DATE <= date_obj <= END_DATE):
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

        print("--> Saved")

        time.sleep(0.01)

print("--> CSV file ready and saved")
