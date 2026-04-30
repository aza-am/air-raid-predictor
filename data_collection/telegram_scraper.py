from telethon.sync import TelegramClient
import csv
import os
from datetime import datetime, timezone

api_id = 'api_id'
api_hash = 'api_hash'

session_name = 'my_session'
channel_username = 'kpszsu' 

if not os.path.exists('daily_data'):
    os.makedirs('daily_data')

start_date = datetime(2022, 2, 24, tzinfo=timezone.utc)
end_date = datetime(2026, 3, 17, tzinfo=timezone.utc)

with TelegramClient(session_name, api_id, api_hash) as client:
    print(f"Downloading messages from the channel @{channel_username}...")
    print(f"Period: February 24, 2022, through March 16, 2026.")
    
    file_path = 'daily_data/telegram_news.csv'
    with open(file_path, 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Date', 'Text']) 
        
    
        for msg in client.iter_messages(channel_username, offset_date=end_date):
            if msg.date < start_date:
                break
                
            
            if msg.text: 
                writer.writerow([msg.date, msg.text])

print(f"The data has been successfully saved: {file_path}")
