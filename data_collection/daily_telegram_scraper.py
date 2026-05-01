from telethon.sync import TelegramClient
import csv
import os
from datetime import datetime, timedelta, timezone

api_id = 'api_id'
api_hash = 'api_hash'

session_name = 'my_session'
channel_username = 'kpszsu' 

if not os.path.exists('daily_data'):
    os.makedirs('daily_data')


now_utc = datetime.now(timezone.utc)
yesterday_utc = (now_utc - timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)


file_path = 'daily_data/telegram_daily.csv'

with TelegramClient(session_name, api_id, api_hash) as client:
    print(f"Сollecting the latest posts from the channel @{channel_username}...")
    print(f"Data collection period: {yesterday_utc.strftime('%Y-%m-%d %H:%M')} - {now_utc.strftime('%Y-%m-%d %H:%M')} (UTC)")
    
    with open(file_path, 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Date', 'Text']) 
        
        messages_count = 0
        
        
        for msg in client.iter_messages(channel_username, offset_date=now_utc):
            if msg.date < yesterday_utc:
                break 
                
            if msg.text: 
                writer.writerow([msg.date, msg.text])
                messages_count += 1

print(f"A total of {messages_count} messages were collected today and yesterday")
print(f"The data has been successfully saved: {file_path}")
