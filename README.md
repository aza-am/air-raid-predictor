Система аналізує активність зі звітів ISW, погодні умови та історичні патерни тривог, щоб передбачати ймовірність повітряної тривоги в 23 регіонах України на наступні 24 години.

### Детальний опис

**data_collection/** зібрання даних історичних та їхнє оновлення: запити до Visual Crossing API для отримання погоди, звіти ISW через BeautifulSoup та витягує останні пости з Telegram через Telethon.

**preprocessing/** містить ноутбуки з очищення та підготовки даних і скрипт для генерації ознак.

- "Preprocessing_weather_alarms_viz.ipynb" — очищення погодних даних та даних про тривоги: видалення аномалій, заповнення пропусків, додавання часових ознак (hour, month, is_weekend, is_night).

- "isw_data_preprocessing.ipynb" — обробка текстів ISW: очищення HTML, токенізація, лематизація, векторизація через TF-IDF (100 ознак + біграми). Кожен щоденний звіт розгортається на 24 годинних рядки.

- "Telegram_preprocessing_viz.ipynb" — обробка повідомлень з каналу новин: видалення URL та пунктуації, токенізація через Stanza, агрегація за годинами, TF-IDF векторизація.

- "Merge_weather_alarms.ipynb" та "merging_data.ipynb" — об'єднання всіх чотирьох датасетів (погода, тривоги, ISW, Telegram) через послідовні LEFT JOIN по регіону та часу у фінальну матрицю (808 940 × 231).

- "build_features.py" — продакшн-скрипт для підготовки вхідних даних перед кожним прогнозом.

**run_pipeline.py** — скрипт, що координує весь pipeline від збору даних до збереження прогнозу для сайту, що працює на AWS. По черзі запускає збір даних, передобробку та фінальний прогноз, зберігаючи результат у JSON.

**models/6_xgboost__v2.pkl** містить навчену модель XGBoost з підібраними регіональними порогами класифікації. **models/regions_map_final.json** зіставляє ID регіонів з їхніми назвами.

**app.py** створює Flask-сервер з API-ендпоінтом "/content/api/v1/forecast/get", який фронтенд використовує для отримання свіжих даних прогнозу.
 "/content/api/v1/forecast/update" - eндпоінт для ручного оновлення прогнозу тривог, запускає run_pipeline.py.

**templates/test.html** — дашборд із графіками ймовірностей на Chart.js та динамічним списком регіонів. 
**static/style.css** описує зовнішній вигляд: картки прогнозів та статус-бейджі.

**requirements.txt** фіксує версії залежностей, зокрема "pandas==3.0.1" та "xgboost==3.2.0", що критично для відтворюваності моделі. **.env.example** показує які ключі потрібні для запуску без розкриття справжніх значень. **.gitignore** захищає файл ".env" та "*.session" файли Telegram від потрапляння в репозиторій.


## Швидкий запуск

Створіть віртуальне середовище та встановіть залежності:

```bash
python -m venv venv
source venv\Scripts\activate      
pip install -r requirements.txt
```


Створіть файл .env на основі представленого нами .env.example:


```env
TG_API_ID=your_telegram_api_id
TG_API_HASH=your_telegram_api_hash
WEATHER_API_KEY=your_visual_crossing_key
```


Запустіть систему:

```bash
python run_pipeline.py
python app.py
```


Відкрийте [http://127.0.0.1:8000](http://127.0.0.1:8000) у браузері.


## Стек технологій

Backend — Python, Pandas, NumPy, Scikit-learn, XGBoost, Flask, [BeautifulSoup4](https://www.crummy.com/software/BeautifulSoup), [Telethon](https://docs.telethon.dev).
Frontend — HTML5, CSS3, JavaScript, [Chart.js](https://www.chartjs.org).
