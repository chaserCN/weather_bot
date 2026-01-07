# Weather Report Prototype

Quick prototype to fetch weather forecasts for Chernihiv or Dnipro, render an hourly chart, and save a text report plus a weekly list.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python weather_report.py --city chernihiv --day today
python weather_report.py --city dnipro --day tomorrow --out-dir outputs
python weather_report.py --city chernihiv --date 2026-01-10
```

Outputs are saved under `outputs/`.

## Telegram bot

Бот читає конфіг із `BOT_CONFIG_JSON` (JSON-рядок у змінній середовища) та токен з `TELEGRAM_BOT_TOKEN`.

Приклад конфігу:
```json
{
  "default_city": "dnipro",
  "default_time_today": "09:40",
  "default_time_tomorrow": "21:00",
  "ventusky": {
    "url": "https://www.ventusky.com/uk#p=48.8;31.4;5",
    "state_path": "outputs/ventusky_state.json",
    "width": 720,
    "height": 720,
    "wait_ms": 6000
  },
  "channels": {
    "-1001234567890": {
      "city": "chernihiv",
      "time_today": "09:40",
      "time_tomorrow": "21:00"
    }
  }
}
```

Запуск:
```bash
python3 bot.py
```

Команди:
```
/settings
/forecast24
/ventusky
```

## Railway

Потрібні змінні середовища:
- `TELEGRAM_BOT_TOKEN`
- `BOT_CONFIG_JSON` (JSON в один рядок)
- `VENTUSKY_STATE_B64` (base64 від `outputs/ventusky_state.json`)

Для Playwright потрібен Chromium. У цьому репозиторії є `nixpacks.toml`, який виконує:
```
python3 -m playwright install chromium
```
