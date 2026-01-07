#!/usr/bin/env python3
import os
import sys
from typing import Any

import requests
from dotenv import load_dotenv


def main() -> None:
    load_dotenv()
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not token:
        raise SystemExit("Встановіть TELEGRAM_BOT_TOKEN.")

    response = requests.get(f"https://api.telegram.org/bot{token}/getUpdates", timeout=30)
    response.raise_for_status()
    data: dict[str, Any] = response.json()

    results = data.get("result", [])
    if not results:
        print("Немає оновлень. Напишіть повідомлення у канал та повторіть.")
        return

    seen = set()
    for item in results:
        message = item.get("message") or item.get("channel_post")
        if not message:
            continue
        chat = message.get("chat", {})
        chat_id = chat.get("id")
        title = chat.get("title") or chat.get("username") or "unknown"
        if chat_id is None or chat_id in seen:
            continue
        seen.add(chat_id)
        print(f"{chat_id}\t{title}")


if __name__ == "__main__":
    try:
        main()
    except requests.RequestException as exc:
        print(f"Помилка запиту: {exc}", file=sys.stderr)
        sys.exit(1)
