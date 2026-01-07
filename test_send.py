#!/usr/bin/env python3
import argparse
import asyncio
from datetime import datetime, timedelta
from pathlib import Path

from dotenv import load_dotenv
from telegram import Bot, InputFile

import bot as bot_module
import ventusky_capture
import weather_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Разове тестове відправлення прогнозу у Telegram."
    )
    parser.add_argument("--chat-id", required=True, type=int, help="ID чату/каналу.")
    parser.add_argument(
        "--mode",
        choices=["today", "tomorrow"],
        default="today",
        help="Який прогноз надіслати.",
    )
    parser.add_argument(
        "--include-ventusky",
        action="store_true",
        help="Надіслати додатково квадратну карту Ventusky.",
    )
    return parser.parse_args()


async def main() -> None:
    load_dotenv()
    args = parse_args()

    channels_cfg, ventusky_cfg = bot_module.load_config()
    channel = bot_module.resolve_channel(channels_cfg, args.chat_id)
    if not channel:
        raise SystemExit("Цей chat_id відсутній у BOT_CONFIG_JSON.")

    token = bot_module.os.environ.get("TELEGRAM_BOT_TOKEN")
    if not token:
        raise SystemExit("Встановіть TELEGRAM_BOT_TOKEN.")

    bot = Bot(token=token)
    out_dir = Path("outputs/bot")

    target_date = datetime.now(bot_module.KYIV_TZ).date()
    if args.mode == "tomorrow":
        target_date += timedelta(days=1)

    _, chart_path, report_text = weather_report.generate_daily_forecast(
        channel.city, target_date, out_dir
    )
    with chart_path.open("rb") as photo_file:
        await bot.send_photo(chat_id=args.chat_id, photo=InputFile(photo_file), caption=report_text)

    if args.include_ventusky:
        filename = f"ventusky_test_{datetime.now(bot_module.KYIV_TZ).strftime('%Y%m%d_%H%M')}.png"
        output_path = out_dir / filename
        await ventusky_capture.capture(
            url=ventusky_cfg.url,
            output_path=output_path,
            width=ventusky_cfg.width,
            height=ventusky_cfg.height,
            wait_ms=ventusky_cfg.wait_ms,
            clean=True,
            clip_map=True,
            debug=False,
            storage_state=ventusky_cfg.state_path,
            save_state=None,
            headed=False,
        )
        with output_path.open("rb") as photo_file:
            await bot.send_photo(chat_id=args.chat_id, photo=InputFile(photo_file))


if __name__ == "__main__":
    asyncio.run(main())
