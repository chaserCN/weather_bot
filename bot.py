#!/usr/bin/env python3
import base64
import json
import os
from dataclasses import dataclass
from datetime import datetime, time as dt_time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from dotenv import load_dotenv
from telegram import BotCommand, BotCommandScopeAllGroupChats, InputFile, Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes

import ventusky_capture
import weather_report


KYIV_TZ = ZoneInfo("Europe/Kyiv")
DEFAULT_CITY = "dnipro"
DEFAULT_TIME_TODAY = "09:40"
DEFAULT_TIME_TOMORROW = "21:00"


@dataclass
class VentuskyConfig:
    url: str
    state_path: str
    width: int = 720
    height: int = 720
    wait_ms: int = 6000


@dataclass
class ChannelConfig:
    chat_id: int
    city: str
    time_today: str
    time_tomorrow: str


def parse_time(value: str) -> dt_time:
    hour_str, minute_str = value.split(":")
    return dt_time(hour=int(hour_str), minute=int(minute_str), tzinfo=KYIV_TZ)


def load_config() -> tuple[list[ChannelConfig], VentuskyConfig]:
    raw = os.environ.get("BOT_CONFIG_JSON")
    if not raw:
        raise SystemExit("Встановіть BOT_CONFIG_JSON з конфігурацією бота.")
    data = json.loads(raw)

    default_city = data.get("default_city", DEFAULT_CITY)
    default_time_today = data.get("default_time_today", DEFAULT_TIME_TODAY)
    default_time_tomorrow = data.get("default_time_tomorrow", DEFAULT_TIME_TOMORROW)

    ventusky_data = data.get("ventusky", {})
    ventusky_url = ventusky_data.get("url", "https://www.ventusky.com/uk#p=48.8;31.4;5")
    ventusky_state = ventusky_data.get("state_path", "outputs/ventusky_state.json")
    ventusky_cfg = VentuskyConfig(
        url=ventusky_url,
        state_path=ventusky_state,
        width=int(ventusky_data.get("width", 720)),
        height=int(ventusky_data.get("height", 720)),
        wait_ms=int(ventusky_data.get("wait_ms", 6000)),
    )
    state_b64 = os.environ.get("VENTUSKY_STATE_B64")
    if state_b64:
        decoded = base64.b64decode(state_b64.encode("utf-8"))
        state_path = Path("outputs/ventusky_state_runtime.json")
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_bytes(decoded)
        ventusky_cfg.state_path = str(state_path)

    channels_cfg = []
    channels = data.get("channels", {})
    if isinstance(channels, dict):
        items = [{"id": key, **value} for key, value in channels.items()]
    else:
        items = channels

    for entry in items:
        if not entry:
            continue
        chat_id = int(entry["id"]) if "id" in entry else int(entry["chat_id"])
        city = normalize_city(entry.get("city", default_city))
        if city not in weather_report.CITY_COORDS:
            raise SystemExit(f"Невідоме місто у конфігу: {city}")
        time_today = entry.get("time_today", default_time_today)
        time_tomorrow = entry.get("time_tomorrow", default_time_tomorrow)
        channels_cfg.append(
            ChannelConfig(
                chat_id=chat_id,
                city=city,
                time_today=time_today,
                time_tomorrow=time_tomorrow,
            )
        )
    if not channels_cfg:
        raise SystemExit("BOT_CONFIG_JSON має містити хоча б один канал у 'channels'.")
    return channels_cfg, ventusky_cfg


def normalize_city(value: str) -> str:
    if not value:
        return DEFAULT_CITY
    cleaned = value.strip().lower()
    mapping = {
        "дніпро": "dnipro",
        "днепр": "dnipro",
        "dnipro": "dnipro",
        "чернігів": "chernihiv",
        "чернигов": "chernihiv",
        "chernihiv": "chernihiv",
    }
    return mapping.get(cleaned, cleaned)


def resolve_channel(channels_cfg: list[ChannelConfig], chat_id: int) -> ChannelConfig | None:
    for channel in channels_cfg:
        if channel.chat_id == chat_id:
            return channel
    return None


async def send_daily_forecast(
    chat_id: int,
    city: str,
    target_date: datetime.date,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:
    out_dir = Path("outputs/bot")
    _, chart_path, report_text = weather_report.generate_daily_forecast(
        city, target_date, out_dir
    )
    with chart_path.open("rb") as photo_file:
        await context.bot.send_photo(
            chat_id=chat_id,
            photo=InputFile(photo_file),
            caption=report_text,
        )


async def send_ventusky_square(
    chat_id: int,
    ventusky_cfg: VentuskyConfig,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:
    out_dir = Path("outputs/bot")
    out_dir.mkdir(parents=True, exist_ok=True)
    filename = f"ventusky_{datetime.now(KYIV_TZ).strftime('%Y%m%d_%H%M')}.png"
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
        await context.bot.send_photo(chat_id=chat_id, photo=InputFile(photo_file))


async def job_tomorrow(context: ContextTypes.DEFAULT_TYPE) -> None:
    channel: ChannelConfig = context.job.data["channel"]
    tomorrow = datetime.now(KYIV_TZ).date() + timedelta(days=1)
    await send_daily_forecast(channel.chat_id, channel.city, tomorrow, context)


async def job_today(context: ContextTypes.DEFAULT_TYPE) -> None:
    channel: ChannelConfig = context.job.data["channel"]
    ventusky_cfg: VentuskyConfig = context.job.data["ventusky"]
    today = datetime.now(KYIV_TZ).date()
    await send_daily_forecast(channel.chat_id, channel.city, today, context)
    await send_ventusky_square(channel.chat_id, ventusky_cfg, context)


async def forecast_24h(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    channels_cfg: list[ChannelConfig] = context.bot_data["channels"]
    channel = resolve_channel(channels_cfg, update.effective_chat.id)
    if not channel:
        await update.message.reply_text("Для цього каналу немає налаштувань.")
        return
    out_dir = Path("outputs/bot")
    today = datetime.now(KYIV_TZ).date()
    _, _, report_text = weather_report.generate_daily_forecast(
        channel.city, today, out_dir
    )
    _, chart_path, _ = weather_report.generate_next24h_forecast(
        channel.city, out_dir
    )
    with chart_path.open("rb") as photo_file:
        await update.message.reply_photo(
            photo=InputFile(photo_file),
            caption=report_text,
        )


async def settings_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    channels_cfg: list[ChannelConfig] = context.bot_data["channels"]
    channel = resolve_channel(channels_cfg, update.effective_chat.id)
    if not channel:
        await update.message.reply_text("Для цього каналу немає налаштувань.")
        return
    text = (
        "Налаштування каналу:\n"
        f"Місто: {channel.city}\n"
        f"Час прогнозу на сьогодні: {channel.time_today}\n"
        f"Час прогнозу на завтра: {channel.time_tomorrow}"
    )
    await update.message.reply_text(text)


async def ventusky_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    ventusky_cfg: VentuskyConfig = context.bot_data["ventusky"]
    await send_ventusky_square(update.effective_chat.id, ventusky_cfg, context)


def schedule_jobs(app, channels_cfg: list[ChannelConfig], ventusky_cfg: VentuskyConfig) -> None:
    for channel in channels_cfg:
        time_today = parse_time(channel.time_today)
        time_tomorrow = parse_time(channel.time_tomorrow)

        app.job_queue.run_daily(
            job_today,
            time=time_today,
            name=f"today_{channel.chat_id}",
            data={"channel": channel, "ventusky": ventusky_cfg},
        )
        app.job_queue.run_daily(
            job_tomorrow,
            time=time_tomorrow,
            name=f"tomorrow_{channel.chat_id}",
            data={"channel": channel},
        )


async def post_init(app) -> None:
    commands = [
        BotCommand("settings", "показати налаштування каналу"),
        BotCommand("forecast24", "прогноз на 24 години"),
        BotCommand("ventusky", "карта Ventusky"),
    ]
    await app.bot.set_my_commands(commands)
    await app.bot.set_my_commands(commands, scope=BotCommandScopeAllGroupChats())


def main() -> None:
    load_dotenv()
    shutil.rmtree("outputs/bot", ignore_errors=True)
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not token:
        raise SystemExit("Встановіть TELEGRAM_BOT_TOKEN.")

    channels_cfg, ventusky_cfg = load_config()

    app = ApplicationBuilder().token(token).post_init(post_init).build()
    app.bot_data["channels"] = channels_cfg
    app.bot_data["ventusky"] = ventusky_cfg
    app.add_handler(CommandHandler("forecast24", forecast_24h))
    app.add_handler(CommandHandler("settings", settings_cmd))
    app.add_handler(CommandHandler("ventusky", ventusky_cmd))

    schedule_jobs(app, channels_cfg, ventusky_cfg)
    app.run_polling()


if __name__ == "__main__":
    main()
import shutil
