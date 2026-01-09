#!/usr/bin/env python3
import argparse
import math
from datetime import date, datetime, timedelta
from datetime import time as dt_time
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FixedLocator, FuncFormatter
from matplotlib.transforms import Bbox
import requests
import re


CITY_COORDS = {
    "chernihiv": {"name": "Чернігів", "lat": 51.4982, "lon": 31.2893},
    "dnipro": {"name": "Дніпро", "lat": 48.4647, "lon": 35.0462},
}

KYIV_TZ = ZoneInfo("Europe/Kyiv")
API_URL = "https://api.open-meteo.com/v1/forecast"
GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
UA_WEEKDAYS = [
    "понеділок",
    "вівторок",
    "середа",
    "четвер",
    "пʼятниця",
    "субота",
    "неділя",
]
UA_WEEKDAYS_SHORT = [
    "Пон",
    "Вів",
    "Сер",
    "Чет",
    "Пʼят",
    "Суб",
    "Нед",
]
UA_MONTHS_GEN = [
    "січня",
    "лютого",
    "березня",
    "квітня",
    "травня",
    "червня",
    "липня",
    "серпня",
    "вересня",
    "жовтня",
    "листопада",
    "грудня",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Отримати та візуалізувати прогноз погоди для вибраних міст України."
    )
    parser.add_argument(
        "--city",
        required=True,
        choices=sorted(CITY_COORDS.keys()),
        help="Місто: chernihiv або dnipro.",
    )
    day_group = parser.add_mutually_exclusive_group(required=True)
    day_group.add_argument(
        "--day",
        choices=["today", "tomorrow"],
        help="День прогнозу: today або tomorrow.",
    )
    day_group.add_argument(
        "--date",
        help="Дата прогнозу у форматі YYYY-MM-DD.",
    )
    parser.add_argument(
        "--out-dir",
        default="outputs",
        help="Каталог для збереження текстового звіту та графіків.",
    )
    parser.add_argument(
        "--debug-labels",
        action="store_true",
        help="Друкувати детальні логи розміщення підписів.",
    )
    return parser.parse_args()


def fetch_forecast(lat: float, lon: float, forecast_days: int) -> dict:
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "temperature_2m,apparent_temperature,precipitation_probability",
        "daily": "temperature_2m_max,temperature_2m_min,precipitation_probability_max",
        "forecast_days": forecast_days,
        "timezone": "Europe/Kyiv",
        "current_weather": True,
    }
    response = requests.get(API_URL, params=params, timeout=30)
    response.raise_for_status()
    return response.json()


def normalize_place_name(text: str) -> str:
    return re.sub(r"[^a-zа-яіїєґ0-9]+", "", text.lower())


def geocode_city(name: str):
    params_base = {
        "name": name,
        "count": 10,
        "format": "json",
    }
    query_norm = normalize_place_name(name)
    best = None
    best_score = (-1, -1)
    for language in ("uk", "ru", "en"):
        params = {**params_base, "language": language}
        response = requests.get(GEOCODING_URL, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        results = data.get("results") or []
        for result in results:
            name_norm = normalize_place_name(result.get("name", ""))
            if not name_norm:
                continue
            if name_norm == query_norm:
                score = 3
            elif name_norm.startswith(query_norm) or query_norm.startswith(name_norm):
                score = 2
            elif query_norm in name_norm:
                score = 1
            else:
                score = 0
            population = result.get("population") or 0
            score_tuple = (score, population)
            if score_tuple > best_score:
                best_score = score_tuple
                best = result
        if best_score[0] >= 3:
            break
    return best


def make_location_slug(lat: float, lon: float) -> str:
    safe = f"loc_{lat:.2f}_{lon:.2f}"
    return re.sub(r"[^a-zA-Z0-9_.-]", "_", safe)


def select_target_date(day: str | None, date_str: str | None) -> date:
    today = datetime.now(KYIV_TZ).date()
    if date_str:
        try:
            parsed = date.fromisoformat(date_str)
        except ValueError as exc:
            raise SystemExit("Невірний формат дати. Використовуйте YYYY-MM-DD.") from exc
        return parsed
    if day == "today":
        return today
    if day == "tomorrow":
        return today + timedelta(days=1)
    raise SystemExit("Вкажіть --day або --date.")


def build_base_name(city_key: str, day: str | None, date_str: str | None) -> str:
    if date_str:
        return f"{city_key}_{date_str}"
    return f"{city_key}_{day}"




def extract_hourly_range(data: dict, start_dt: datetime, end_dt: datetime):
    times = data["hourly"]["time"]
    temps = data["hourly"]["temperature_2m"]
    precip = data["hourly"]["precipitation_probability"]
    items = []
    for time_str, temp, prob in zip(times, temps, precip):
        dt = datetime.fromisoformat(time_str).replace(tzinfo=KYIV_TZ)
        if start_dt <= dt <= end_dt:
            items.append((dt, temp, prob))
    return items


def extract_weekly_plot_data(data: dict):
    daily_days = [datetime.fromisoformat(day).replace(tzinfo=KYIV_TZ) for day in data["daily"]["time"]]
    daily_min = data["daily"]["temperature_2m_min"]
    daily_max = data["daily"]["temperature_2m_max"]
    daily_precip = compute_daily_precip_probabilities(data)
    return daily_days, daily_min, daily_max, daily_precip


def compute_daily_precip_probabilities(data: dict) -> list[float]:
    days = data["daily"]["time"]
    hourly_times = data["hourly"]["time"]
    hourly_probs = data["hourly"]["precipitation_probability"]
    if not hourly_times or not hourly_probs:
        return data["daily"]["precipitation_probability_max"]
    today = datetime.now(KYIV_TZ).date()
    now = datetime.now(KYIV_TZ)
    result = []
    for day_str in days:
        target = datetime.fromisoformat(day_str).date()
        probs = []
        for time_str, prob in zip(hourly_times, hourly_probs):
            dt = datetime.fromisoformat(time_str).replace(tzinfo=KYIV_TZ)
            if dt.date() != target:
                continue
            if target == today and dt < now:
                continue
            probs.append(prob / 100.0)
        if not probs:
            result.append(0.0)
            continue
        p_max = max(probs)
        p_union = 1.0
        for p in probs:
            p_union *= (1 - p)
        p_union = 1 - p_union
        f = 0.2
        p_final = p_max + f * (p_union - p_max)
        result.append(p_final * 100.0)
    return result


def generate_daily_forecast(
    city_key: str,
    target_date: date,
    out_dir: Path,
    debug_labels: bool = False,
) -> tuple[Path, Path, str]:
    city = CITY_COORDS[city_key]
    today = datetime.now(KYIV_TZ).date()
    days_ahead = (target_date - today).days
    if days_ahead < 0:
        raise SystemExit("Дата прогнозу вже минула.")
    if days_ahead >= 16:
        raise SystemExit("Доступний прогноз лише на 16 днів уперед.")

    forecast_days = max(10, days_ahead + 1)
    data = fetch_forecast(city["lat"], city["lon"], forecast_days)
    hourly_items = extract_hourly(data, target_date)
    if not hourly_items:
        raise SystemExit("Немає погодинних даних для вибраної дати.")

    report_text = build_two_day_summary_from_data(data, today)

    out_dir.mkdir(parents=True, exist_ok=True)
    base_name = build_base_name(city_key, None, target_date.isoformat())
    text_path = out_dir / f"{base_name}.txt"
    chart_path = out_dir / f"{base_name}.png"

    text_path.write_text(report_text, encoding="utf-8")
    daily_days, daily_min, daily_max, daily_precip = extract_weekly_plot_data(data)
    plot_chart_base(
        [item[0] for item in hourly_items],
        [item[1] for item in hourly_items],
        [item[2] for item in hourly_items],
        format_chart_title(city["name"], format_ua_date(target_date)),
        chart_path,
        datetime.combine(target_date, dt_time(0, 0), tzinfo=KYIV_TZ),
        datetime.combine(target_date, dt_time(0, 0), tzinfo=KYIV_TZ) + timedelta(hours=23),
        debug_labels,
        weekly_days=daily_days,
        weekly_min=daily_min,
        weekly_max=daily_max,
        weekly_precip=daily_precip,
    )
    return text_path, chart_path, report_text


def generate_next24h_forecast(
    city_key: str,
    out_dir: Path,
    start_dt: datetime | None = None,
    debug_labels: bool = False,
) -> tuple[Path, Path, str]:
    city = CITY_COORDS[city_key]
    if start_dt is None:
        start_dt = datetime.now(KYIV_TZ)
    start_dt = start_dt.replace(minute=0, second=0, microsecond=0)
    end_dt = start_dt + timedelta(hours=23)

    today = datetime.now(KYIV_TZ).date()
    days_ahead = (end_dt.date() - today).days
    forecast_days = max(10, days_ahead + 1)
    data = fetch_forecast(city["lat"], city["lon"], forecast_days)
    hourly_items = extract_hourly_range(data, start_dt, end_dt)
    if not hourly_items:
        raise SystemExit("Немає погодинних даних для наступних 24 годин.")

    report_text = build_two_day_summary_from_data(data, today)

    out_dir.mkdir(parents=True, exist_ok=True)
    base_name = f"{city_key}_next24h_{start_dt.strftime('%Y-%m-%d_%H%M')}"
    text_path = out_dir / f"{base_name}.txt"
    chart_path = out_dir / f"{base_name}.png"

    text_path.write_text(report_text, encoding="utf-8")
    title = format_chart_title(city["name"], f"{start_dt.strftime('%d.%m %H:%M')} +24г")
    times = [item[0] for item in hourly_items]
    temps = [item[1] for item in hourly_items]
    precip = [item[2] for item in hourly_items]
    daily_days, daily_min, daily_max, daily_precip = extract_weekly_plot_data(data)
    plot_chart_base(
        times,
        temps,
        precip,
        title,
        chart_path,
        start_dt,
        end_dt,
        debug_labels,
        weekly_days=daily_days,
        weekly_min=daily_min,
        weekly_max=daily_max,
        weekly_precip=daily_precip,
    )
    return text_path, chart_path, report_text


def generate_next24h_forecast_for_coords(
    location_name: str,
    lat: float,
    lon: float,
    out_dir: Path,
    start_dt: datetime | None = None,
    debug_labels: bool = False,
) -> tuple[Path, Path, str]:
    if start_dt is None:
        start_dt = datetime.now(KYIV_TZ)
    start_dt = start_dt.replace(minute=0, second=0, microsecond=0)
    end_dt = start_dt + timedelta(hours=23)

    today = datetime.now(KYIV_TZ).date()
    days_ahead = (end_dt.date() - today).days
    forecast_days = max(10, days_ahead + 1)
    data = fetch_forecast(lat, lon, forecast_days)
    hourly_items = extract_hourly_range(data, start_dt, end_dt)
    if not hourly_items:
        raise SystemExit("Немає погодинних даних для наступних 24 годин.")

    report_text = build_two_day_summary_from_data(data, today)

    out_dir.mkdir(parents=True, exist_ok=True)
    base_name = f"{make_location_slug(lat, lon)}_next24h_{start_dt.strftime('%Y-%m-%d_%H%M')}"
    text_path = out_dir / f"{base_name}.txt"
    chart_path = out_dir / f"{base_name}.png"

    text_path.write_text(report_text, encoding="utf-8")
    title = format_chart_title(location_name, f"{start_dt.strftime('%d.%m %H:%M')} +24г")
    times = [item[0] for item in hourly_items]
    temps = [item[1] for item in hourly_items]
    precip = [item[2] for item in hourly_items]
    daily_days, daily_min, daily_max, daily_precip = extract_weekly_plot_data(data)
    plot_chart_base(
        times,
        temps,
        precip,
        title,
        chart_path,
        start_dt,
        end_dt,
        debug_labels,
        weekly_days=daily_days,
        weekly_min=daily_min,
        weekly_max=daily_max,
        weekly_precip=daily_precip,
    )
    return text_path, chart_path, report_text


def extract_hourly(data: dict, target_date: date):
    times = data["hourly"]["time"]
    temps = data["hourly"]["temperature_2m"]
    precip = data["hourly"]["precipitation_probability"]
    items = []
    for time_str, temp, prob in zip(times, temps, precip):
        dt = datetime.fromisoformat(time_str).replace(tzinfo=KYIV_TZ)
        if dt.date() == target_date:
            items.append((dt, temp, prob))
    return items


def build_weekly_lines(data: dict, today: date) -> list[str]:
    days = data["daily"]["time"]
    tmax = data["daily"]["temperature_2m_max"]
    tmin = data["daily"]["temperature_2m_min"]
    pmax = data["daily"]["precipitation_probability_max"]
    lines = []
    for day_str, mx, mn, pr in zip(days, tmax, tmin, pmax):
        day_dt = datetime.fromisoformat(day_str).date()
        label = format_ua_date(day_dt)
        if day_dt == today:
            label = "сьогодні"
        lines.append(f"{label}: 🌡 {mn:.0f}..{mx:.0f}°  🌧 {pr:.0f}%")
        lines.append("")
    return lines


def format_ua_date(target_date: date) -> str:
    weekday = UA_WEEKDAYS[target_date.weekday()]
    month = UA_MONTHS_GEN[target_date.month - 1]
    return f"{weekday} {target_date.day} {month}"


def format_chart_title(location: str, date_label: str) -> str:
    title = f"{location} — {date_label}"
    if len(title) > 26:
        return f"{location}\n— {date_label}"
    return title


def build_text_report(city_name: str, target_date: date, hourly_items, weekly_lines) -> str:
    date_label = format_ua_date(target_date)
    header = [
        f"Прогноз для міста {city_name}",
        f"Дата: {date_label}",
        "",
    ]
    return "\n".join(header + weekly_lines + [""])


def build_hourly_24h_lines(hourly_items) -> list[str]:
    lines = ["Прогноз на 24 години:"]
    for dt, temp, prob in hourly_items:
        lines.append(f"{dt.strftime('%d.%m %H:%M')}: {temp:.0f}°, опади {prob:.0f}%")
    return lines


def build_hourly_24h_lines_table(hourly_items) -> list[str]:
    lines = []
    for dt, temp, prob in hourly_items:
        time_col = dt.strftime("%d.%m %H:%M")
        temp_col = f"{temp:>3.0f}°"
        prec_col = f"{prob:>3.0f}%"
        lines.append(f"{time_col}  {temp_col:>4}  {prec_col:>4}")
    return lines


def build_text_report_24h(city_name: str, start_dt: datetime, hourly_items) -> str:
    header = [
        f"Прогноз для міста {city_name}",
        f"Від {start_dt.strftime('%d.%m %H:%M')} на 24 години",
        "",
    ]
    return "\n".join(header + build_hourly_24h_lines_table(hourly_items) + [""])


def build_two_day_summary_from_data(data: dict, today: date | None = None) -> str:
    if today is None:
        today = datetime.now(KYIV_TZ).date()
    tomorrow = today + timedelta(days=1)
    days = data["daily"]["time"]
    tmin = data["daily"]["temperature_2m_min"]
    tmax = data["daily"]["temperature_2m_max"]
    pmax = data["daily"]["precipitation_probability_max"]
    hourly_times = data["hourly"]["time"]
    hourly_probs = data["hourly"]["precipitation_probability"]
    now = datetime.now(KYIV_TZ)

    def find_range(target: date):
        for day_str, mn, mx, pr in zip(days, tmin, tmax, pmax):
            if datetime.fromisoformat(day_str).date() == target:
                return mn, mx, pr
        return None

    def prob_from_hours(target: date, from_now: bool) -> float | None:
        if not hourly_times or not hourly_probs:
            return None
        probs = []
        for time_str, prob in zip(hourly_times, hourly_probs):
            dt = datetime.fromisoformat(time_str).replace(tzinfo=KYIV_TZ)
            if dt.date() != target:
                continue
            if from_now and dt < now:
                continue
            probs.append(prob / 100.0)
        if not probs:
            return None
        p_max = max(probs)
        p_union = 1.0
        for p in probs:
            p_union *= (1 - p)
        p_union = 1 - p_union
        f = 0.2
        p_final = p_max + f * (p_union - p_max)
        return p_final * 100.0

    def find_current_hour_prob():
        times = data["hourly"]["time"]
        probs = data["hourly"]["precipitation_probability"]
        if not times or not probs:
            return None
        best_idx = None
        best_delta = None
        for idx, time_str in enumerate(times):
            dt = datetime.fromisoformat(time_str).replace(tzinfo=KYIV_TZ)
            delta = abs((dt - now).total_seconds())
            if best_delta is None or delta < best_delta:
                best_delta = delta
                best_idx = idx
        if best_idx is None:
            return None
        return probs[best_idx]

    def find_current_apparent():
        times = data["hourly"]["time"]
        feels = data["hourly"]["apparent_temperature"]
        if not times or not feels:
            return None
        best_idx = None
        best_delta = None
        for idx, time_str in enumerate(times):
            dt = datetime.fromisoformat(time_str).replace(tzinfo=KYIV_TZ)
            delta = abs((dt - now).total_seconds())
            if best_delta is None or delta < best_delta:
                best_delta = delta
                best_idx = idx
        if best_idx is None:
            return None
        return feels[best_idx]

    lines = []
    today_range = find_range(today)
    if today_range:
        mn, mx, pr = today_range
        pr_from_now = prob_from_hours(today, from_now=True)
        if pr_from_now is not None:
            pr = pr_from_now
        lines.append(f"сьогодні: мін {mn:.0f}º, макс {mx:.0f}º, 🌧 {pr:.0f}%")
    tomorrow_range = find_range(tomorrow)
    if tomorrow_range:
        mn, mx, pr = tomorrow_range
        pr_full = prob_from_hours(tomorrow, from_now=False)
        if pr_full is not None:
            pr = pr_full
        lines.append(f"завтра: мін {mn:.0f}º, макс {mx:.0f}º, 🌧 {pr:.0f}%")
    current = data.get("current_weather") or {}
    current_temp = current.get("temperature")
    current_feels = find_current_apparent()
    current_prob = find_current_hour_prob()
    if current_temp is not None or current_feels is not None or current_prob is not None:
        lines.append("")
        parts = []
        if current_temp is not None:
            parts.append(f"{current_temp:.0f}º")
        if current_feels is not None:
            parts.append(f"відчувається як {current_feels:.0f}º")
        if current_prob is not None:
            parts.append(f"🌧 {current_prob:.0f}%")
        lines.append(f"зараз: {', '.join(parts)}")
    return "\n".join(lines)


def build_day_summary_from_data(data: dict, target: date, label: str) -> str:
    days = data["daily"]["time"]
    tmin = data["daily"]["temperature_2m_min"]
    tmax = data["daily"]["temperature_2m_max"]
    pmax = data["daily"]["precipitation_probability_max"]
    for day_str, mn, mx, pr in zip(days, tmin, tmax, pmax):
        if datetime.fromisoformat(day_str).date() == target:
            return f"{label}: мін {mn:.0f}º, макс {mx:.0f}º, 🌧 {pr:.0f}%"
    return ""


def build_tomorrow_with_current_from_data(data: dict, today: date | None = None) -> str:
    if today is None:
        today = datetime.now(KYIV_TZ).date()
    tomorrow = today + timedelta(days=1)
    tomorrow_line = build_day_summary_from_data(data, tomorrow, "завтра")
    current_block = build_two_day_summary_from_data(data, today).split("\n\n", 1)
    current_line = ""
    if len(current_block) == 2 and current_block[1].strip():
        current_line = current_block[1].strip()
    if current_line:
        return "\n\n".join([tomorrow_line, current_line])
    return tomorrow_line


def build_two_day_summary_for_city(city_key: str, today: date | None = None) -> str:
    city = CITY_COORDS[city_key]
    data = fetch_forecast(city["lat"], city["lon"], 2)
    return build_two_day_summary_from_data(data, today)


def build_day_summary_for_city(city_key: str, target: date, label: str) -> str:
    city = CITY_COORDS[city_key]
    data = fetch_forecast(city["lat"], city["lon"], 2)
    return build_day_summary_from_data(data, target, label)


def build_tomorrow_with_current_for_city(city_key: str, today: date | None = None) -> str:
    city = CITY_COORDS[city_key]
    data = fetch_forecast(city["lat"], city["lon"], 2)
    return build_tomorrow_with_current_from_data(data, today)


def build_two_day_summary_for_coords(
    lat: float,
    lon: float,
    today: date | None = None,
) -> str:
    data = fetch_forecast(lat, lon, 2)
    return build_two_day_summary_from_data(data, today)


def set_time_axis(ax, start_dt: datetime, end_dt: datetime) -> None:
    padding = timedelta(minutes=45)
    ax.set_xlim(start_dt - padding, end_dt + padding)


def plot_chart_base(
    times,
    temps,
    precip,
    title: str,
    output_path: Path,
    start_dt: datetime,
    end_dt: datetime,
    debug_labels: bool = False,
    weekly_days=None,
    weekly_min=None,
    weekly_max=None,
    weekly_precip=None,
) -> None:
    matplotlib.use("Agg")
    plt.rcParams.update(
        {
            "font.size": 13,
            "axes.titlesize": 16,
            "axes.labelsize": 14,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
        }
    )
    grid_style = {"linestyle": "--", "alpha": 0.3}
    grid_color = matplotlib.rcParams["grid.color"]
    grid_width = matplotlib.rcParams["grid.linewidth"]

    rows = 3 if weekly_days is not None else 2
    height = 6.9 if rows == 3 else 5.9
    fig, axes = plt.subplots(
        nrows=rows,
        ncols=1,
        figsize=(4.2, height),
        sharex=False,
        gridspec_kw={"height_ratios": [1.6, 1.0, 1.0] if rows == 3 else None},
    )
    ax_temp = axes[0]
    ax_precip = axes[1]
    ax_week = axes[2] if rows == 3 else None
    if ax_week is not None:
        ax_precip.sharex(ax_temp)

    times_num = mdates.date2num(times)
    dense_num = np.linspace(times_num[0], times_num[-1], len(times_num) * 4)
    dense_temps = np.interp(dense_num, times_num, temps)
    window = 9
    kernel = np.ones(window) / window
    pad = window // 2
    padded = np.pad(dense_temps, (pad, pad), mode="edge")
    smooth_temps = np.convolve(padded, kernel, mode="valid")
    smooth_times = mdates.num2date(dense_num, tz=KYIV_TZ)

    now = datetime.now(KYIV_TZ)
    past_mask = [t <= now for t in smooth_times]
    if any(past_mask):
        past_times = [t for t, is_past in zip(smooth_times, past_mask) if is_past]
        past_temps = [v for v, is_past in zip(smooth_temps, past_mask) if is_past]
        ax_temp.plot(past_times, past_temps, color="#dddddd", linewidth=2.5)
    if not all(past_mask):
        future_times = [t for t, is_past in zip(smooth_times, past_mask) if not is_past]
        future_temps = [v for v, is_past in zip(smooth_temps, past_mask) if not is_past]
        ax_temp.plot(future_times, future_temps, color="#1f77b4", linewidth=2.5)
    temp_min = min(temps)
    temp_max = max(temps)
    temp_range = max(temp_max - temp_min, 1.0)
    ax_temp.set_ylim(temp_min - 0.45 * temp_range, temp_max + 0.45 * temp_range)

    labeled_indices = [
        idx
        for idx, time in enumerate(times)
        if time.minute == 0 and time.hour in {0, 6, 12, 18}
    ]
    marker_times = [times[idx] for idx in labeled_indices]
    marker_temps = [temps[idx] for idx in labeled_indices]
    marker_colors = ["#dddddd" if t <= now else "#1f77b4" for t in marker_times]
    ax_temp.scatter(marker_times, marker_temps, color=marker_colors, s=60, zorder=3)

    min_index = temps.index(temp_min)
    max_index = temps.index(temp_max)
    annotate_indices = set(labeled_indices)

    bar_colors = ["#dddddd" if t <= now else "#ff7f0e" for t in times]
    ax_precip.bar(times, precip, color=bar_colors, alpha=0.6, width=0.03)
    ax_precip.set_ylim(0, 100)
    ax_precip.set_axisbelow(True)
    ax_precip.set_yticks([0, 25, 50, 75, 100])
    ax_precip.set_yticklabels(["0%", "25%", "50%", "75%", ""])
    ax_precip.tick_params(axis="y")
    for level in (25, 75):
        ax_precip.axhline(
            level,
            color=grid_color,
            linestyle=(0, (3, 3)),
            linewidth=grid_width,
            alpha=grid_style["alpha"],
        )

    ax_temp.set_ylabel("")
    ax_precip.set_ylabel("")
    if ax_week is not None:
        ax_week.set_ylabel("")

    set_time_axis(ax_temp, start_dt, end_dt)
    tick_times = []
    tick = start_dt.replace(minute=0, second=0, microsecond=0)
    while tick <= end_dt:
        tick_times.append(tick)
        tick += timedelta(hours=3)
    hour_ticks = FixedLocator(mdates.date2num(tick_times))
    ax_temp.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=KYIV_TZ))
    ax_temp.xaxis.set_major_locator(hour_ticks)
    ax_temp.tick_params(axis="x", labelrotation=45, labelbottom=False)

    ax_precip.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=KYIV_TZ))
    ax_precip.xaxis.set_major_locator(hour_ticks)
    plt.setp(ax_precip.get_xticklabels(), rotation=45, ha="right")
    plt.setp(ax_temp.get_xticklabels(), rotation=45, ha="right")
    if ax_week is not None:
        ax_week.tick_params(axis="x", labelrotation=45)
        plt.setp(ax_week.get_xticklabels(), ha="right")

    def format_temp_value(value: float, decimals: int) -> str:
        if decimals == 0:
            truncated = math.trunc(value)
            return f"{truncated:.0f}°"
        return f"{value:.1f}°"

    y_ticks = ax_temp.get_yticks()
    has_fractional_ticks = any(abs(tick - round(tick)) > 1e-6 for tick in y_ticks)
    decimals = 1 if has_fractional_ticks else 0
    ax_temp.yaxis.set_major_formatter(
        FuncFormatter(lambda v, pos: format_temp_value(v, decimals))
    )
    if ax_week is not None:
        week_ticks = ax_week.get_yticks()
        week_has_fractional = any(abs(tick - round(tick)) > 1e-6 for tick in week_ticks)
        week_decimals = 1 if week_has_fractional else 0
        ax_week.yaxis.set_major_formatter(
            FuncFormatter(lambda v, pos: format_temp_value(v, week_decimals))
        )

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    axes_bbox = ax_temp.get_window_extent(renderer)
    placed_bboxes = []

    def segment_intersects_bbox(p1, p2, bbox: Bbox) -> bool:
        if bbox.contains(p1[0], p1[1]) or bbox.contains(p2[0], p2[1]):
            return True

        def ccw(a, b, c) -> bool:
            return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])

        def intersects(a, b, c, d) -> bool:
            return ccw(a, c, d) != ccw(b, c, d) and ccw(a, b, c) != ccw(a, b, d)

        corners = [
            (bbox.x0, bbox.y0),
            (bbox.x1, bbox.y0),
            (bbox.x1, bbox.y1),
            (bbox.x0, bbox.y1),
        ]
        edges = list(zip(corners, corners[1:] + corners[:1]))
        return any(intersects(p1, p2, e1, e2) for e1, e2 in edges)

    def place_label(index: int, required: bool) -> None:
        time = times[index]
        temp = temps[index]
        time_num = mdates.date2num(time)

        if 0 < index < len(temps) - 1:
            slope = temps[index + 1] - temps[index - 1]
        elif index == 0:
            slope = temps[1] - temps[0]
        else:
            slope = temps[-1] - temps[-2]

        if index == max_index:
            candidates = [((0, 10), "center", "bottom"), ((0, -10), "center", "top")]
        elif index == min_index:
            candidates = [((0, -10), "center", "top"), ((0, 10), "center", "bottom")]
        elif slope < 0:
            candidates = [((0, 10), "center", "bottom"), ((0, -10), "center", "top")]
        elif slope > 0:
            candidates = [((0, -10), "center", "top"), ((0, 10), "center", "bottom")]
        else:
            candidates = [((0, 10), "center", "bottom"), ((0, -10), "center", "top")]

        if debug_labels:
            print(
                f"[label] {time.strftime('%H:%M')} temp={temp:.2f} "
                f"slope={slope:.3f} required={required}"
            )

        slope_abs = abs(slope)
        steep_threshold = 0.15 * temp_range
        flat_threshold = 0.05 * temp_range

        if index in {min_index, max_index}:
            scales = (0.25, 0.45, 0.8, 1.2)
        elif slope_abs >= steep_threshold:
            scales = (0.3, 0.55, 0.9, 1.3)
        elif slope_abs <= flat_threshold:
            scales = (0.45, 0.8, 1.2, 1.6)
        else:
            scales = (0.35, 0.6, 1.0, 1.4)

        for scale in scales:
            for (dx, dy), ha, va in candidates:
                if debug_labels:
                    print(f"  try offset=({dx * scale:.1f},{dy * scale:.1f}) ha={ha} va={va}")
                text = ax_temp.annotate(
                    format_temp_value(temp, decimals),
                    (time, temp),
                    textcoords="offset points",
                    xytext=(dx * scale, dy * scale),
                    ha=ha,
                    va=va,
                    fontsize=11,
                    clip_on=True,
                )
                fig.canvas.draw()
                bbox = text.get_window_extent(renderer)
                point_disp = ax_temp.transData.transform((time_num, temp))
                padded_bbox = Bbox.from_extents(
                    bbox.x0 - 2,
                    bbox.y0 - 2,
                    bbox.x1 + 2,
                    bbox.y1 + 2,
                )
                line_overlap = False
                if index > 0:
                    p1 = ax_temp.transData.transform((times_num[index - 1], temps[index - 1]))
                    p2 = ax_temp.transData.transform((times_num[index], temps[index]))
                    line_overlap |= segment_intersects_bbox(p1, p2, padded_bbox)
                if index < len(temps) - 1:
                    p1 = ax_temp.transData.transform((times_num[index], temps[index]))
                    p2 = ax_temp.transData.transform((times_num[index + 1], temps[index + 1]))
                    line_overlap |= segment_intersects_bbox(p1, p2, padded_bbox)

                inside_axes = (
                    axes_bbox.contains(bbox.x0, bbox.y0)
                    and axes_bbox.contains(bbox.x1, bbox.y1)
                )
                overlaps = any(bbox.overlaps(prev) for prev in placed_bboxes)
                covers_point = bbox.contains(point_disp[0], point_disp[1])
                point_inflated = bbox.expanded(1.35, 1.6)
                close_to_point = point_inflated.contains(point_disp[0], point_disp[1])

                if debug_labels:
                    print(
                        f"    inside={inside_axes} overlap={overlaps} line={line_overlap} "
                        f"covers_point={covers_point} close_to_point={close_to_point}"
                    )

                if (
                    inside_axes
                    and not overlaps
                    and not line_overlap
                    and not covers_point
                    and not close_to_point
                ):
                    placed_bboxes.append(bbox)
                    if debug_labels:
                        print("    -> placed")
                    return

                text.remove()

        if required:
            if debug_labels:
                print("  -> forced placement")
            text = ax_temp.annotate(
                format_temp_value(temp, decimals),
                (time, temp),
                textcoords="offset points",
                xytext=(0, 10),
                ha="center",
                va="bottom",
                fontsize=11,
                clip_on=True,
            )
            fig.canvas.draw()
            placed_bboxes.append(text.get_window_extent(renderer))

    for idx in sorted(annotate_indices):
        place_label(idx, required=False)

    ax_temp.grid(True, **grid_style)
    ax_precip.grid(True, **grid_style)
    if ax_week is not None:
        ax_week.grid(True, **grid_style)

    if ax_week is not None:
        weekly_num = mdates.date2num(weekly_days)
        dense_week_num = np.linspace(weekly_num[0], weekly_num[-1], len(weekly_num) * 6)
        dense_min = np.interp(dense_week_num, weekly_num, weekly_min)
        dense_max = np.interp(dense_week_num, weekly_num, weekly_max)
        kernel = np.ones(5) / 5
        pad = 2
        smooth_min = np.convolve(np.pad(dense_min, (pad, pad), mode="edge"), kernel, mode="valid")
        smooth_max = np.convolve(np.pad(dense_max, (pad, pad), mode="edge"), kernel, mode="valid")
        smooth_days = mdates.num2date(dense_week_num, tz=KYIV_TZ)

        ax_week.plot(smooth_days, smooth_max, color="#d62728", linewidth=2)
        ax_week.plot(smooth_days, smooth_min, color="#1f77b4", linewidth=2)
        below_top = np.minimum(smooth_max, 0)
        above_bottom = np.maximum(smooth_min, 0)
        ax_week.fill_between(
            smooth_days,
            smooth_min,
            below_top,
            where=smooth_min < 0,
            color="#1f77b4",
            alpha=0.35,
            interpolate=True,
        )
        ax_week.fill_between(
            smooth_days,
            above_bottom,
            smooth_max,
            where=smooth_max > 0,
            color="#1f77b4",
            alpha=0.15,
            interpolate=True,
        )
        ax_week.xaxis.set_major_formatter(
            FuncFormatter(
                lambda x, pos: f"{UA_WEEKDAYS_SHORT[mdates.num2date(x, tz=KYIV_TZ).weekday()]} ({mdates.num2date(x, tz=KYIV_TZ).strftime('%d')})"
            )
        )
        ax_week.xaxis.set_major_locator(mdates.DayLocator(interval=1))
        ax_week.set_title("Прогноз на 10 днів", loc="left", fontsize=16, fontweight="semibold", pad=20)
        week_min = min(weekly_min)
        week_max = max(weekly_max)
        week_range = max(week_max - week_min, 1.0)
        ax_week.set_ylim(week_min - 0.35 * week_range, week_max + 0.35 * week_range)
        if weekly_precip is not None:
            for day, pmax in zip(weekly_days, weekly_precip):
                ax_week.annotate(
                    f"{pmax:.0f}%",
                    (mdates.date2num(day), 1.0),
                    xycoords=("data", "axes fraction"),
                    textcoords="offset points",
                    xytext=(0, 2),
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    color="#666666",
                    clip_on=False,
                )
        for day, tmin, tmax in zip(weekly_days, weekly_min, weekly_max):
            ax_week.annotate(
                f"{tmax:.0f}°",
                (day, tmax),
                textcoords="offset points",
                xytext=(0, 6),
                ha="center",
                va="bottom",
                fontsize=9,
            )
            ax_week.annotate(
                f"{tmin:.0f}°",
                (day, tmin),
                textcoords="offset points",
                xytext=(0, -8),
                ha="center",
                va="top",
                fontsize=9,
            )

    fig.tight_layout()
    title_lines = title.count("\n") + 1
    top_margin = 0.93 - 0.03 * (title_lines - 1)
    fig.subplots_adjust(hspace=0.45, top=max(0.85, top_margin))
    if ax_week is not None:
        pos_temp = ax_temp.get_position()
        pos_precip = ax_precip.get_position()
        gap_01 = pos_temp.y0 - pos_precip.y1
        if gap_01 > 0:
            ax_precip.set_position(
                [pos_precip.x0, pos_precip.y0 + gap_01, pos_precip.width, pos_precip.height]
            )
    if ax_week is not None:
        pos_temp = ax_temp.get_position()
        pos_precip = ax_precip.get_position()
        desired_gap_01 = 0.01
        gap_01 = pos_temp.y0 - pos_precip.y1
        if gap_01 > desired_gap_01:
            shift_up = gap_01 - desired_gap_01
            ax_precip.set_position(
                [pos_precip.x0, pos_precip.y0 + shift_up, pos_precip.width, pos_precip.height]
            )
        pos_precip = ax_precip.get_position()
        pos_week = ax_week.get_position()
        desired_gap_12 = 0.1
        gap_12 = pos_precip.y0 - pos_week.y1
        if gap_12 < desired_gap_12:
            shift_down = desired_gap_12 - gap_12
            ax_week.set_position(
                [pos_week.x0, pos_week.y0 - shift_down, pos_week.width, pos_week.height]
            )
    if ax_week is not None:
        fig.align_ylabels([ax_temp, ax_precip, ax_week])
    else:
        fig.align_ylabels([ax_temp, ax_precip])
    pos_temp = ax_temp.get_position()
    title_align = "left" if "\n" in title else "right"
    title_x = pos_temp.x0 if title_align == "left" else pos_temp.x1
    fig.text(
        title_x,
        0.985,
        title,
        ha=title_align,
        va="top",
        fontsize=16,
        fontweight="semibold",
    )
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_chart(
    city_name: str,
    target_date: date,
    hourly_items,
    output_path: Path,
    debug_labels: bool = False,
) -> None:
    times = [item[0] for item in hourly_items]
    temps = [item[1] for item in hourly_items]
    precip = [item[2] for item in hourly_items]
    start_dt = datetime.combine(target_date, dt_time(0, 0), tzinfo=KYIV_TZ)
    end_dt = start_dt + timedelta(hours=23)
    title = format_chart_title(city_name, format_ua_date(target_date))
    plot_chart_base(times, temps, precip, title, output_path, start_dt, end_dt, debug_labels)


def main() -> None:
    args = parse_args()
    target_date = select_target_date(args.day, args.date)
    out_dir = Path(args.out_dir)
    text_path, chart_path, report_text = generate_daily_forecast(
        args.city,
        target_date,
        out_dir,
        args.debug_labels,
    )

    print(report_text)
    print(f"Збережено текстовий звіт: {text_path}")
    print(f"Збережено графіки: {chart_path}")


if __name__ == "__main__":
    main()
