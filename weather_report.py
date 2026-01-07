#!/usr/bin/env python3
import argparse
from datetime import date, datetime, timedelta
from datetime import time as dt_time
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FixedLocator
import requests


CITY_COORDS = {
    "chernihiv": {"name": "Чернігів", "lat": 51.4982, "lon": 31.2893},
    "dnipro": {"name": "Дніпро", "lat": 48.4647, "lon": 35.0462},
}

KYIV_TZ = ZoneInfo("Europe/Kyiv")
API_URL = "https://api.open-meteo.com/v1/forecast"
UA_WEEKDAYS = [
    "понеділок",
    "вівторок",
    "середа",
    "четвер",
    "пʼятниця",
    "субота",
    "неділя",
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
        "hourly": "temperature_2m,precipitation_probability",
        "daily": "temperature_2m_max,temperature_2m_min,precipitation_probability_max",
        "forecast_days": forecast_days,
        "timezone": "Europe/Kyiv",
    }
    response = requests.get(API_URL, params=params, timeout=30)
    response.raise_for_status()
    return response.json()


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

    forecast_days = max(7, days_ahead + 1)
    data = fetch_forecast(city["lat"], city["lon"], forecast_days)
    hourly_items = extract_hourly(data, target_date)
    if not hourly_items:
        raise SystemExit("Немає погодинних даних для вибраної дати.")

    weekly_lines = build_weekly_lines(data, today)
    report_text = build_text_report(city["name"], target_date, hourly_items, weekly_lines)

    out_dir.mkdir(parents=True, exist_ok=True)
    base_name = build_base_name(city_key, None, target_date.isoformat())
    text_path = out_dir / f"{base_name}.txt"
    chart_path = out_dir / f"{base_name}.png"

    text_path.write_text(report_text, encoding="utf-8")
    plot_chart(city["name"], target_date, hourly_items, chart_path, debug_labels=debug_labels)
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
    forecast_days = max(2, days_ahead + 1)
    data = fetch_forecast(city["lat"], city["lon"], forecast_days)
    hourly_items = extract_hourly_range(data, start_dt, end_dt)
    if not hourly_items:
        raise SystemExit("Немає погодинних даних для наступних 24 годин.")

    report_text = build_text_report_24h(city["name"], start_dt, hourly_items)

    out_dir.mkdir(parents=True, exist_ok=True)
    base_name = f"{city_key}_next24h_{start_dt.strftime('%Y-%m-%d_%H%M')}"
    text_path = out_dir / f"{base_name}.txt"
    chart_path = out_dir / f"{base_name}.png"

    text_path.write_text(report_text, encoding="utf-8")
    title = f"{city['name']} — {start_dt.strftime('%d.%m %H:%M')} +24г"
    times = [item[0] for item in hourly_items]
    temps = [item[1] for item in hourly_items]
    precip = [item[2] for item in hourly_items]
    plot_chart_base(times, temps, precip, title, chart_path, start_dt, end_dt, debug_labels)
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
) -> None:
    matplotlib.use("Agg")
    plt.rcParams.update(
        {
            "font.size": 13,
            "axes.titlesize": 16,
            "axes.labelsize": 14,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
        }
    )

    fig, (ax_temp, ax_precip) = plt.subplots(
        nrows=2, ncols=1, figsize=(4.2, 5.9), sharex=True
    )

    times_num = mdates.date2num(times)
    dense_num = np.linspace(times_num[0], times_num[-1], len(times_num) * 4)
    dense_temps = np.interp(dense_num, times_num, temps)
    window = 9
    kernel = np.ones(window) / window
    pad = window // 2
    padded = np.pad(dense_temps, (pad, pad), mode="edge")
    smooth_temps = np.convolve(padded, kernel, mode="valid")
    smooth_times = mdates.num2date(dense_num, tz=KYIV_TZ)

    ax_temp.plot(smooth_times, smooth_temps, color="#1f77b4", linewidth=2.5)
    temp_min = min(temps)
    temp_max = max(temps)
    temp_range = max(temp_max - temp_min, 1.0)
    ax_temp.set_ylim(temp_min - 0.35 * temp_range, temp_max + 0.35 * temp_range)

    labeled_indices = [
        idx
        for idx, time in enumerate(times)
        if time.minute == 0 and time.hour in {0, 6, 12, 18}
    ]
    marker_times = [times[idx] for idx in labeled_indices]
    marker_temps = [temps[idx] for idx in labeled_indices]
    ax_temp.scatter(marker_times, marker_temps, color="#1f77b4", s=60, zorder=3)

    min_index = temps.index(temp_min)
    max_index = temps.index(temp_max)
    annotate_indices = {0, len(temps) - 1, *labeled_indices}

    ax_precip.bar(times, precip, color="#ff7f0e", alpha=0.6, width=0.03)
    ax_precip.set_ylim(0, 100)

    ax_temp.set_title(title, fontweight="semibold", pad=22)
    ax_temp.set_ylabel("Температура (°C)")
    ax_precip.set_ylabel("Ймовірність опадів (%)")

    set_time_axis(ax_temp, start_dt, end_dt)
    tick_times = []
    tick = start_dt.replace(minute=0, second=0, microsecond=0)
    while tick <= end_dt:
        tick_times.append(tick)
        tick += timedelta(hours=3)
    hour_ticks = FixedLocator(mdates.date2num(tick_times))
    ax_temp.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=KYIV_TZ))
    ax_temp.xaxis.set_major_locator(hour_ticks)
    ax_temp.tick_params(axis="x", labelrotation=45, labelbottom=True)

    ax_precip.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=KYIV_TZ))
    ax_precip.xaxis.set_major_locator(hour_ticks)
    plt.setp(ax_precip.get_xticklabels(), rotation=45, ha="right")
    plt.setp(ax_temp.get_xticklabels(), rotation=45, ha="right")

    y_ticks = ax_temp.get_yticks()
    has_fractional_ticks = any(abs(tick - round(tick)) > 1e-6 for tick in y_ticks)
    label_fmt = "{:.1f}°" if has_fractional_ticks else "{:.0f}°"

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    axes_bbox = ax_temp.get_window_extent(renderer)
    placed_bboxes = []

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
            candidates = [
                ((0, 10), "center", "bottom"),
                ((8, 10), "left", "bottom"),
                ((-8, 10), "right", "bottom"),
            ]
        elif index == min_index:
            candidates = [
                ((0, -10), "center", "top"),
                ((8, -10), "left", "top"),
                ((-8, -10), "right", "top"),
            ]
        elif slope < 0:
            candidates = [((8, 10), "left", "bottom"), ((-8, -10), "right", "top")]
        elif slope > 0:
            candidates = [((-8, 10), "right", "bottom"), ((8, -10), "left", "top")]
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
                    label_fmt.format(temp),
                    (time, temp),
                    textcoords="offset points",
                    xytext=(dx * scale, dy * scale),
                    ha=ha,
                    va=va,
                    fontsize=12,
                    clip_on=True,
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=0.3),
                )
                fig.canvas.draw()
                bbox = text.get_window_extent(renderer)
                point_disp = ax_temp.transData.transform((time_num, temp))

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
                        f"    inside={inside_axes} overlap={overlaps} "
                        f"covers_point={covers_point} close_to_point={close_to_point}"
                    )

                if inside_axes and not overlaps and not covers_point and not close_to_point:
                    placed_bboxes.append(bbox)
                    if debug_labels:
                        print("    -> placed")
                    return

                text.remove()

        if required:
            if debug_labels:
                print("  -> forced placement")
            text = ax_temp.annotate(
                label_fmt.format(temp),
                (time, temp),
                textcoords="offset points",
                xytext=(0, 10),
                ha="center",
                va="bottom",
                fontsize=12,
                clip_on=True,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=0.3),
            )
            fig.canvas.draw()
            placed_bboxes.append(text.get_window_extent(renderer))

    required_indices = {0, len(temps) - 1}
    for idx in sorted(annotate_indices):
        place_label(idx, required=idx in required_indices)

    ax_temp.grid(True, linestyle="--", alpha=0.3)
    ax_precip.grid(True, linestyle="--", alpha=0.3)

    fig.tight_layout()
    fig.subplots_adjust(hspace=0.35, top=0.9)
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
    title = f"{city_name} — {format_ua_date(target_date)}"
    plot_chart_base(times, temps, precip, title, output_path, start_dt, end_dt, debug_labels)


def main() -> None:
    args = parse_args()
    city = CITY_COORDS[args.city]
    target_date = select_target_date(args.day, args.date)

    today = datetime.now(KYIV_TZ).date()
    days_ahead = (target_date - today).days
    if days_ahead < 0:
        raise SystemExit("Дата прогнозу вже минула.")
    if days_ahead >= 16:
        raise SystemExit("Доступний прогноз лише на 16 днів уперед.")

    forecast_days = max(7, days_ahead + 1)
    data = fetch_forecast(city["lat"], city["lon"], forecast_days)
    hourly_items = extract_hourly(data, target_date)
    if not hourly_items:
        raise SystemExit("Немає погодинних даних для вибраної дати.")

    weekly_lines = build_weekly_lines(data, today)
    report_text = build_text_report(city["name"], target_date, hourly_items, weekly_lines)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    base_name = build_base_name(args.city, args.day, args.date)
    text_path = out_dir / f"{base_name}.txt"
    chart_path = out_dir / f"{base_name}.png"

    text_path.write_text(report_text, encoding="utf-8")
    plot_chart(
        city["name"],
        target_date,
        hourly_items,
        chart_path,
        debug_labels=args.debug_labels,
    )

    print(report_text)
    print(f"Збережено текстовий звіт: {text_path}")
    print(f"Збережено графіки: {chart_path}")


if __name__ == "__main__":
    main()
