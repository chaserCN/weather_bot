#!/usr/bin/env python3
import argparse
import asyncio
import os
from pathlib import Path

os.environ.setdefault("PLAYWRIGHT_BROWSERS_PATH", "0")

from playwright.async_api import async_playwright


DEFAULT_URL = "https://www.ventusky.com/#p=48.8;31.4;5"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Зробити скріншот карти Ventusky за URL."
    )
    parser.add_argument(
        "--url",
        default=DEFAULT_URL,
        help="Посилання Ventusky з координатами та масштабом.",
    )
    parser.add_argument(
        "--out",
        default="outputs/ventusky.png",
        help="Шлях до файлу зображення.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1280,
        help="Ширина вікна браузера.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=720,
        help="Висота вікна браузера.",
    )
    parser.add_argument(
        "--wait",
        type=int,
        default=5000,
        help="Додатковий час очікування (мс) після завантаження сторінки.",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Сховати елементи інтерфейсу перед знімком.",
    )
    parser.add_argument(
        "--clip-map",
        action="store_true",
        help="Обрізати знімок до найбільшого canvas (карта).",
    )
    parser.add_argument(
        "--storage-state",
        help="Шлях до файлу стану браузера (cookies/localStorage).",
    )
    parser.add_argument(
        "--save-state",
        help="Куди зберегти стан браузера після ручного налаштування.",
    )
    parser.add_argument(
        "--headed",
        action="store_true",
        help="Запускати браузер у видимому режимі.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Друкувати діагностичну інформацію про карту.",
    )
    return parser.parse_args()


async def hide_ui(page) -> None:
    await page.add_style_tag(
        content="""
        header, nav, aside, .sidebar, .side-panel, .panel, .toolbar, .menu,
        .top-bar, .bottom-bar, .controls, .leaflet-control-container,
        .popup, .modal, .legend, .search, .logo {
            display: none !important;
        }
        """
    )


async def find_map_element(page):
    selectors = [
        "#map",
        ".map",
        ".map-container",
        ".leaflet-container",
        ".mapboxgl-canvas",
        "canvas",
    ]
    best = None
    best_area = 0.0
    best_selector = None
    for selector in selectors:
        locator = page.locator(selector)
        count = await locator.count()
        for idx in range(count):
            handle = await locator.nth(idx).element_handle()
            if not handle:
                continue
            box = await handle.bounding_box()
            if not box:
                continue
            area = box["width"] * box["height"]
            if area > best_area:
                best = handle
                best_area = area
                best_selector = selector
    return best, best_selector, best_area


async def promote_map_element(page, element):
    if not element:
        return None, None
    tag = await element.evaluate("el => el.tagName")
    if tag != "CANVAS":
        return element, None
    parent = await page.evaluate_handle(
        """
        (el) => el.closest('#map, .map, .map-container, .leaflet-container') || el.parentElement
        """,
        element,
    )
    if parent:
        return parent.as_element(), "promoted-parent"
    return element, None


async def capture(
    url: str,
    output_path: Path,
    width: int,
    height: int,
    wait_ms: int,
    clean: bool,
    clip_map: bool,
    debug: bool,
    storage_state: str | None,
    save_state: str | None,
    headed: bool,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=not headed)
        context = await browser.new_context(
            viewport={"width": width, "height": height},
            device_scale_factor=2,
            storage_state=storage_state,
        )
        page = await context.new_page()
        await page.goto(url, wait_until="networkidle", timeout=60000)
        await page.wait_for_timeout(wait_ms)
        if save_state:
            print("Налаштуйте шар на сторінці, потім натисніть Enter...")
            await asyncio.get_running_loop().run_in_executor(None, input)
            await context.storage_state(path=save_state)
            await page.wait_for_timeout(1000)
        map_element, map_selector, map_area = await find_map_element(page)
        map_element, promoted_note = await promote_map_element(page, map_element)
        if debug:
            extra = f" ({promoted_note})" if promoted_note else ""
            print(f"map selector: {map_selector} area: {map_area:.0f}{extra}")

        if clean:
            await hide_ui(page)
            if map_element:
                await page.evaluate(
                    """
                    (target) => {
                        const keep = (el) => el === target || target.contains(el) || el.contains(target);
                        document.querySelectorAll("body *").forEach((el) => {
                            if (!keep(el)) {
                                el.style.visibility = "hidden";
                            } else {
                                el.style.visibility = "visible";
                            }
                        });
                    }
                    """,
                    map_element,
                )
            await page.wait_for_timeout(500)

        if clip_map and map_element:
            box = await map_element.bounding_box()
            if box:
                await page.screenshot(path=str(output_path), clip=box)
            else:
                await page.screenshot(path=str(output_path), full_page=True)
        else:
            await page.screenshot(path=str(output_path), full_page=True)
        await browser.close()


def main() -> None:
    args = parse_args()
    output_path = Path(args.out)
    asyncio.run(
        capture(
            args.url,
            output_path,
            args.width,
            args.height,
            args.wait,
            args.clean,
            args.clip_map,
            args.debug,
            args.storage_state,
            args.save_state,
            args.headed,
        )
    )
    print(f"Збережено зображення: {output_path}")


if __name__ == "__main__":
    main()
