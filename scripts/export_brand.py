"""Rasterize the original SVG identity in Chromium for favicon and sharing formats."""

import base64
import struct
from pathlib import Path

from playwright.sync_api import sync_playwright

BRAND = Path(__file__).resolve().parents[1] / "src/rag_jev/static/brand"


def main():
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()

        def render(source, target, width, height, image_height=None):
            svg = base64.b64encode((BRAND / source).read_bytes()).decode()
            page = browser.new_page(viewport={"width": width, "height": height})
            page.set_content(
                '<html><body style="margin:0;display:grid;place-items:center;'
                'height:100vh;background:#f5f1e7">'
                f'<img width="{width}" height="{image_height or height}" '
                f'src="data:image/svg+xml;base64,{svg}"></body></html>'
            )
            page.locator("img").evaluate("image => image.decode()")
            page.screenshot(path=str(BRAND / target))
            page.close()

        render("banner.svg", "banner.png", 1280, 420)
        render("banner.svg", "social-preview.png", 1280, 640, 420)
        render("favicon.svg", "favicon-32.png", 32, 32)
        render("favicon.svg", "apple-touch-icon.png", 180, 180)
        png = (BRAND / "favicon-32.png").read_bytes()
        header = struct.pack("<HHH", 0, 1, 1)
        directory = struct.pack("<BBBBHHII", 32, 32, 0, 0, 1, 32, len(png), 22)
        (BRAND / "favicon.ico").write_bytes(header + directory + png)
        browser.close()


if __name__ == "__main__":
    main()
