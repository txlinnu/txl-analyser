"""One-off script: generates the PWA icon set (static/icons/). Not part of
the running app - run manually if you ever want to regenerate them."""

import math
from pathlib import Path

from PIL import Image, ImageDraw

ACCENT = (193, 95, 60, 255)   # #C15F3C - matches the app's --accent
WHITE = (255, 255, 255, 255)

OUT_DIR = Path(__file__).resolve().parent.parent / "static" / "icons"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def draw_star(draw: ImageDraw.ImageDraw, cx: float, cy: float, r_outer: float, r_inner: float, fill):
    points = []
    for i in range(8):
        angle = math.pi / 2 + i * math.pi / 4
        r = r_outer if i % 2 == 0 else r_inner
        points.append((cx + r * math.cos(angle), cy - r * math.sin(angle)))
    draw.polygon(points, fill=fill)


def make_icon(size: int, filename: str, maskable: bool = False):
    img = Image.new("RGBA", (size, size), ACCENT)
    draw = ImageDraw.Draw(img)
    cx = cy = size / 2
    # Maskable icons need the important content inside a safe zone (~80% of
    # the canvas) since the OS may crop the rest into a circle/rounded shape.
    scale = 0.34 if maskable else 0.42
    draw_star(draw, cx, cy, size * scale, size * scale * 0.42, WHITE)
    img.save(OUT_DIR / filename)


make_icon(192, "icon-192.png")
make_icon(512, "icon-512.png")
make_icon(512, "icon-maskable-512.png", maskable=True)
print("Icons written to", OUT_DIR)
