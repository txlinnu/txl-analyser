"""One-off script: generates the PWA icon set for TXL Remote Control
(static/icons/, txlremote- prefixed so they don't collide with the other
apps' icon files in the same shared static folder). Not part of the
running app - run manually if you ever want to regenerate them."""

from pathlib import Path

from PIL import Image, ImageDraw

ACCENT = (180, 83, 9, 255)  # #b45309 - matches TXL Remote Control's --accent
WHITE = (255, 255, 255, 255)

OUT_DIR = Path(__file__).resolve().parent.parent / "static" / "icons"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def draw_signal(draw: ImageDraw.ImageDraw, size: int, scale: float):
    # A broadcast/radar dot: a filled center dot plus two concentric rings -
    # echoes the 📡 favicon without depending on emoji font rendering.
    cx = cy = size / 2
    r = size * scale
    dot_r = r * 0.22
    draw.ellipse([cx - dot_r, cy - dot_r, cx + dot_r, cy + dot_r], fill=WHITE)
    for i, frac in enumerate((0.55, 1.0)):
        ring_r = r * frac
        width = max(2, int(size * 0.035))
        draw.arc(
            [cx - ring_r, cy - ring_r, cx + ring_r, cy + ring_r],
            start=215, end=325, fill=WHITE, width=width,
        )


def make_icon(size: int, filename: str, maskable: bool = False):
    img = Image.new("RGBA", (size, size), ACCENT)
    draw = ImageDraw.Draw(img)
    scale = 0.30 if maskable else 0.40
    draw_signal(draw, size, scale)
    img.save(OUT_DIR / filename)


make_icon(192, "txlremote-icon-192.png")
make_icon(512, "txlremote-icon-512.png")
make_icon(512, "txlremote-icon-maskable-512.png", maskable=True)
print("Icons written to", OUT_DIR)
