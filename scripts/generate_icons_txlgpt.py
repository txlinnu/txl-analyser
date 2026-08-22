"""One-off script: generates the PWA icon set for Txl GPT (static/icons/,
txlgpt- prefixed so they don't collide with TXL Cloud's own icon files in
the same shared static folder). Not part of the running app - run manually
if you ever want to regenerate them."""

from pathlib import Path

from PIL import Image, ImageDraw

ACCENT = (16, 163, 127, 255)  # #10A37F - matches Txl GPT's --accent
WHITE = (255, 255, 255, 255)

OUT_DIR = Path(__file__).resolve().parent.parent / "static" / "icons"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def draw_t(draw: ImageDraw.ImageDraw, size: int, scale: float):
    # A simple bold "T" glyph, built from two rectangles - no font dependency.
    w = size * scale
    bar_h = w * 0.22
    stem_w = w * 0.26
    cx = size / 2
    top = size / 2 - w / 2
    draw.rectangle([cx - w / 2, top, cx + w / 2, top + bar_h], fill=WHITE)
    draw.rectangle([cx - stem_w / 2, top, cx + stem_w / 2, top + w], fill=WHITE)


def make_icon(size: int, filename: str, maskable: bool = False):
    img = Image.new("RGBA", (size, size), ACCENT)
    draw = ImageDraw.Draw(img)
    # Maskable icons need the important content inside a safe zone (~80% of
    # the canvas) since the OS may crop the rest into a circle/rounded shape.
    scale = 0.34 if maskable else 0.42
    draw_t(draw, size, scale)
    img.save(OUT_DIR / filename)


make_icon(192, "txlgpt-icon-192.png")
make_icon(512, "txlgpt-icon-512.png")
make_icon(512, "txlgpt-icon-maskable-512.png", maskable=True)
print("Icons written to", OUT_DIR)
