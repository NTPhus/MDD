"""Create labeled previews of the 35 original extracted notebook images."""
import json
from pathlib import Path
import textwrap

from PIL import Image, ImageDraw, ImageFont

REPORT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
MANIFEST = json.loads((REPORT / "figures_manifest.json").read_text(encoding="utf-8"))
FONT = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 24)
BOLD = ImageFont.truetype("C:/Windows/Fonts/arialbd.ttf", 28)
SMALL = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 22)
WIDTH, HEIGHT = 2400, 2250
TILE_W, TILE_H = 1170, 700

for page, start in enumerate(range(0, len(MANIFEST["figures"]), 6), start=1):
    sheet = Image.new("RGB", (WIDTH, HEIGHT), "#e9edf2")
    draw = ImageDraw.Draw(sheet)
    draw.text((35, 20), f"Notebook figures — sheet {page:02d} — original images retained separately", font=BOLD, fill="#142334")
    for offset, entry in enumerate(MANIFEST["figures"][start:start + 6]):
        left = 25 + (offset % 2) * 1190
        top = 80 + (offset // 2) * 715
        draw.rectangle((left, top, left + TILE_W, top + TILE_H), fill="white", outline="#9aa8b7", width=2)
        stem = Path(entry["notebook"]).stem
        label = f"ID {entry['figure_id']:02d} | {stem} | cell {entry['cell_number_1']}"
        draw.text((left + 15, top + 10), label, font=FONT, fill="#142334")
        heading = entry.get("nearest_heading", "")
        for row, line in enumerate(textwrap.wrap(heading, width=85)[:2]):
            draw.text((left + 15, top + 45 + 27 * row), line, font=SMALL, fill="#35455a")
        original = Image.open(REPORT / entry["file"]).convert("RGB")
        original.thumbnail((TILE_W - 30, TILE_H - 140), Image.Resampling.LANCZOS)
        x = left + (TILE_W - original.width) // 2
        y = top + 110 + (TILE_H - 140 - original.height) // 2
        sheet.paste(original, (x, y))
        dimensions = entry["dimensions"]
        draw.text((left + 15, top + TILE_H - 30), f"{dimensions['width']} × {dimensions['height']} px | output {entry['output_index_0'] + 1}", font=SMALL, fill="#596a7d")
    destination = OUT / f"figures_contact_{page:02d}.png"
    sheet.save(destination, optimize=True)
    print(destination)
