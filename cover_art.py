"""Per-episode cover art via Pillow.

Loads docs/cover.png as a base, overlays date + lead-headline. Saves to
docs/episodes/YYYY-MM-DD.jpg. Referenced from the per-item <itunes:image>
in the RSS feed so episode lists in Apple/Spotify show fresh art each day.

Falls back to copying the base cover if Pillow isn't installed or the
overlay fails.
"""
from __future__ import annotations
import shutil
from datetime import datetime
from pathlib import Path
from config import DOCS, EPISODES_DIR

BASE_COVER_PNG = DOCS / "cover.png"
BASE_COVER_JPG = DOCS / "cover.jpg"


def _load_base() -> "Image.Image | None":  # type: ignore
    try:
        from PIL import Image
    except ImportError:
        return None
    src = BASE_COVER_PNG if BASE_COVER_PNG.exists() else BASE_COVER_JPG
    if not src.exists():
        return None
    try:
        return Image.open(src).convert("RGB")
    except Exception:
        return None


def _find_font():
    """Locate a usable serif/sans font on the runner. Falls back to PIL's
    default bitmap font if nothing system-installed is available."""
    from PIL import ImageFont
    candidates = [
        "/System/Library/Fonts/Supplemental/Iowan Old Style.ttc",
        "/System/Library/Fonts/Supplemental/Charter.ttc",
        "/System/Library/Fonts/Supplemental/Georgia.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSerif-Bold.ttf",
    ]
    for p in candidates:
        if Path(p).exists():
            return p
    return None


def write_episode_cover(date_str: str, lead_title: str) -> Path | None:
    """Render and save a per-episode JPG. Returns path or None on failure."""
    from PIL import Image, ImageDraw, ImageFont
    base = _load_base()
    if base is None:
        return None
    try:
        target = base.copy().resize((3000, 3000), Image.LANCZOS)
        draw = ImageDraw.Draw(target, "RGBA")
        font_path = _find_font()

        try:
            d = datetime.strptime(date_str, "%Y-%m-%d")
            date_label = d.strftime("%a %b %-d, %Y").upper()
        except ValueError:
            date_label = date_str.upper()

        # bottom strip with translucent overlay
        strip_h = 700
        draw.rectangle([(0, 3000 - strip_h), (3000, 3000)], fill=(0, 0, 0, 180))

        if font_path:
            date_font = ImageFont.truetype(font_path, 80)
            title_font = ImageFont.truetype(font_path, 130)
        else:
            date_font = ImageFont.load_default()
            title_font = ImageFont.load_default()

        # date label, top of strip
        draw.text((100, 3000 - strip_h + 70), date_label, font=date_font, fill=(220, 220, 220, 255))

        # word-wrap headline
        max_chars_per_line = 30
        words = lead_title.split()
        lines: list[str] = []
        current = ""
        for w in words:
            if len(current) + 1 + len(w) > max_chars_per_line and current:
                lines.append(current)
                current = w
            else:
                current = (current + " " + w).strip()
        if current:
            lines.append(current)
        lines = lines[:3]

        y = 3000 - strip_h + 200
        for line in lines:
            draw.text((100, y), line, font=title_font, fill=(255, 255, 255, 255))
            y += 160

        EPISODES_DIR.mkdir(parents=True, exist_ok=True)
        out = EPISODES_DIR / f"{date_str}.jpg"
        target.save(out, "JPEG", quality=85, optimize=True)
        return out
    except Exception as e:
        print(f"[cover] generation failed ({e}); using base cover")
        return None
