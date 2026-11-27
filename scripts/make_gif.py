"""README hero GIF from the backup-demo frames (frontend/scripts/record_demo.mjs --gif-frames DIR).

  python scripts/make_gif.py DIR docs/media/hero.gif   (PIL only; ~16 s highlight: campus, circuit, neighbours, scorecard)"""
import json
import sys
from pathlib import Path

from PIL import Image

WINDOWS = [("2", 2.0), ("4c", 3.0), ("4d", 2.0), ("5", 5.0), ("7a", 3.0)]   # chapter, seconds to keep from its start


def main(src: str, dst: str, width: int = 720):
    d = Path(src)
    meta = json.loads((d / "marks.json").read_text(encoding="utf-8"))
    marks = {m["stop"]: m["t"] for m in meta["marks"]}
    files = sorted(d.glob("f*.jpg"))
    n = len(files)
    span = marks[max(marks, key=marks.get)] + 4.0          # approximate: frames are evenly spread over the recording
    keep = []
    for stop, secs in WINDOWS:
        a = marks[stop] + 0.6
        keep += [f for k, f in enumerate(files) if a <= k / n * span <= a + secs]
    ims = []
    for f in keep:
        im = Image.open(f).convert("RGB")
        im = im.resize((width, round(im.height * width / im.width)), Image.LANCZOS)
        ims.append(im.quantize(colors=128, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE))
    ims[0].save(dst, save_all=True, append_images=ims[1:], duration=130, loop=0, optimize=True)
    print(f"wrote {dst}: {len(ims)} frames, {Path(dst).stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
