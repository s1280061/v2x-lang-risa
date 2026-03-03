from pathlib import Path
import json
import shutil
from collections import defaultdict

FRAMES = Path(r"D:\V2X\pair_gallery_full\frames")
META = Path(r"D:\V2X\pair_gallery_full\frames_meta")
OUT = Path(r"D:\V2X\pair_gallery_full\by_sequence")

OUT.mkdir(parents=True, exist_ok=True)

groups = defaultdict(list)

for meta_path in sorted(META.glob("*.json")):
    idx = meta_path.stem
    m = json.loads(meta_path.read_text(encoding="utf-8"))
    seq = m.get("vehicle_sequence", "unknown")
    groups[seq].append(idx)

print("num sequences:", len(groups))

for seq, ids in groups.items():
    d = OUT / f"seq_{seq}"
    d.mkdir(parents=True, exist_ok=True)

    for i in ids:
        src = FRAMES / f"{i}.jpg"
        dst = d / f"{i}.jpg"
        if not dst.exists():
            shutil.copy2(src, dst)

    print(f"seq {seq}: {len(ids)} frames")

print("Done.")
