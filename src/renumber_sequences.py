from pathlib import Path

BASE = Path(r"D:\V2X\pair_gallery_full\by_sequence")

for seq_dir in BASE.iterdir():
    if not seq_dir.is_dir():
        continue

    files = sorted(seq_dir.glob("*.jpg"))
    for i, f in enumerate(files):
        new_name = seq_dir / f"{i:06d}.jpg"
        if f != new_name:
            f.rename(new_name)

    print(f"{seq_dir.name}: renamed {len(files)} files")

print("Done.")
