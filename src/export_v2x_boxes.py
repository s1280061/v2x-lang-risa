from __future__ import annotations
from pathlib import Path
import json
import csv

# ====== Paths ======
BASE = Path(r"D:\V2X\Public-V2X-Datasets\V2X-Seq (CVPR2023)\Sequential-Perception-Dataset\Full Dataset (train & val)\V2X-Seq-SPD\V2X-Seq-SPD")

PAIR_ROOT = Path(r"D:\V2X\pair_V2X")
INFRA_DIR = PAIR_ROOT / "infra"
EGO_DIR   = PAIR_ROOT / "ego"
META_DIR  = PAIR_ROOT / "meta"

OUT_DIR = PAIR_ROOT / "exports"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_JSONL = OUT_DIR / "pairs_boxes.jsonl"
OUT_CSV   = OUT_DIR / "pairs_boxes.csv"

# 全ペア数（必要なら変更）
MAX_PAIRS = 10761

# ====== Utils ======
def load_json(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))

def norm_box(b: dict) -> dict | None:
    """Return normalized box dict or None."""
    if not isinstance(b, dict):
        return None
    keys = ("xmin", "ymin", "xmax", "ymax")
    if not all(k in b for k in keys):
        return None
    try:
        xmin = float(b["xmin"]); ymin = float(b["ymin"])
        xmax = float(b["xmax"]); ymax = float(b["ymax"])
    except Exception:
        return None
    # 変なboxは落とす
    if xmax <= xmin or ymax <= ymin:
        return None
    return {"xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax}

def to_obj_record(obj: dict) -> dict | None:
    """
    Keep minimal but useful fields for VLM:
      - track_id, token (if any), type/class
      - 2d_box
      - 3d_location (if any)
      - 3d_dimensions (if any)
    """
    b = norm_box(obj.get("2d_box"))
    if b is None:
        return None

    rec = {
        "track_id": obj.get("track_id"),
        "token": obj.get("token"),
        "type": obj.get("type") or obj.get("category") or obj.get("name"),
        "2d_box": b,
    }
    if "3d_location" in obj:
        rec["3d_location"] = obj.get("3d_location")
    if "3d_dimensions" in obj:
        rec["3d_dimensions"] = obj.get("3d_dimensions")
    return rec

def extract_seen_hidden(
    coop_label: list[dict],
    infra_label: list[dict],
) -> tuple[list[dict], list[dict]]:
    """
    BLUE(seen): from_side=='coop' で紐づいた infra_track_id を持つ物体
    RED(hidden): infra側に存在するが、seenに含まれない物体 (inf-only)
    """
    seen_ids = {
        x.get("inf_track_id") for x in coop_label
        if x.get("from_side") == "coop" and x.get("inf_track_id") not in (None, "-1", -1)
    }

    seen: list[dict] = []
    hidden: list[dict] = []

    for obj in infra_label:
        rec = to_obj_record(obj)
        if rec is None:
            continue
        tid = rec.get("track_id")
        if tid in seen_ids:
            seen.append(rec)
        else:
            hidden.append(rec)

    return seen, hidden

def main():
    # CSVはbboxをJSON文字列で入れる（壊れないように）
    csv_fields = [
        "pair_index",
        "vehicle_frame",
        "infrastructure_frame",
        "vehicle_sequence",
        "infrastructure_sequence",
        "infra_image_timestamp",
        "ego_image_timestamp",
        "infra_path",
        "ego_path",
        "seen_count",
        "hidden_count",
        "seen_boxes_json",
        "hidden_boxes_json",
    ]

    wrote = 0
    skipped = 0

    with OUT_JSONL.open("w", encoding="utf-8") as fj, OUT_CSV.open("w", encoding="utf-8", newline="") as fc:
        writer = csv.DictWriter(fc, fieldnames=csv_fields)
        writer.writeheader()

        for i in range(MAX_PAIRS):
            meta_path = META_DIR / f"pair_{i:06d}.json"
            infra_img = INFRA_DIR / f"pair_{i:06d}.jpg"
            ego_img   = EGO_DIR   / f"pair_{i:06d}.jpg"

            if not meta_path.exists() or not infra_img.exists() or not ego_img.exists():
                skipped += 1
                continue

            meta = load_json(meta_path)
            veh_frame = str(meta["vehicle_frame"]).zfill(6)
            inf_frame = str(meta["infrastructure_frame"]).zfill(6)

            coop_path = BASE / "cooperative" / "label" / f"{veh_frame}.json"
            inf_label_path = BASE / "infrastructure-side" / "label" / "camera" / f"{inf_frame}.json"

            if not coop_path.exists() or not inf_label_path.exists():
                skipped += 1
                continue

            coop_label = load_json(coop_path)
            infra_label = load_json(inf_label_path)

            seen, hidden = extract_seen_hidden(coop_label, infra_label)

            record = {
                "pair_index": i,
                "vehicle_frame": veh_frame,
                "infrastructure_frame": inf_frame,
                "vehicle_sequence": meta.get("vehicle_sequence"),
                "infrastructure_sequence": meta.get("infrastructure_sequence"),
                "infra_image_timestamp": meta.get("infra_image_timestamp"),
                "ego_image_timestamp": meta.get("ego_image_timestamp"),
                "infra_path": str(infra_img),
                "ego_path": str(ego_img),
                "seen_count": len(seen),
                "hidden_count": len(hidden),
                "seen_boxes": seen,
                "hidden_boxes": hidden,
            }

            # JSONL
            fj.write(json.dumps(record, ensure_ascii=False) + "\n")

            # CSV（ネストはJSON文字列化）
            writer.writerow({
                "pair_index": i,
                "vehicle_frame": veh_frame,
                "infrastructure_frame": inf_frame,
                "vehicle_sequence": meta.get("vehicle_sequence"),
                "infrastructure_sequence": meta.get("infrastructure_sequence"),
                "infra_image_timestamp": meta.get("infra_image_timestamp"),
                "ego_image_timestamp": meta.get("ego_image_timestamp"),
                "infra_path": str(infra_img),
                "ego_path": str(ego_img),
                "seen_count": len(seen),
                "hidden_count": len(hidden),
                "seen_boxes_json": json.dumps(seen, ensure_ascii=False),
                "hidden_boxes_json": json.dumps(hidden, ensure_ascii=False),
            })

            wrote += 1
            if wrote % 500 == 0:
                print(f"wrote {wrote} records...")

    print("Done.")
    print("Wrote:", wrote)
    print("Skipped:", skipped)
    print("JSONL:", OUT_JSONL)
    print("CSV:  ", OUT_CSV)

if __name__ == "__main__":
    main()
