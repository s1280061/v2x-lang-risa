from pathlib import Path
import json
import cv2

BASE = Path(r"D:\V2X\Public-V2X-Datasets\V2X-Seq (CVPR2023)\Sequential-Perception-Dataset\Full Dataset (train & val)\V2X-Seq-SPD\V2X-Seq-SPD")

PAIR_ROOT = Path(r"D:\V2X\pair_V2X")
INFRA_DIR = PAIR_ROOT / "infra"
META_DIR  = PAIR_ROOT / "meta"
OUT_DIR   = PAIR_ROOT / "infra_coop_bbox_samples_nonzero"

OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET = 20          # 欲しいサンプル枚数
MAX_SCAN = 10761     # 全ペア数（必要ならもっと大きくしてOK）

def load_json(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))

def draw_one(pair_idx: int) -> int:
    meta = load_json(META_DIR / f"pair_{pair_idx:06d}.json")
    veh_frame = meta["vehicle_frame"]
    inf_frame = meta["infrastructure_frame"]

    coop_path = BASE / "cooperative" / "label" / f"{veh_frame}.json"
    inf_label_path = BASE / "infrastructure-side" / "label" / "camera" / f"{inf_frame}.json"

    coop = load_json(coop_path)
    inf_label = load_json(inf_label_path)

    # ★ tokenじゃなく track_id で対応付けする
    inf_ids = [x.get("inf_track_id") for x in coop
               if x.get("from_side") == "coop" and x.get("inf_track_id") not in (None, "-1", -1)]

    if not inf_ids:
        return 0

    infra_img_path = INFRA_DIR / f"pair_{pair_idx:06d}.jpg"
    img = cv2.imread(str(infra_img_path))
    if img is None:
        return 0

    # infra track_id -> object
    inf_by_track = {x.get("track_id"): x for x in inf_label}

    count = 0
    for tid in inf_ids:
        obj = inf_by_track.get(tid)
        if not obj:
            continue
        b = obj.get("2d_box")
        if isinstance(b, dict) and all(k in b for k in ("xmin","ymin","xmax","ymax")):
            xmin,ymin,xmax,ymax = map(int, (b["xmin"],b["ymin"],b["xmax"],b["ymax"]))
            cv2.rectangle(img, (xmin,ymin), (xmax,ymax), (0,0,255), 3)
            cv2.putText(img, "COOP", (xmin, max(0, ymin-6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2, cv2.LINE_AA)
            count += 1

    if count == 0:
        return 0

    out = OUT_DIR / f"pair_{pair_idx:06d}_coop{count}.jpg"
    cv2.imwrite(str(out), img, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    return count


saved = 0
for i in range(MAX_SCAN):
    c = draw_one(i)
    if c > 0:
        print(f"SAVED pair {i:06d}: coop boxes = {c}")
        saved += 1
        if saved >= TARGET:
            break

print("Done.")
print("Saved:", saved)
print("Output:", OUT_DIR)
