from pathlib import Path
import json
import cv2

BASE = Path(r"D:\V2X\Public-V2X-Datasets\V2X-Seq (CVPR2023)\Sequential-Perception-Dataset\Full Dataset (train & val)\V2X-Seq-SPD\V2X-Seq-SPD")

PAIR_ROOT = Path(r"D:\V2X\pair_V2X")
INFRA_DIR = PAIR_ROOT / "infra"
EGO_DIR   = PAIR_ROOT / "ego"
META_DIR  = PAIR_ROOT / "meta"

OUT_DIR = PAIR_ROOT / "pair_panels_samples_nonzero"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET = 20
MAX_SCAN = 10761

def load_json(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))

def draw_infra_coop(infra_img, inf_label, coop) -> int:
    # track_idで対応付け
    inf_ids = [x.get("inf_track_id") for x in coop
               if x.get("from_side") == "coop" and x.get("inf_track_id") not in (None, "-1", -1)]
    if not inf_ids:
        return 0

    inf_by_track = {x.get("track_id"): x for x in inf_label}
    count = 0
    for tid in inf_ids:
        obj = inf_by_track.get(tid)
        if not obj:
            continue
        b = obj.get("2d_box")
        if isinstance(b, dict) and all(k in b for k in ("xmin","ymin","xmax","ymax")):
            xmin,ymin,xmax,ymax = map(int, (b["xmin"],b["ymin"],b["xmax"],b["ymax"]))
            cv2.rectangle(infra_img, (xmin,ymin), (xmax,ymax), (0,0,255), 3)
            cv2.putText(infra_img, "COOP", (xmin, max(0, ymin-6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2, cv2.LINE_AA)
            count += 1
    return count

def make_panel(pair_idx: int) -> int:
    meta = load_json(META_DIR / f"pair_{pair_idx:06d}.json")
    veh_frame = meta["vehicle_frame"]
    inf_frame = meta["infrastructure_frame"]

    coop_path = BASE / "cooperative" / "label" / f"{veh_frame}.json"
    inf_label_path = BASE / "infrastructure-side" / "label" / "camera" / f"{inf_frame}.json"

    coop = load_json(coop_path)
    inf_label = load_json(inf_label_path)

    infra_path = INFRA_DIR / f"pair_{pair_idx:06d}.jpg"
    ego_path   = EGO_DIR   / f"pair_{pair_idx:06d}.jpg"

    infra = cv2.imread(str(infra_path))
    ego   = cv2.imread(str(ego_path))
    if infra is None or ego is None:
        return 0

    coop_count = draw_infra_coop(infra, inf_label, coop)
    if coop_count == 0:
        return 0

    # 高さを合わせて横連結
    h = 540
    def resize_keep(img, h):
        H, W = img.shape[:2]
        scale = h / H
        return cv2.resize(img, (int(W*scale), h), interpolation=cv2.INTER_AREA)

    infra_r = resize_keep(infra, h)
    ego_r   = resize_keep(ego, h)

    panel = cv2.hconcat([infra_r, ego_r])

    # 上に説明テキスト
    cv2.putText(panel, f"pair {pair_idx:06d} | infra(left) + COOP bbox | ego(right)", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 3, cv2.LINE_AA)
    cv2.putText(panel, f"pair {pair_idx:06d} | infra(left) + COOP bbox | ego(right)", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,0), 1, cv2.LINE_AA)

    out = OUT_DIR / f"pair_{pair_idx:06d}_coop{coop_count}_panel.jpg"
    cv2.imwrite(str(out), panel, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    return coop_count


saved = 0
for i in range(MAX_SCAN):
    c = make_panel(i)
    if c > 0:
        print(f"SAVED pair {i:06d}: coop boxes = {c}")
        saved += 1
        if saved >= TARGET:
            break

print("Done.")
print("Saved:", saved)
print("Output:", OUT_DIR)
