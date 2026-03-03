from pathlib import Path
import json
import cv2

BASE = Path(r"D:\V2X\Public-V2X-Datasets\V2X-Seq (CVPR2023)\Sequential-Perception-Dataset\Full Dataset (train & val)\V2X-Seq-SPD\V2X-Seq-SPD")

PAIR_ROOT = Path(r"D:\V2X\pair_V2X")
INFRA_DIR = PAIR_ROOT / "infra"
META_DIR = PAIR_ROOT / "meta"
OUT_DIR = PAIR_ROOT / "infra_coop_bbox_samples"

OUT_DIR.mkdir(parents=True, exist_ok=True)

N = 20  # サンプル枚数

def draw_one(pair_idx):
    meta = json.loads((META_DIR / f"pair_{pair_idx:06d}.json").read_text(encoding="utf-8"))

    veh_frame = meta["vehicle_frame"]
    inf_frame = meta["infrastructure_frame"]

    coop_path = BASE / "cooperative" / "label" / f"{veh_frame}.json"
    inf_label_path = BASE / "infrastructure-side" / "label" / "camera" / f"{inf_frame}.json"

    coop = json.loads(coop_path.read_text(encoding="utf-8"))
    inf_label = json.loads(inf_label_path.read_text(encoding="utf-8"))

    inf_by_token = {x.get("token"): x for x in inf_label}

    inf_tokens = [x.get("inf_token") for x in coop
                  if x.get("from_side")=="coop" and x.get("inf_token") not in (None,-1)]

    img = cv2.imread(str(INFRA_DIR / f"pair_{pair_idx:06d}.jpg"))
    if img is None:
        return 0

    count = 0
    for t in inf_tokens:
        obj = inf_by_token.get(t)
        if not obj:
            continue
        b = obj.get("2d_box")
        if isinstance(b, dict):
            xmin,ymin,xmax,ymax = map(int,(b["xmin"],b["ymin"],b["xmax"],b["ymax"]))
            cv2.rectangle(img,(xmin,ymin),(xmax,ymax),(0,0,255),3)
            cv2.putText(img,"COOP",(xmin,max(0,ymin-6)),
                        cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,255),2)
            count += 1

    out = OUT_DIR / f"pair_{pair_idx:06d}.jpg"
    cv2.imwrite(str(out), img, [int(cv2.IMWRITE_JPEG_QUALITY),92])
    return count


for i in range(N):
    c = draw_one(i)
    print(f"pair {i:06d}: coop boxes = {c}")

print("Done. Samples in:", OUT_DIR)
