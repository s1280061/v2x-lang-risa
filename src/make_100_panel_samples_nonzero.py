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

TARGET = 100
MAX_SCAN = 10761


def load_json(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))


def draw_infra_boxes(infra_img, inf_label, coop):
    """
    blue = ego-visible (coop)
    red  = ego-invisible (inf-only)
    """

    # coop 側で対応が取れた infra track_id
    coop_inf_ids = {
        x.get("inf_track_id") for x in coop
        if x.get("from_side") == "coop" and x.get("inf_track_id") not in (None, "-1", -1)
    }

    if len(coop_inf_ids) == 0:
        return 0, 0

    coop_cnt = 0
    inf_only_cnt = 0

    for obj in inf_label:
        tid = obj.get("track_id")
        b = obj.get("2d_box")
        if not isinstance(b, dict):
            continue
        if not all(k in b for k in ("xmin","ymin","xmax","ymax")):
            continue

        xmin,ymin,xmax,ymax = map(int,(b["xmin"],b["ymin"],b["xmax"],b["ymax"]))

        # ego-visible (coop) → BLUE
        if tid in coop_inf_ids:
            cv2.rectangle(infra_img,(xmin,ymin),(xmax,ymax),(255,0,0),3)
            cv2.putText(infra_img,"EGO-SEEN",(xmin,max(0,ymin-6)),
                        cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,0,0),2,cv2.LINE_AA)
            coop_cnt += 1

        # ego-invisible (infra only) → RED
        else:
            cv2.rectangle(infra_img,(xmin,ymin),(xmax,ymax),(0,0,255),2)
            cv2.putText(infra_img,"INF-ONLY",(xmin,max(0,ymin-6)),
                        cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,255),2,cv2.LINE_AA)
            inf_only_cnt += 1

    return coop_cnt, inf_only_cnt


def make_panel(pair_idx: int) -> int:
    meta = load_json(META_DIR / f"pair_{pair_idx:06d}.json")
    veh_frame = meta["vehicle_frame"]
    inf_frame = meta["infrastructure_frame"]

    coop_path = BASE / "cooperative" / "label" / f"{veh_frame}.json"
    inf_label_path = BASE / "infrastructure-side" / "label" / "camera" / f"{inf_frame}.json"

    coop = load_json(coop_path)
    inf_label = load_json(inf_label_path)

    infra = cv2.imread(str(INFRA_DIR / f"pair_{pair_idx:06d}.jpg"))
    ego   = cv2.imread(str(EGO_DIR   / f"pair_{pair_idx:06d}.jpg"))

    if infra is None or ego is None:
        return 0

    coop_cnt, inf_only_cnt = draw_infra_boxes(infra, inf_label, coop)

    # coop が1つも無いフレームはスキップ
    if coop_cnt == 0:
        return 0

    # panel 作成
    h = 540
    def resize_keep(img):
        H,W = img.shape[:2]
        s = h / H
        return cv2.resize(img,(int(W*s),h),interpolation=cv2.INTER_AREA)

    infra_r = resize_keep(infra)
    ego_r   = resize_keep(ego)

    panel = cv2.hconcat([infra_r, ego_r])

    title = f"pair {pair_idx:06d} | BLUE: ego-seen={coop_cnt} | RED: infra-only={inf_only_cnt}"
    cv2.putText(panel,title,(20,40),
                cv2.FONT_HERSHEY_SIMPLEX,0.9,(255,255,255),3,cv2.LINE_AA)
    cv2.putText(panel,title,(20,40),
                cv2.FONT_HERSHEY_SIMPLEX,0.9,(0,0,0),1,cv2.LINE_AA)

    out = OUT_DIR / f"pair_{pair_idx:06d}_seen{coop_cnt}_hidden{inf_only_cnt}.jpg"
    cv2.imwrite(str(out),panel,[int(cv2.IMWRITE_JPEG_QUALITY),92])

    return coop_cnt


saved = 0
for i in range(MAX_SCAN):
    c = make_panel(i)
    if c > 0:
        print(f"SAVED pair {i:06d} | ego-seen={c}")
        saved += 1
        if saved >= TARGET:
            break

print("Done.")
print("Saved:", saved)
print("Output:", OUT_DIR)