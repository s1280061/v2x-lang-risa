from pathlib import Path
import json
import cv2

BASE = Path(r"D:\V2X\Public-V2X-Datasets\V2X-Seq (CVPR2023)\Sequential-Perception-Dataset\Full Dataset (train & val)\V2X-Seq-SPD\V2X-Seq-SPD")

# ペア0に対応（君の meta から）
veh_frame = "000009"
inf_frame = "000000"

# 入力（infra画像は、君の整理済みフォルダを使う）
infra_img_path = Path(r"D:\V2X\pair_V2X\infra\pair_000001.jpg")

# 出力
out_dir = Path(r"D:\V2X\pair_V2X\infra_coop_bbox")
out_path = out_dir / "pair_000001.jpg"

coop_path = BASE / "cooperative" / "label" / f"{veh_frame}.json"
inf_label_path = BASE / "infrastructure-side" / "label" / "camera" / f"{inf_frame}.json"

coop = json.loads(coop_path.read_text(encoding="utf-8"))
inf_label = json.loads(inf_label_path.read_text(encoding="utf-8"))

inf_by_token = {x.get("token"): x for x in inf_label}

inf_tokens = [x.get("inf_token") for x in coop
              if x.get("from_side") == "coop" and x.get("inf_token") not in (None, -1)]

img = cv2.imread(str(infra_img_path))
if img is None:
    raise SystemExit(f"cannot read: {infra_img_path}")

for t in inf_tokens:
    obj = inf_by_token.get(t)
    if not obj:
        continue
    b = obj.get("2d_box")
    if isinstance(b, dict) and all(k in b for k in ("xmin","ymin","xmax","ymax")):
        xmin, ymin, xmax, ymax = map(int, (b["xmin"], b["ymin"], b["xmax"], b["ymax"]))
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 0, 255), 3)
        cv2.putText(img, "COOP", (xmin, max(0, ymin - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)

out_dir.mkdir(parents=True, exist_ok=True)
cv2.imwrite(str(out_path), img, [int(cv2.IMWRITE_JPEG_QUALITY), 92])

print("wrote", out_path)
print("num coop boxes:", len(inf_tokens))
