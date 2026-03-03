from __future__ import annotations
from pathlib import Path
import json
import argparse
import textwrap

import cv2

from src.spd_pairs import get_one_synced_pair

BASE_IMAGES = Path(r"D:\V2X\Public-V2X-Datasets\V2X-Seq (CVPR2023)\Sequential-Perception-Dataset\Full Dataset (train & val)")
BASE_SPD = Path(r"D:\V2X\Public-V2X-Datasets\V2X-Seq (CVPR2023)\Sequential-Perception-Dataset\Full Dataset (train & val)\V2X-Seq-SPD\V2X-Seq-SPD")

OUT_DIR = Path(r"D:\V2X\pair_gallery_full")
FRAMES_DIR = OUT_DIR / "frames"
META_DIR = OUT_DIR / "frames_meta"

FRAMES_DIR.mkdir(parents=True, exist_ok=True)
META_DIR.mkdir(parents=True, exist_ok=True)

def resize_keep_aspect(img, h: int):
    hh, ww = img.shape[:2]
    if hh == h:
        return img
    scale = h / hh
    return cv2.resize(img, (max(1, int(ww * scale)), h), interpolation=cv2.INTER_AREA)

def safe_get(meta: dict, key: str, default=""):
    v = meta.get(key, default)
    return "" if v is None else v

def draw_header(frame, lines, pad=10):
    """
    上部に黒帯を作ってテキストを描画（見やすい）
    """
    h, w = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55
    thickness = 1

    # 行高ざっくり計算
    (tw, th), _ = cv2.getTextSize("Ag", font, font_scale, thickness)
    line_h = th + 8
    header_h = pad * 2 + line_h * len(lines)

    header = frame.copy()
    # 黒帯
    cv2.rectangle(header, (0, 0), (w, header_h), (0, 0, 0), -1)

    y = pad + th + 2
    for s in lines:
        cv2.putText(header, s, (pad, y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
        y += line_h

    # header部分だけ置換
    header[0:header_h, 0:w] = header[0:header_h, 0:w]
    return header

def build_meta_lines(i: int, meta: dict, max_width_chars: int = 110):
    # よく使うキーを優先表示（存在すれば）
    infra_f = safe_get(meta, "infrastructure_frame")
    ego_f = safe_get(meta, "vehicle_frame")

    inf_ts = safe_get(meta, "inf_pointcloud_timestamp", safe_get(meta, "inf_timestamp", ""))
    veh_ts = safe_get(meta, "veh_pointcloud_timestamp", safe_get(meta, "veh_timestamp", ""))
    sys_off = safe_get(meta, "system_error_offset", "")

    line1 = f"pair={i:06d} | infra_frame={infra_f} | ego_frame={ego_f}"

    extras = []
    if inf_ts != "" or veh_ts != "":
        extras.append(f"ts(inf)={inf_ts}  ts(veh)={veh_ts}")
    if sys_off != "":
        extras.append(f"system_error_offset={sys_off}")

    line2 = " | ".join(extras) if extras else ""

    lines = [line1]
    if line2:
        # 文字が長すぎたら折り返し
        wrapped = textwrap.wrap(line2, width=max_width_chars)
        lines.extend(wrapped)

    return lines

def main(
    start: int = 0,
    end: int | None = None,
    resize_h: int = 540,
    jpeg_quality: int = 92,
    draw_meta: bool = True,
    save_meta_json: bool = True,
    skip_existing: bool = True,
):
    coop_path = BASE_SPD / "cooperative" / "data_info.json"
    coop = json.loads(coop_path.read_text(encoding="utf-8"))
    total = len(coop)

    if end is None or end > total:
        end = total
    if start < 0:
        start = 0

    print(f"Total pairs: {total} | generating [{start}, {end}) into {FRAMES_DIR}")

    for i in range(start, end):
        out_path = FRAMES_DIR / f"{i:06d}.jpg"
        meta_path = META_DIR / f"{i:06d}.json"

        if skip_existing and out_path.exists():
            # すでに作ってあれば飛ばす（再開が楽）
            continue

        infra_img, ego_img, meta = get_one_synced_pair(
            base_spd_dir=BASE_SPD,
            base_images_dir=BASE_IMAGES,
            pair_index=i,
        )

        infra = cv2.imread(str(infra_img))
        ego = cv2.imread(str(ego_img))
        if infra is None or ego is None:
            print(f"[WARN] skip {i}: read failed | infra={infra_img} ego={ego_img}")
            continue

        infra_r = resize_keep_aspect(infra, resize_h)
        ego_r = resize_keep_aspect(ego, resize_h)
        frame = cv2.hconcat([infra_r, ego_r])

        if draw_meta:
            lines = build_meta_lines(i, meta)
            frame = draw_header(frame, lines)

        ok = cv2.imwrite(
            str(out_path),
            frame,
            [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)]
        )
        if not ok:
            print(f"[WARN] failed to write: {out_path}")

        if save_meta_json:
            meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

        if (i - start) % 200 == 0:
            print(f"wrote {out_path.name}")

    print("Done!")
    print(f"Frames dir: {FRAMES_DIR}")
    print(f"Meta dir:   {META_DIR}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=-1, help="end index (exclusive). -1 means all.")
    ap.add_argument("--resize_h", type=int, default=540)
    ap.add_argument("--jpeg_quality", type=int, default=92)
    ap.add_argument("--no-draw-meta", action="store_true")
    ap.add_argument("--no-save-meta-json", action="store_true")
    ap.add_argument("--no-skip-existing", action="store_true")
    args = ap.parse_args()

    end = None if args.end == -1 else args.end

    main(
        start=args.start,
        end=end,
        resize_h=args.resize_h,
        jpeg_quality=args.jpeg_quality,
        draw_meta=(not args.no_draw_meta),
        save_meta_json=(not args.no_save_meta_json),
        skip_existing=(not args.no_skip_existing),
    )
