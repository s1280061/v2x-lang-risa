from __future__ import annotations
from pathlib import Path
import argparse
import csv
import json
import os
import shutil

from src.spd_pairs import get_one_synced_pair


BASE_IMAGES = Path(r"D:\V2X\Public-V2X-Datasets\V2X-Seq (CVPR2023)\Sequential-Perception-Dataset\Full Dataset (train & val)")
BASE_SPD = Path(r"D:\V2X\Public-V2X-Datasets\V2X-Seq (CVPR2023)\Sequential-Perception-Dataset\Full Dataset (train & val)\V2X-Seq-SPD\V2X-Seq-SPD")


def link_or_copy(src: Path, dst: Path, mode: str) -> None:
    """
    mode:
      - hardlink: try os.link (same drive recommended). fallback to copy if fails.
      - copy: shutil.copy2
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return

    if mode == "hardlink":
        try:
            os.link(str(src), str(dst))
            return
        except Exception:
            # fallback
            shutil.copy2(src, dst)
            return

    shutil.copy2(src, dst)


def safe_get(d: dict, key: str, default=""):
    v = d.get(key, default)
    return "" if v is None else v


def main(
    out_root: Path,
    start: int = 0,
    end: int | None = None,
    image_mode: str = "hardlink",
    write_meta_json: bool = True,
    skip_existing: bool = True,
):
    out_infra = out_root / "infra"
    out_ego = out_root / "ego"
    out_meta = out_root / "meta"
    out_root.mkdir(parents=True, exist_ok=True)
    out_infra.mkdir(parents=True, exist_ok=True)
    out_ego.mkdir(parents=True, exist_ok=True)
    out_meta.mkdir(parents=True, exist_ok=True)

    coop_path = BASE_SPD / "cooperative" / "data_info.json"
    coop = json.loads(coop_path.read_text(encoding="utf-8"))
    total = len(coop)

    if end is None or end > total:
        end = total
    if start < 0:
        start = 0

    csv_path = out_root / "pairs.csv"

    # CSV header（よく使う項目をまず固定で置く）
    fieldnames = [
        "pair_index",
        "vehicle_sequence",
        "infrastructure_sequence",
        "vehicle_frame",
        "infrastructure_frame",
        "veh_pointcloud_timestamp",
        "inf_pointcloud_timestamp",
        "system_error_offset",
        "infra_src_path",
        "ego_src_path",
        "infra_dst_path",
        "ego_dst_path",
        "meta_json_path",
    ]

    # 既存CSVがあっても、今回は「作り直し」前提（壊れにくい）
    # 追記にしたい場合は、mode="a" に変更してヘッダ判定を入れる。
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()

        print(f"Total pairs: {total} | exporting [{start}, {end})")
        print(f"Output root: {out_root}")
        print(f"Image mode: {image_mode} (hardlink recommended on same drive)")

        for i in range(start, end):
            infra_img, ego_img, meta = get_one_synced_pair(
                base_spd_dir=BASE_SPD,
                base_images_dir=BASE_IMAGES,
                pair_index=i,
            )

            infra_dst = out_infra / f"pair_{i:06d}.jpg"
            ego_dst = out_ego / f"pair_{i:06d}.jpg"
            meta_dst = out_meta / f"pair_{i:06d}.json"

            if skip_existing and infra_dst.exists() and ego_dst.exists() and (not write_meta_json or meta_dst.exists()):
                # CSVだけ再生成したいなら skip_existing=False にする
                pass
            else:
                link_or_copy(infra_img, infra_dst, image_mode)
                link_or_copy(ego_img, ego_dst, image_mode)
                if write_meta_json:
                    meta_dst.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

            row = {
                "pair_index": i,
                "vehicle_sequence": safe_get(meta, "vehicle_sequence"),
                "infrastructure_sequence": safe_get(meta, "infrastructure_sequence"),
                "vehicle_frame": safe_get(meta, "vehicle_frame"),
                "infrastructure_frame": safe_get(meta, "infrastructure_frame"),
                "veh_pointcloud_timestamp": safe_get(meta, "veh_pointcloud_timestamp"),
                "inf_pointcloud_timestamp": safe_get(meta, "inf_pointcloud_timestamp"),
                "system_error_offset": safe_get(meta, "system_error_offset"),
                "infra_src_path": str(infra_img),
                "ego_src_path": str(ego_img),
                "infra_dst_path": str(infra_dst),
                "ego_dst_path": str(ego_dst),
                "meta_json_path": str(meta_dst) if write_meta_json else "",
            }
            w.writerow(row)

            if i % 500 == 0:
                print(f"processed {i}/{end-1}")

    print("Done!")
    print(f"Wrote CSV: {csv_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_root", type=str, default=r"D:\V2X\pair_V2X")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=-1, help="exclusive end; -1 means all")
    ap.add_argument("--image_mode", type=str, default="hardlink", choices=["hardlink", "copy"])
    ap.add_argument("--no_meta_json", action="store_true")
    ap.add_argument("--no_skip_existing", action="store_true")
    args = ap.parse_args()

    end = None if args.end == -1 else args.end

    main(
        out_root=Path(args.out_root),
        start=args.start,
        end=end,
        image_mode=args.image_mode,
        write_meta_json=(not args.no_meta_json),
        skip_existing=(not args.no_skip_existing),
    )
