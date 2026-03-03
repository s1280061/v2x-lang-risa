from pathlib import Path
from typing import Tuple
import html
import json

from PIL import Image, ImageOps

from src.spd_pairs import get_one_synced_pair

def _open_rgb(path: Path) -> Image.Image:
    im = Image.open(path)
    im = ImageOps.exif_transpose(im)
    return im.convert("RGB")

def make_side_by_side(infra_path: Path, ego_path: Path, target_height: int = 512, gap: int = 12) -> Image.Image:
    infra = _open_rgb(infra_path)
    ego = _open_rgb(ego_path)

    def resize_h(im: Image.Image) -> Image.Image:
        w, h = im.size
        new_w = int(w * (target_height / h))
        return im.resize((new_w, target_height))

    infra = resize_h(infra)
    ego = resize_h(ego)

    W = infra.size[0] + gap + ego.size[0]
    H = target_height
    canvas = Image.new("RGB", (W, H), (255, 255, 255))
    canvas.paste(infra, (0, 0))
    canvas.paste(ego, (infra.size[0] + gap, 0))
    return canvas

def main():
    base_images = Path(r"D:\V2X\Public-V2X-Datasets\V2X-Seq (CVPR2023)\Sequential-Perception-Dataset\Full Dataset (train & val)")
    base_spd = Path(r"D:\V2X\Public-V2X-Datasets\V2X-Seq (CVPR2023)\Sequential-Perception-Dataset\Full Dataset (train & val)\V2X-Seq-SPD\V2X-Seq-SPD")

    out_dir = Path("outputs/pairs")
    out_dir.mkdir(parents=True, exist_ok=True)

    N = 30          # 何ペア作るか（増やしてOK）
    height = 512    # 見やすい高さ

    rows = []
    for i in range(N):
        infra_img, ego_img, meta = get_one_synced_pair(base_spd_dir=base_spd, base_images_dir=base_images, pair_index=i)

        merged = make_side_by_side(infra_img, ego_img, target_height=height)
        out_name = f"pair_{i:04d}_infra{meta['infrastructure_frame']}_ego{meta['vehicle_frame']}.jpg"
        out_path = out_dir / out_name
        merged.save(out_path, quality=92)

        rows.append({
            "index": i,
            "out": out_name,
            "infra": str(infra_img),
            "ego": str(ego_img),
            "meta": meta,
        })
        print(f"[{i:04d}] saved {out_path}")

    # JSON保存（あとで検索/分析に便利）
    (out_dir / "index.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    # HTMLギャラリー作成（ダブルクリックで見れる）
    html_lines = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        "<title>V2X-Seq SPD Synced Pairs</title>",
        "<style>body{font-family:Arial;margin:20px} .card{margin:18px 0} img{max-width:100%;height:auto;border:1px solid #ddd}</style>",
        "</head><body>",
        "<h1>V2X-Seq SPD Synced Pairs (infra | ego)</h1>",
        "<p>Generated under outputs/pairs/</p>",
    ]
    for r in rows:
        meta = html.escape(json.dumps(r["meta"], ensure_ascii=False))
        html_lines += [
            "<div class='card'>",
            f"<h3>pair {r['index']:04d} — infra_frame={r['meta'].get('infrastructure_frame')} ego_frame={r['meta'].get('vehicle_frame')}</h3>",
            f"<div><img src='{html.escape(r['out'])}'></div>",
            f"<details><summary>meta</summary><pre>{meta}</pre></details>",
            "</div>",
        ]
    html_lines += ["</body></html>"]
    (out_dir / "index.html").write_text("\n".join(html_lines), encoding="utf-8")

    print("\nDone!")
    print(f"Open: {out_dir / 'index.html'}")

if __name__ == "__main__":
    main()
