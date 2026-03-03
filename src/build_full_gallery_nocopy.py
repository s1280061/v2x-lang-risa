from __future__ import annotations
from pathlib import Path
import math
import html
import json
from urllib.parse import quote

from src.spd_pairs import get_one_synced_pair

BASE_IMAGES = Path(r"D:\V2X\Public-V2X-Datasets\V2X-Seq (CVPR2023)\Sequential-Perception-Dataset\Full Dataset (train & val)")
BASE_SPD = Path(r"D:\V2X\Public-V2X-Datasets\V2X-Seq (CVPR2023)\Sequential-Perception-Dataset\Full Dataset (train & val)\V2X-Seq-SPD\V2X-Seq-SPD")

OUT_DIR = Path(r"D:\V2X\pair_gallery_full")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def file_url(p: Path) -> str:
    # Windows path -> file:/// URL（スペース等も安全に）
    s = p.resolve().as_posix()
    return "file:///" + quote(s)

def main(per_page: int = 200, show_meta: bool = False):
    coop_path = BASE_SPD / "cooperative" / "data_info.json"
    coop = json.loads(coop_path.read_text(encoding="utf-8"))
    total = len(coop)
    pages = math.ceil(total / per_page)

    # 個別ページ
    for pi in range(pages):
        start = pi * per_page
        end = min((pi + 1) * per_page, total)

        lines = [
            "<!doctype html>",
            "<html><head><meta charset='utf-8'>",
            "<title>V2X-Seq SPD Synced Pairs (no-copy)</title>",
            "<style>",
            "body{font-family:Arial;margin:20px}",
            ".nav{margin:10px 0}",
            ".card{margin:18px 0;padding:12px;border:1px solid #ddd;border-radius:8px}",
            ".row{display:flex;gap:12px;flex-wrap:wrap}",
            ".imgbox{flex:1;min-width:360px}",
            "img{max-width:100%;height:auto;border:1px solid #ccc}",
            "pre{white-space:pre-wrap;word-break:break-word;background:#f7f7f7;padding:10px;border-radius:6px}",
            "a{color:#0b66c3;text-decoration:none}",
            "</style>",
            "</head><body>",
            f"<h1>V2X-Seq SPD Synced Pairs (no-copy)</h1>",
            f"<p>Total: {total} | per page: {per_page} | page {pi+1}/{pages}</p>",
            "<div class='nav'>",
        ]
        if pi > 0:
            lines.append(f"<a href='index_{pi-1:03d}.html'>&laquo; prev</a> ")
        lines.append("<a href='index.html'>master</a> ")
        if pi < pages - 1:
            lines.append(f"<a href='index_{pi+1:03d}.html'>next &raquo;</a>")
        lines.append("</div>")

        for i in range(start, end):
            infra_img, ego_img, meta = get_one_synced_pair(
                base_spd_dir=BASE_SPD,
                base_images_dir=BASE_IMAGES,
                pair_index=i,
            )

            lines.append("<div class='card'>")
            lines.append(f"<h3>pair {i:05d} — infra={html.escape(str(meta.get('infrastructure_frame')))} ego={html.escape(str(meta.get('vehicle_frame')))}</h3>")
            lines.append("<div class='row'>")

            lines.append("<div class='imgbox'>")
            lines.append("<div><b>infra</b></div>")
            lines.append(f"<img src='{html.escape(file_url(infra_img))}' loading='lazy'>")
            lines.append("</div>")

            lines.append("<div class='imgbox'>")
            lines.append("<div><b>ego</b></div>")
            lines.append(f"<img src='{html.escape(file_url(ego_img))}' loading='lazy'>")
            lines.append("</div>")

            lines.append("</div>")  # row

            if show_meta:
                meta_str = json.dumps(meta, ensure_ascii=False, indent=2)
                lines.append("<details><summary>meta</summary>")
                lines.append(f"<pre>{html.escape(meta_str)}</pre>")
                lines.append("</details>")

            lines.append("</div>")  # card

        lines.append("</body></html>")
        (OUT_DIR / f"index_{pi:03d}.html").write_text("\n".join(lines), encoding="utf-8")
        print(f"wrote index_{pi:03d}.html  ({start}-{end-1})")

    # master
    master = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        "<title>V2X-Seq SPD Master</title>",
        "<style>body{font-family:Arial;margin:20px} a{display:inline-block;margin:4px 0;color:#0b66c3;text-decoration:none}</style>",
        "</head><body>",
        "<h1>Master index</h1>",
        f"<p>Total: {total} | Pages: {pages} | per page: {per_page}</p>",
    ]
    for pi in range(pages):
        s = pi * per_page
        e = min((pi + 1) * per_page, total) - 1
        master.append(f"<a href='index_{pi:03d}.html'>page {pi:03d} (pairs {s}–{e})</a><br>")
    master.append("</body></html>")
    (OUT_DIR / "index.html").write_text("\n".join(master), encoding="utf-8")

    print("Done!")
    print(f"Open: {OUT_DIR / 'index.html'}")

if __name__ == "__main__":
    main(per_page=200, show_meta=False)