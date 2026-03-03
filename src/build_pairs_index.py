from __future__ import annotations
from pathlib import Path
import math
import html
import json

PAIRS_ROOT_DEFAULT = Path(r"D:\V2X\pairs_all")

def _read_text_safe(p: Path) -> str:
    try:
        return p.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return p.read_text(encoding="utf-8-sig")

def build_paged_gallery(
    pairs_root: Path = PAIRS_ROOT_DEFAULT,
    per_page: int = 200,
    show_meta: bool = False,
):
    pairs = sorted([p for p in pairs_root.iterdir() if p.is_dir() and p.name.startswith("pair_")])
    total = len(pairs)
    if total == 0:
        raise RuntimeError(f"No pair_* folders found under: {pairs_root}")

    pages = math.ceil(total / per_page)

    # 個別ページ生成
    for pi in range(pages):
        start = pi * per_page
        end = min((pi + 1) * per_page, total)
        lines = [
            "<!doctype html>",
            "<html><head><meta charset='utf-8'>",
            "<title>V2X Pairs Gallery</title>",
            "<style>",
            "body{font-family:Arial;margin:20px}",
            ".nav{margin:10px 0}",
            ".card{margin:18px 0; padding:12px; border:1px solid #ddd; border-radius:8px}",
            ".row{display:flex; gap:12px; align-items:flex-start; flex-wrap:wrap}",
            ".imgbox{flex:1; min-width:360px}",
            "img{max-width:100%; height:auto; border:1px solid #ccc}",
            "pre{white-space:pre-wrap; word-break:break-word; background:#f7f7f7; padding:10px; border-radius:6px}",
            "a{color:#0b66c3; text-decoration:none}",
            "</style>",
            "</head><body>",
            f"<h1>V2X pairs (paged)</h1>",
            f"<p>Total pairs: {total} | per page: {per_page} | page {pi+1}/{pages}</p>",
            "<div class='nav'>",
        ]
        # nav links
        if pi > 0:
            lines.append(f"<a href='index_{pi-1:03d}.html'>&laquo; prev</a> ")
        lines.append(f"<a href='index.html'>master</a> ")
        if pi < pages - 1:
            lines.append(f"<a href='index_{pi+1:03d}.html'>next &raquo;</a>")
        lines.append("</div>")

        for p in pairs[start:end]:
            infra = p / "infra.jpg"
            ego = p / "ego.jpg"
            merged = p / "merged.jpg"
            meta = p / "meta.json"

            # 画像は merged があればそれを優先。なければ infra/ego を並べる
            lines.append("<div class='card'>")
            lines.append(f"<h3>{html.escape(p.name)}</h3>")
            lines.append("<div class='row'>")

            if merged.exists():
                lines.append("<div class='imgbox'>")
                lines.append(f"<div><b>merged</b></div>")
                lines.append(f"<img src='{html.escape(merged.name)}' loading='lazy'>")
                lines.append("</div>")
            else:
                if infra.exists():
                    lines.append("<div class='imgbox'>")
                    lines.append(f"<div><b>infra</b></div>")
                    lines.append(f"<img src='{html.escape(infra.name)}' loading='lazy'>")
                    lines.append("</div>")
                if ego.exists():
                    lines.append("<div class='imgbox'>")
                    lines.append(f"<div><b>ego</b></div>")
                    lines.append(f"<img src='{html.escape(ego.name)}' loading='lazy'>")
                    lines.append("</div>")

            lines.append("</div>")  # row

            if show_meta and meta.exists():
                try:
                    meta_obj = json.loads(_read_text_safe(meta))
                    meta_str = json.dumps(meta_obj, ensure_ascii=False, indent=2)
                except Exception:
                    meta_str = _read_text_safe(meta)
                lines.append("<details><summary>meta</summary>")
                lines.append(f"<pre>{html.escape(meta_str)}</pre>")
                lines.append("</details>")

            lines.append("</div>")  # card

        lines.append("</body></html>")
        (pairs_root / f"index_{pi:03d}.html").write_text("\n".join(lines), encoding="utf-8")

    # master index
    master = [
        "<!doctype html>",
        "<html><head><meta charset='utf-8'>",
        "<title>V2X Pairs Master</title>",
        "<style>body{font-family:Arial;margin:20px} a{display:inline-block;margin:4px 0;color:#0b66c3;text-decoration:none}</style>",
        "</head><body>",
        "<h1>V2X pairs master index</h1>",
        f"<p>Total pairs: {total} | Pages: {pages} | per page: {per_page}</p>",
    ]
    for i in range(pages):
        s = i * per_page
        e = min((i + 1) * per_page, total) - 1
        master.append(f"<a href='index_{i:03d}.html'>page {i:03d} (pairs {s}–{e})</a><br>")
    master.append("</body></html>")
    (pairs_root / "index.html").write_text("\n".join(master), encoding="utf-8")

    print("Done!")
    print(f"Open: {pairs_root / 'index.html'}")

def main():
    # meta表示したければ True に
    build_paged_gallery(pairs_root=PAIRS_ROOT_DEFAULT, per_page=200, show_meta=False)

if __name__ == "__main__":
    main()
