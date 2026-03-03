import json
from pathlib import Path
from typing import Optional

from src.spd_pairs import get_one_synced_pair, _read_json  # _read_jsonは同ファイル内関数

def hardlink_or_copy(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        # ハードリンク（同一ドライブなら超省容量）
        dst.hardlink_to(src)
    except Exception:
        # ダメならコピー（最悪でも動く）
        dst.write_bytes(src.read_bytes())

def main():
    base_images = Path(r"D:\V2X\Public-V2X-Datasets\V2X-Seq (CVPR2023)\Sequential-Perception-Dataset\Full Dataset (train & val)")
    base_spd = Path(r"D:\V2X\Public-V2X-Datasets\V2X-Seq (CVPR2023)\Sequential-Perception-Dataset\Full Dataset (train & val)\V2X-Seq-SPD\V2X-Seq-SPD")

    out_root = Path(r"D:\V2X\pairs_all")
    out_root.mkdir(parents=True, exist_ok=True)

    # cooperative/data_info.json のペア総数
    coop = json.loads((base_spd / "cooperative" / "data_info.json").read_text(encoding="utf-8"))
    total = len(coop)
    print(f"Total pairs in cooperative/data_info.json = {total}")

    # 全部やると時間かかるので、まずはここを小さくして動作確認→増やすのがおすすめ
    LIMIT: Optional[int] = None  # None にすると全件
    n = total if LIMIT is None else min(LIMIT, total)

    for i in range(n):
        infra_img, ego_img, meta = get_one_synced_pair(base_spd_dir=base_spd, base_images_dir=base_images, pair_index=i)

        pair_dir = out_root / f"pair_{i:06d}"
        pair_dir.mkdir(parents=True, exist_ok=True)

        # 見やすい固定名にする
        hardlink_or_copy(infra_img, pair_dir / "infra.jpg")
        hardlink_or_copy(ego_img, pair_dir / "ego.jpg")

        (pair_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

        if i % 50 == 0:
            print(f"[{i:06d}] done -> {pair_dir}")

    print("Done.")

if __name__ == "__main__":
    main()
