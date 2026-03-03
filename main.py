from pathlib import Path
from src.spd_pairs import get_one_synced_pair
from src.infra_summary import summarize_infra_image
from src.ego_advice import generate_ego_advice

def main():
    base_images = Path(r"D:\V2X\Public-V2X-Datasets\V2X-Seq (CVPR2023)\Sequential-Perception-Dataset\Full Dataset (train & val)")
    base_spd = Path(r"D:\V2X\Public-V2X-Datasets\V2X-Seq (CVPR2023)\Sequential-Perception-Dataset\Full Dataset (train & val)\V2X-Seq-SPD\V2X-Seq-SPD")

    infra_img, ego_img, meta = get_one_synced_pair(base_spd_dir=base_spd, base_images_dir=base_images, pair_index=0)

    print("=== SYNC META ===")
    print(meta)
    print("\\n=== INFRA IMAGE ===")
    print(infra_img)
    print("\\n=== EGO IMAGE ===")
    print(ego_img)

    summary = summarize_infra_image(infra_img)
    advice = generate_ego_advice(ego_img, summary)

    print("\\n=== INFRA SUMMARY ===")
    print(summary)
    print("\\n=== EGO ADVICE ===")
    print(advice)

if __name__ == "__main__":
    main()
