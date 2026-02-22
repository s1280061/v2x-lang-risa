from pathlib import Path
from src.infra_summary import summarize_infra_image
from src.ego_advice import generate_ego_advice

def normalize_dataset_dir(d: Path) -> Path:
    inner = d / d.name
    return inner if inner.exists() and inner.is_dir() else d

def pick_first_image(folder: Path) -> Path:
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        hit = next(folder.rglob(ext), None)
        if hit is not None:
            return hit
    raise FileNotFoundError(f"No images found under: {folder}")

def main():
    base = Path(r"D:\V2X\Public-V2X-Datasets\V2X-Seq (CVPR2023)\Sequential-Perception-Dataset\Full Dataset (train & val)")

    infra_dir = normalize_dataset_dir(base / "V2X-Seq-SPD-infrastructure-side-image")
    ego_dir   = normalize_dataset_dir(base / "V2X-Seq-SPD-vehicle-side-image")

    infra_first = pick_first_image(infra_dir)
    ego_first   = pick_first_image(ego_dir)

    summary = summarize_infra_image(infra_first)
    advice  = generate_ego_advice(ego_first, summary)

    print("=== INFRA IMAGE ===")
    print(infra_first)
    print("\n=== EGO IMAGE ===")
    print(ego_first)
    print("\n=== INFRA SUMMARY ===")
    print(summary)
    print("\n=== EGO ADVICE ===")
    print(advice)

if __name__ == "__main__":
    main()
