import json
from pathlib import Path
from typing import Tuple, Dict, Any, List

def _read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except UnicodeDecodeError:
        return json.loads(path.read_text(encoding="utf-8-sig"))

def _normalize_dataset_dir(d: Path) -> Path:
    inner = d / d.name
    return inner if inner.exists() and inner.is_dir() else d

def _build_index_by_frame_id(data_info: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    idx = {}
    for row in data_info:
        fid = str(row.get("frame_id"))
        idx[fid] = row
    return idx

def get_one_synced_pair(
    base_spd_dir: Path,
    base_images_dir: Path,
    pair_index: int = 0,
) -> Tuple[Path, Path, Dict[str, Any]]:

    base_spd_dir = _normalize_dataset_dir(base_spd_dir)

    coop_info = _read_json(base_spd_dir / "cooperative" / "data_info.json")
    infra_info = _read_json(base_spd_dir / "infrastructure-side" / "data_info.json")
    veh_info = _read_json(base_spd_dir / "vehicle-side" / "data_info.json")

    infra_by_fid = _build_index_by_frame_id(infra_info)
    veh_by_fid = _build_index_by_frame_id(veh_info)

    pair = coop_info[pair_index]

    veh_fid = str(pair["vehicle_frame"])
    inf_fid = str(pair["infrastructure_frame"])

    infra_row = infra_by_fid[inf_fid]
    ego_row = veh_by_fid[veh_fid]

    infra_rel = Path(infra_row["image_path"])
    ego_rel = Path(ego_row["image_path"])

    infra_img_root = _normalize_dataset_dir(base_images_dir / "V2X-Seq-SPD-infrastructure-side-image")
    ego_img_root   = _normalize_dataset_dir(base_images_dir / "V2X-Seq-SPD-vehicle-side-image")

    infra_img = infra_img_root / infra_rel.name
    ego_img   = ego_img_root / ego_rel.name

    meta = {
        "pair_index": pair_index,
        "vehicle_frame": veh_fid,
        "infrastructure_frame": inf_fid,
        "vehicle_sequence": pair.get("vehicle_sequence"),
        "infrastructure_sequence": pair.get("infrastructure_sequence"),
        "system_error_offset": pair.get("system_error_offset"),
        "infra_image_timestamp": infra_row.get("image_timestamp"),
        "ego_image_timestamp": ego_row.get("image_timestamp"),
    }

    return infra_img, ego_img, meta
