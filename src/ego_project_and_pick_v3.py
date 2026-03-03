from __future__ import annotations

from pathlib import Path
import json
import math
import re
from functools import lru_cache

import cv2
import numpy as np


# ===== user paths =====
BASE_HINT = Path(
    r"D:\V2X\Public-V2X-Datasets\V2X-Seq (CVPR2023)\Sequential-Perception-Dataset\Full Dataset (train & val)\V2X-Seq-SPD\V2X-Seq-SPD\V2X-Seq-SPD"
)
PAIR_ROOT = Path(r"D:\V2X\pair_V2X")

INFRA_IMG_DIR = PAIR_ROOT / "infra"
META_DIR = PAIR_ROOT / "meta"


# ===== options (FAST SET) =====
SAVE_JSON_ONLY_WHEN_HIT = True     # 緑bboxが見つかったpairだけjson出す（最速）
DRAW_YELLOW_POINT = False          # VLM入力用: 黄色点は描かない
JPEG_QUALITY = 85                  # 92より速い＆十分きれい

MARGIN_PX = 12.0                   # bboxを少し膨らませて inside 判定を緩める（誤差吸収）


# ===== io helpers =====
def load_json(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))


def list_json_files(dir_path: Path) -> list[Path]:
    if not dir_path.exists():
        return []
    return sorted([p for p in dir_path.glob("*.json") if p.is_file()])


def load_json_nearest(dir_path: Path, frame6: str) -> dict:
    files = list_json_files(dir_path)
    if not files:
        raise FileNotFoundError(f"No json files found in directory: {dir_path}")

    exact = dir_path / f"{frame6}.json"
    if exact.exists():
        return load_json(exact)

    target = int(frame6)
    cand: list[tuple[int, Path]] = []
    for p in files:
        try:
            cand.append((int(p.stem), p))
        except ValueError:
            continue

    if not cand:
        return load_json(files[0])

    cand.sort(key=lambda x: abs(x[0] - target))
    _, nearest_path = cand[0]
    return load_json(nearest_path)


# ===== BASE resolver =====
def looks_like_v2xseq_root(p: Path) -> bool:
    return (p / "vehicle-side").exists() and (p / "infrastructure-side").exists()


def resolve_base(base_hint: Path) -> Path:
    if looks_like_v2xseq_root(base_hint):
        print(f"[INFO] BASE resolved (direct): {base_hint}")
        return base_hint

    ancestors = [base_hint] + list(base_hint.parents)[:6]
    checked = set()

    for anc in ancestors:
        if anc in checked:
            continue
        checked.add(anc)
        if not anc.exists():
            continue

        if looks_like_v2xseq_root(anc):
            print(f"[INFO] BASE resolved (ancestor): {anc}")
            return anc

        queue: list[tuple[Path, int]] = [(anc, 0)]
        while queue:
            cur, depth = queue.pop(0)
            if depth > 5:
                continue
            if looks_like_v2xseq_root(cur):
                print(f"[INFO] BASE resolved (search): {cur}")
                return cur
            try:
                for child in cur.iterdir():
                    if child.is_dir():
                        queue.append((child, depth + 1))
            except PermissionError:
                continue

    raise FileNotFoundError(f"Could not resolve BASE automatically from hint: {base_hint}")


def find_calib_dir(calib_root: Path, include_keywords: list[str]) -> Path:
    if not calib_root.exists():
        raise FileNotFoundError(f"calib root not found: {calib_root}")

    kws = [k.lower() for k in include_keywords]
    candidates: list[Path] = []

    for d in calib_root.rglob("*"):
        if not d.is_dir():
            continue
        if not list_json_files(d):
            continue
        s = str(d).lower()
        if all(k in s for k in kws):
            candidates.append(d)

    if not candidates:
        top = [p.name for p in calib_root.iterdir() if p.is_dir()]
        raise FileNotFoundError(
            f"Could not find calib directory under {calib_root} with keywords={include_keywords}. "
            f"Top-level dirs: {top}"
        )

    candidates.sort(key=lambda p: len(p.parts))
    chosen = candidates[0]
    print(f"[INFO] calib dir resolved: keywords={include_keywords} -> {chosen}")
    return chosen


# ===== math helpers =====
def Rt_to_T(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t.reshape(3)
    return T


def quat_to_R(x: float, y: float, z: float, w: float) -> np.ndarray:
    q = np.array([x, y, z, w], dtype=np.float64)
    n = float(np.linalg.norm(q))
    if n <= 1e-12:
        return np.eye(3, dtype=np.float64)
    q /= n
    x, y, z, w = q.tolist()

    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z

    return np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float64,
    )


def _vec3_from_any(v) -> np.ndarray:
    if isinstance(v, dict) and all(k in v for k in ("x", "y", "z")):
        return np.array([v["x"], v["y"], v["z"]], dtype=np.float64).reshape(3)
    a = np.array(v, dtype=np.float64).reshape(-1)
    if a.size >= 3:
        return a[:3].reshape(3)
    raise ValueError(f"Cannot parse vec3 from: {type(v)} {v}")


def parse_T(j: dict) -> np.ndarray:
    if "transform" in j and isinstance(j["transform"], dict):
        tr = j["transform"]
        t = _vec3_from_any(tr["translation"])
        rot = tr["rotation"]
        if isinstance(rot, dict) and all(k in rot for k in ("x", "y", "z", "w")):
            R = quat_to_R(float(rot["x"]), float(rot["y"]), float(rot["z"]), float(rot["w"]))
            return Rt_to_T(R, t)
        R = np.array(rot, dtype=np.float64).reshape(3, 3)
        return Rt_to_T(R, t)

    if "rotation" in j and "translation" in j:
        R = np.array(j["rotation"], dtype=np.float64).reshape(3, 3)
        t = _vec3_from_any(j["translation"])
        return Rt_to_T(R, t)

    if "R" in j and ("T" in j or "translation" in j):
        R = np.array(j["R"], dtype=np.float64).reshape(3, 3)
        t = _vec3_from_any(j.get("T", j.get("translation")))
        return Rt_to_T(R, t)

    raise KeyError(f"Unknown transform keys: {list(j.keys())[:15]}")


def load_K(cam_intrinsic_json: dict) -> np.ndarray:
    return np.array(cam_intrinsic_json["cam_K"], dtype=np.float64).reshape(3, 3)


def project_point(K: np.ndarray, p_cam: np.ndarray) -> tuple[float, float] | None:
    z = float(p_cam[2])
    if z <= 1e-6:
        return None
    u = float(K[0, 0] * (p_cam[0] / z) + K[0, 2])
    v = float(K[1, 1] * (p_cam[1] / z) + K[1, 2])
    return u, v


def point_in_rect(u: float, v: float, xmin: float, ymin: float, xmax: float, ymax: float, margin: float = 0.0) -> bool:
    return (xmin - margin) <= u <= (xmax + margin) and (ymin - margin) <= v <= (ymax + margin)


# ===== meta selection =====
PAIR_META_RE = re.compile(r"^pair_(\d{6})$")  # pair_000123.json only


# ===== main per-pair (FAST: no image IO unless hit) =====
def process_one_pair_fast(
    base: Path,
    pair_idx: int,
    meta_path: Path,
    out_img_dir: Path,
    out_json_dir: Path,
    calib_dirs: dict[str, Path],
) -> tuple[bool, str]:
    meta = load_json(meta_path)
    veh_frame = str(meta["vehicle_frame"]).zfill(6)
    inf_frame = str(meta["infrastructure_frame"]).zfill(6)

    # vehicle -> world
    T_lidar_to_nov = parse_T(load_json_nearest(calib_dirs["veh_lidar2nov"], veh_frame))
    T_nov_to_world = parse_T(load_json_nearest(calib_dirs["veh_nov2world"], veh_frame))
    p_world = T_nov_to_world @ (T_lidar_to_nov @ np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64))
    p_world_xyz = p_world[:3].copy()

    # world -> infra cam
    T_vlidar_to_world = parse_T(load_json_nearest(calib_dirs["inf_vlidar2world"], inf_frame))
    T_vlidar_to_cam = parse_T(load_json_nearest(calib_dirs["inf_vlidar2cam"], inf_frame))
    K_infra = load_K(load_json_nearest(calib_dirs["inf_intrinsic"], inf_frame))
    T_world_to_cam = T_vlidar_to_cam @ np.linalg.inv(T_vlidar_to_world)

    p_cam_h = T_world_to_cam @ np.array([p_world_xyz[0], p_world_xyz[1], p_world_xyz[2], 1.0], dtype=np.float64)
    p_cam = p_cam_h[:3].copy()

    uv = project_point(K_infra, p_cam)
    if uv is None:
        return False, "projection_failed_or_behind_camera"

    u, v = uv

    # label load (still needed)
    infra_label_path = base / "infrastructure-side" / "label" / "camera" / f"{inf_frame}.json"
    if not infra_label_path.exists():
        return False, f"infra_label_missing:{infra_label_path}"
    inf_label = load_json(infra_label_path)

    # find bbox that contains point (with margin)
    picked_obj = None
    picked_bbox = None

    for obj in inf_label:
        b = obj.get("2d_box")
        if not isinstance(b, dict):
            continue
        if not all(k in b for k in ("xmin", "ymin", "xmax", "ymax")):
            continue
        xmin, ymin, xmax, ymax = map(float, (b["xmin"], b["ymin"], b["xmax"], b["ymax"]))
        if point_in_rect(u, v, xmin, ymin, xmax, ymax, margin=MARGIN_PX):
            picked_obj = obj
            picked_bbox = (int(xmin), int(ymin), int(xmax), int(ymax))
            break

    if picked_obj is None:
        # 最速：ヒットしないなら画像を読まない・保存しない
        if not SAVE_JSON_ONLY_WHEN_HIT:
            out_json = out_json_dir / f"pair_{pair_idx:06d}.json"
            out_json.write_text(
                json.dumps(
                    {
                        "pair_index": pair_idx,
                        "vehicle_frame": veh_frame,
                        "infrastructure_frame": inf_frame,
                        "ego_world_xyz": p_world_xyz.tolist(),
                        "ego_infra_uv": [float(u), float(v)],
                        "picked": False,
                        "picked_reason": f"point_not_in_any_bbox(margin={MARGIN_PX})",
                        "input_infra_label": str(infra_label_path),
                        "meta_path": str(meta_path),
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
        return False, "no_hit"

    # ===== HIT ONLY: now do image IO =====
    img_path = INFRA_IMG_DIR / f"pair_{pair_idx:06d}.jpg"
    img = cv2.imread(str(img_path))
    if img is None:
        return False, f"infra_image_missing:{img_path}"

    xmin, ymin, xmax, ymax = picked_bbox
    # green bbox only
    cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 3)

    # optional: yellow point (OFF for VLM)
    if DRAW_YELLOW_POINT:
        cv2.circle(img, (int(round(u)), int(round(v))), 6, (0, 255, 255), -1)

    # pair header (small; keep if you want; remove if pure VLM input)
    #text = f"pair {pair_idx:06d} veh={veh_frame} inf={inf_frame}"
    #cv2.putText(img, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 3, cv2.LINE_AA)
    #cv2.putText(img, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 1, cv2.LINE_AA)

    out_img = out_img_dir / f"pair_{pair_idx:06d}.jpg"
    ok = cv2.imwrite(str(out_img), img, [int(cv2.IMWRITE_JPEG_QUALITY), int(JPEG_QUALITY)])
    if not ok:
        return False, f"cv2_imwrite_failed:{out_img}"

    # hit json
    out_json = out_json_dir / f"pair_{pair_idx:06d}.json"
    out_json.write_text(
        json.dumps(
            {
                "pair_index": pair_idx,
                "vehicle_frame": veh_frame,
                "infrastructure_frame": inf_frame,
                "ego_world_xyz": p_world_xyz.tolist(),
                "ego_infra_uv": [float(u), float(v)],
                "picked": True,
                "picked_reason": f"point_inside_bbox(margin={MARGIN_PX})",
                "picked_obj_track_id": picked_obj.get("track_id"),
                "picked_obj_type": picked_obj.get("type"),
                "picked_bbox_xyxy": [xmin, ymin, xmax, ymax],
                "output_image": str(out_img),
                "input_infra_image": str(img_path),
                "input_infra_label": str(infra_label_path),
                "meta_path": str(meta_path),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    return True, "hit_saved"


def main_all(limit: int | None = None):
    # outputs
    out_root = PAIR_ROOT / "ego_projection_hits_fast"
    out_img_dir = out_root / "images"
    out_json_dir = out_root / "json"
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_json_dir.mkdir(parents=True, exist_ok=True)

    base = resolve_base(BASE_HINT)
    print(f"[INFO] BASE = {base}")
    print(f"[INFO] outputs -> {out_root}")

    veh_calib_root = base / "vehicle-side" / "calib"
    inf_calib_root = base / "infrastructure-side" / "calib"
    calib_dirs = {
        "veh_lidar2nov": find_calib_dir(veh_calib_root, ["lidar", "novatel"]),
        "veh_nov2world": find_calib_dir(veh_calib_root, ["novatel", "world"]),
        "inf_vlidar2world": find_calib_dir(inf_calib_root, ["virtuallidar", "world"]),
        "inf_vlidar2cam": find_calib_dir(inf_calib_root, ["virtuallidar", "camera"]),
        "inf_intrinsic": find_calib_dir(inf_calib_root, ["intrinsic"]),
    }
    print("[INFO] calib dirs resolved.")

    # meta files: pair_000123.json only
    raw = sorted(META_DIR.glob("pair_*.json"))
    meta_files: list[tuple[int, Path]] = []
    for p in raw:
        m = PAIR_META_RE.match(p.stem)
        if not m:
            continue
        meta_files.append((int(m.group(1)), p))
    meta_files.sort(key=lambda x: x[0])

    total = len(meta_files)
    print(f"[INFO] found pair metas: {total}")

    hit = 0
    miss = 0
    err = 0

    for i, (pair_idx, meta_path) in enumerate(meta_files, 1):
        if limit is not None and i > limit:
            break
        try:
            ok, reason = process_one_pair_fast(base, pair_idx, meta_path, out_img_dir, out_json_dir, calib_dirs)
            if ok:
                hit += 1
            else:
                if reason == "no_hit":
                    miss += 1
                else:
                    err += 1
                    print(f"[ERR] pair {pair_idx:06d}: {reason}")
        except Exception as e:
            err += 1
            print(f"[ERR] pair {pair_idx:06d}: {type(e).__name__}: {e}")

        if i % 500 == 0:
            print(f"[PROG] {i}/{total} hit={hit} miss={miss} err={err}")

    (out_root / "summary.json").write_text(
        json.dumps(
            {
                "total_pairs_meta": total,
                "processed": min(total, limit) if limit is not None else total,
                "hit_saved": hit,
                "miss_no_save": miss,
                "error": err,
                "out_root": str(out_root),
                "out_images": str(out_img_dir),
                "out_json": str(out_json_dir),
                "resolved_BASE": str(base),
                "margin_px": MARGIN_PX,
                "jpeg_quality": JPEG_QUALITY,
                "draw_yellow_point": DRAW_YELLOW_POINT,
                "save_json_only_when_hit": SAVE_JSON_ONLY_WHEN_HIT,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("[DONE]", f"hit={hit}", f"miss={miss}", f"err={err}")
    print("[DONE] summary:", out_root / "summary.json")


if __name__ == "__main__":
    # まず軽く試すなら limit=200
    main_all(limit=None)