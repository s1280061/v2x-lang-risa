from __future__ import annotations

from pathlib import Path
import json
import math

import cv2
import numpy as np


# ===== user paths (HINTS) =====
BASE_HINT = Path(
    r"D:\V2X\Public-V2X-Datasets\V2X-Seq (CVPR2023)\Sequential-Perception-Dataset\Full Dataset (train & val)\V2X-Seq-SPD\V2X-Seq-SPD\V2X-Seq-SPD"
)
PAIR_ROOT = Path(r"D:\V2X\pair_V2X")

INFRA_IMG_DIR = PAIR_ROOT / "infra"
META_DIR = PAIR_ROOT / "meta"


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
        print(f"[WARN] Non-numeric json stems in {dir_path}. Using first: {files[0].name}")
        return load_json(files[0])

    cand.sort(key=lambda x: abs(x[0] - target))
    nearest_num, nearest_path = cand[0]
    print(
        f"[WARN] Missing calib file: {exact.name} -> using nearest: {nearest_path.name} "
        f"(target={target}, nearest={nearest_num})"
    )
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
        if "translation" not in tr or "rotation" not in tr:
            raise KeyError(f"'transform' found but missing keys: {list(tr.keys())}")
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


# ===== new helpers for robust ego bbox selection =====
def point_in_rect(
    u: float, v: float, xmin: float, ymin: float, xmax: float, ymax: float, margin: float = 0.0
) -> bool:
    return (xmin - margin) <= u <= (xmax + margin) and (ymin - margin) <= v <= (ymax + margin)


def point_to_rect_distance(u: float, v: float, xmin: float, ymin: float, xmax: float, ymax: float) -> float:
    """
    Distance from point to rectangle (0 if inside). Pixel units.
    """
    dx = 0.0
    if u < xmin:
        dx = xmin - u
    elif u > xmax:
        dx = u - xmax

    dy = 0.0
    if v < ymin:
        dy = ymin - v
    elif v > ymax:
        dy = v - ymax

    return math.hypot(dx, dy)


def process_one_pair(
    base: Path,
    pair_idx: int,
    out_img_dir: Path,
    out_json_dir: Path,
    calib_dirs: dict[str, Path],
) -> tuple[bool, str]:
    """
    Returns (ok, reason). ok=True means saved.
    """
    meta_path = META_DIR / f"pair_{pair_idx:06d}.json"
    if not meta_path.exists():
        return False, f"meta missing: {meta_path}"

    meta = load_json(meta_path)
    veh_frame = str(meta["vehicle_frame"]).zfill(6)
    inf_frame = str(meta["infrastructure_frame"]).zfill(6)

    # ===== transforms: vehicle -> world =====
    T_lidar_to_nov = parse_T(load_json_nearest(calib_dirs["veh_lidar2nov"], veh_frame))
    T_nov_to_world = parse_T(load_json_nearest(calib_dirs["veh_nov2world"], veh_frame))

    p_lidar = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    p_world = T_nov_to_world @ (T_lidar_to_nov @ p_lidar)
    p_world_xyz = p_world[:3].copy()

    # ===== transforms: world -> infra cam =====
    T_vlidar_to_world = parse_T(load_json_nearest(calib_dirs["inf_vlidar2world"], inf_frame))
    T_vlidar_to_cam = parse_T(load_json_nearest(calib_dirs["inf_vlidar2cam"], inf_frame))
    K_infra = load_K(load_json_nearest(calib_dirs["inf_intrinsic"], inf_frame))

    T_world_to_cam = T_vlidar_to_cam @ np.linalg.inv(T_vlidar_to_world)

    p_world_h = np.array([p_world_xyz[0], p_world_xyz[1], p_world_xyz[2], 1.0], dtype=np.float64)
    p_cam_h = T_world_to_cam @ p_world_h
    p_cam = p_cam_h[:3].copy()

    uv = project_point(K_infra, p_cam)

    # ===== load label + image =====
    infra_label_path = base / "infrastructure-side" / "label" / "camera" / f"{inf_frame}.json"
    if not infra_label_path.exists():
        return False, f"infra label missing: {infra_label_path}"
    inf_label = load_json(infra_label_path)

    img_path = INFRA_IMG_DIR / f"pair_{pair_idx:06d}.jpg"
    img = cv2.imread(str(img_path))
    if img is None:
        return False, f"infra image missing: {img_path}"

    # ===== ego point + robust bbox selection =====
    picked_obj = None
    picked_dist_px = None
    picked_reason = None

    if uv is None:
        picked_reason = "projection_failed_or_behind_camera"
    else:
        u, v = uv
        # draw only the ego projected point
        cv2.circle(img, (int(round(u)), int(round(v))), 8, (0, 255, 255), -1)  # yellow

        # --- tuning ---
        MARGIN_PX = 12.0          # bboxを少し膨らませて「入ってる判定」を緩める
        MAX_OUTSIDE_DIST = 0.0    # 0なら “bbox内（+margin）” のときだけ採用（誤検出最小）
        TYPE_FILTER = False       # typeが安定しているなら True にして vehicle のみに絞ってOK

        def is_vehicle_type(obj: dict) -> bool:
            t = (obj.get("type") or "").lower()
            return t in ("car", "truck", "bus", "van", "vehicle")

        best_inside = None
        best_near = None
        best_near_dist = 1e18

        # まず「中に入ってるbbox」を探す
        for obj in inf_label:
            b = obj.get("2d_box")
            if not isinstance(b, dict):
                continue
            if not all(k in b for k in ("xmin", "ymin", "xmax", "ymax")):
                continue

            if TYPE_FILTER and (obj.get("type") is not None) and (not is_vehicle_type(obj)):
                continue

            xmin, ymin, xmax, ymax = map(float, (b["xmin"], b["ymin"], b["xmax"], b["ymax"]))

            if point_in_rect(u, v, xmin, ymin, xmax, ymax, margin=MARGIN_PX):
                best_inside = obj
                break

            # 保険：rectまでの距離（内なら0）
            d_rect = point_to_rect_distance(u, v, xmin, ymin, xmax, ymax)
            if d_rect < best_near_dist:
                best_near_dist = d_rect
                best_near = obj

        if best_inside is not None:
            picked_obj = best_inside
            picked_dist_px = 0.0
            picked_reason = f"point_inside_bbox(margin={MARGIN_PX})"
        else:
            # insideが無いなら基本採用しない（誤検出回避）
            if best_near is not None and float(best_near_dist) <= MAX_OUTSIDE_DIST:
                picked_obj = best_near
                picked_dist_px = float(best_near_dist)
                picked_reason = f"point_near_bbox(dist<={MAX_OUTSIDE_DIST})"
            else:
                picked_reason = f"point_not_in_any_bbox(best_near_dist={float(best_near_dist):.1f}, margin={MARGIN_PX})"

        # 緑bboxは採用できた場合だけ描く（文字は出さない）
        if picked_obj is not None and isinstance(picked_obj.get("2d_box"), dict):
            b = picked_obj["2d_box"]
            xmin, ymin, xmax, ymax = map(int, (b["xmin"], b["ymin"], b["xmax"], b["ymax"]))
            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 3)

    # header (pair info only)
    text = f"pair {pair_idx:06d} veh={veh_frame} inf={inf_frame}"
    cv2.putText(img, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 3, cv2.LINE_AA)
    cv2.putText(img, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 1, cv2.LINE_AA)

    # ===== save outputs =====
    out_img = out_img_dir / f"pair_{pair_idx:06d}.jpg"
    cv2.imwrite(str(out_img), img, [int(cv2.IMWRITE_JPEG_QUALITY), 92])

    dump = {
        "pair_index": pair_idx,
        "vehicle_frame": veh_frame,
        "infrastructure_frame": inf_frame,
        "ego_world_xyz": p_world_xyz.tolist(),
        "ego_infra_uv": [float(uv[0]), float(uv[1])] if uv is not None else None,
        "p_cam_xyz": p_cam.tolist(),
        # d is JSON only (point->bbox dist in px; inside => 0)
        "picked_bbox_point_dist_px": picked_dist_px,
        "picked_reason": picked_reason,
        "picked_obj_track_id": picked_obj.get("track_id") if picked_obj else None,
        "picked_obj_type": picked_obj.get("type") if picked_obj else None,
        "output_image": str(out_img),
        "input_infra_image": str(img_path),
        "input_infra_label": str(infra_label_path),
    }
    (out_json_dir / f"pair_{pair_idx:06d}.json").write_text(
        json.dumps(dump, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return True, "ok"


def main_all(limit: int | None = None):
    # ===== output folders (must be under D:\V2X\pair_V2X) =====
    out_root = PAIR_ROOT / "ego_projection_all"
    out_img_dir = out_root / "images"
    out_json_dir = out_root / "json"

    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_json_dir.mkdir(parents=True, exist_ok=True)

    # ===== resolve dataset base =====
    base = resolve_base(BASE_HINT)
    print(f"[INFO] BASE = {base}")

    # ===== resolve calib dirs once =====
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

    meta_files = sorted(META_DIR.glob("pair_*.json"))
    if not meta_files:
        raise FileNotFoundError(f"No pair meta json found under: {META_DIR}")

    total = len(meta_files)
    print(f"[INFO] found pair metas: {total}")
    print(f"[INFO] outputs -> {out_root}")

    ok = 0
    skip = 0

    for i, mp in enumerate(meta_files, 1):
        pair_idx = int(mp.stem.split("_")[1])

        if limit is not None and ok >= limit:
            break

        try:
            saved, reason = process_one_pair(base, pair_idx, out_img_dir, out_json_dir, calib_dirs)
            if saved:
                ok += 1
            else:
                skip += 1
                print(f"[SKIP] pair {pair_idx:06d}: {reason}")
        except Exception as e:
            skip += 1
            print(f"[ERR] pair {pair_idx:06d}: {type(e).__name__}: {e}")

        if i % 200 == 0:
            print(f"[PROG] {i}/{total} done, ok={ok}, skip/err={skip}")

    # summary
    summary = {
        "total_pairs_meta": total,
        "ok": ok,
        "skip_or_error": skip,
        "out_root": str(out_root),
        "out_images": str(out_img_dir),
        "out_json": str(out_json_dir),
        "resolved_BASE": str(base),
        "calib_dirs": {k: str(v) for k, v in calib_dirs.items()},
    }
    (out_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("[DONE]", f"ok={ok}", f"skip/err={skip}")
    print("[DONE] summary:", out_root / "summary.json")


if __name__ == "__main__":
    # まず試すなら main_all(limit=50) にしてOK
    main_all(limit=None)