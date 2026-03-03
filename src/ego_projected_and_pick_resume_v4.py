from __future__ import annotations

from pathlib import Path
import json
import re
import time
import gc
from bisect import bisect_left
from functools import lru_cache
from concurrent.futures import ProcessPoolExecutor, as_completed

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


# ===== meta selection =====
PAIR_META_RE = re.compile(r"^pair_(\d{6})$")  # pair_000123.json only


# ===== io helpers =====
def load_json(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))


def _dir_has_any_json(d: Path) -> bool:
    # list_json_files を作らず最速チェック（find_calib_dir 用）
    try:
        next(d.glob("*.json"))
        return True
    except StopIteration:
        return False


@lru_cache(maxsize=None)
def _build_json_index(dir_path_str: str) -> tuple[list[int], list[str]]:
    """
    calib dir の *.json を最初の1回だけ走査して index 化（各プロセス内でキャッシュ）
    returns:
      frames: sorted int list
      paths:  sorted str list (same order)
    """
    d = Path(dir_path_str)
    if not d.exists():
        raise FileNotFoundError(f"calib dir not found: {d}")

    frames: list[int] = []
    paths: list[str] = []
    for p in d.glob("*.json"):
        if not p.is_file():
            continue
        try:
            frames.append(int(p.stem))
            paths.append(str(p))
        except ValueError:
            continue

    if not frames:
        raise FileNotFoundError(f"No json files found in directory: {d}")

    order = sorted(range(len(frames)), key=frames.__getitem__)
    frames = [frames[i] for i in order]
    paths = [paths[i] for i in order]
    return frames, paths


def load_json_nearest(dir_path: Path, frame6: str) -> dict:
    # exact 優先（最速）
    exact = dir_path / f"{frame6}.json"
    if exact.exists():
        return load_json(exact)

    frames, paths = _build_json_index(str(dir_path))
    target = int(frame6)

    pos = bisect_left(frames, target)
    if pos <= 0:
        return load_json(Path(paths[0]))
    if pos >= len(frames):
        return load_json(Path(paths[-1]))

    li = pos - 1
    ri = pos
    if abs(frames[li] - target) <= abs(frames[ri] - target):
        return load_json(Path(paths[li]))
    return load_json(Path(paths[ri]))


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
        if not _dir_has_any_json(d):
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


def point_in_rect(
    u: float, v: float, xmin: float, ymin: float, xmax: float, ymax: float, margin: float = 0.0
) -> bool:
    return (xmin - margin) <= u <= (xmax + margin) and (ymin - margin) <= v <= (ymax + margin)


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

    infra_label_path = base / "infrastructure-side" / "label" / "camera" / f"{inf_frame}.json"
    if not infra_label_path.exists():
        return False, f"infra_label_missing:{infra_label_path}"
    inf_label = load_json(infra_label_path)

    picked_obj = None
    picked_bbox = None

    cands = []  # (score, area, dist2, obj, bbox_int)

    for obj in inf_label:
        b = obj.get("2d_box")
        if not isinstance(b, dict):
            continue
        if not all(k in b for k in ("xmin", "ymin", "xmax", "ymax")):
            continue

        xmin, ymin, xmax, ymax = map(float, (b["xmin"], b["ymin"], b["xmax"], b["ymax"]))
        if not point_in_rect(u, v, xmin, ymin, xmax, ymax, margin=MARGIN_PX):
            continue

        w = max(0.0, xmax - xmin)
        h = max(0.0, ymax - ymin)
        area = w * h

        cx = 0.5 * (xmin + xmax)
        cy = 0.5 * (ymin + ymax)
        dist2 = (u - cx) ** 2 + (v - cy) ** 2

        obj_type = (obj.get("type") or "").lower()

        type_bias = 0.0
        if "car" in obj_type:
            type_bias = -1e6  # car を超優遇
        elif "truck" in obj_type or "bus" in obj_type:
            type_bias = +1e6  # 大型車は後回し

        score = area + 0.5 * dist2 + type_bias

        cands.append((score, area, dist2, obj, (int(xmin), int(ymin), int(xmax), int(ymax))))

    if cands:
        cands.sort(key=lambda x: x[0])
        _, _, _, picked_obj, picked_bbox = cands[0]

    if picked_obj is None:
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

    # ===== HIT ONLY: image IO =====
    img_path = INFRA_IMG_DIR / f"pair_{pair_idx:06d}.jpg"
    img = cv2.imread(str(img_path))
    if img is None:
        return False, f"infra_image_missing:{img_path}"

    xmin, ymin, xmax, ymax = picked_bbox
    cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 3)

    if DRAW_YELLOW_POINT:
        cv2.circle(img, (int(round(u)), int(round(v))), 6, (0, 255, 255), -1)

    out_img = out_img_dir / f"pair_{pair_idx:06d}.jpg"
    ok = cv2.imwrite(str(out_img), img, [int(cv2.IMWRITE_JPEG_QUALITY), int(JPEG_QUALITY)])
    if not ok:
        return False, f"cv2_imwrite_failed:{out_img}"

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


def _should_retry_oserror(e: OSError) -> bool:
    winerror = getattr(e, "winerror", None)
    if winerror == 1450:
        return True
    if getattr(e, "errno", None) == 22:
        return True
    return False


def _process_one_pair_with_retry(args) -> tuple[int, bool, str]:
    (
        base_str,
        pair_idx,
        meta_path_str,
        out_img_dir_str,
        out_json_dir_str,
        calib_dirs_str,
        retry,
        retry_sleep_sec,
    ) = args

    base = Path(base_str)
    meta_path = Path(meta_path_str)
    out_img_dir = Path(out_img_dir_str)
    out_json_dir = Path(out_json_dir_str)
    calib_dirs = {k: Path(v) for k, v in calib_dirs_str.items()}

    for attempt in range(retry + 1):
        try:
            ok, reason = process_one_pair_fast(
                base=base,
                pair_idx=pair_idx,
                meta_path=meta_path,
                out_img_dir=out_img_dir,
                out_json_dir=out_json_dir,
                calib_dirs=calib_dirs,
            )
            return pair_idx, ok, reason

        except OSError as e:
            if attempt < retry and _should_retry_oserror(e):
                gc.collect()
                time.sleep(retry_sleep_sec)
                continue
            return pair_idx, False, f"{type(e).__name__}: {e}"

        except Exception as e:
            return pair_idx, False, f"{type(e).__name__}: {e}"

    return pair_idx, False, "retry_exhausted"


def main_all_parallel(
    limit: int | None = None,
    start_pair: int = 0,
    retry: int = 3,
    retry_sleep_sec: float = 2.0,
    max_workers: int = 2,    # ★ workers=2
    chunksize: int = 50,
):
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

    raw = sorted(META_DIR.glob("pair_*.json"))
    meta_files: list[tuple[int, Path]] = []
    for p in raw:
        m = PAIR_META_RE.match(p.stem)
        if not m:
            continue
        idx = int(m.group(1))
        if idx < start_pair:
            continue
        meta_files.append((idx, p))
    meta_files.sort(key=lambda x: x[0])

    if limit is not None:
        meta_files = meta_files[:limit]

    total = len(meta_files)
    print(f"[INFO] found pair metas (after start_pair={start_pair}): {total}")
    print(f"[INFO] parallel: max_workers={max_workers}, chunksize={chunksize}")

    hit = 0
    miss = 0
    err = 0
    done_n = 0

    base_str = str(base)
    out_img_dir_str = str(out_img_dir)
    out_json_dir_str = str(out_json_dir)
    calib_dirs_str = {k: str(v) for k, v in calib_dirs.items()}

    tasks = [
        (
            base_str,
            pair_idx,
            str(meta_path),
            out_img_dir_str,
            out_json_dir_str,
            calib_dirs_str,
            retry,
            retry_sleep_sec,
        )
        for (pair_idx, meta_path) in meta_files
    ]

    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = []
        for k in range(0, len(tasks), chunksize):
            batch = tasks[k : k + chunksize]
            for t in batch:
                futures.append(ex.submit(_process_one_pair_with_retry, t))

            for fut in as_completed(futures):
                pair_idx, ok, reason = fut.result()
                done_n += 1

                if ok:
                    hit += 1
                else:
                    if reason == "no_hit":
                        miss += 1
                    else:
                        err += 1
                        print(f"[ERR] pair {pair_idx:06d}: {reason}")

                if done_n % 500 == 0:
                    print(f"[PROG] {done_n}/{total} hit={hit} miss={miss} err={err}")

            futures.clear()

    (out_root / "summary.json").write_text(
        json.dumps(
            {
                "start_pair": start_pair,
                "total_pairs_meta_after_start": total,
                "processed": total,
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
                "retry": retry,
                "retry_sleep_sec": retry_sleep_sec,
                "max_workers": max_workers,
                "chunksize": chunksize,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("[DONE]", f"hit={hit}", f"miss={miss}", f"err={err}")
    print("[DONE] summary:", out_root / "summary.json")


if __name__ == "__main__":
    # ★ 最初から開始、workers=2
    main_all_parallel(
        limit=None,
        start_pair=0,   # ← ここを 0 に変更
        retry=3,
        retry_sleep_sec=2.0,
        max_workers=2,
        chunksize=50,
    )