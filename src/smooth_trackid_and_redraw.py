from __future__ import annotations

from pathlib import Path
import json
import re
import time
import gc
from bisect import bisect_left
from functools import lru_cache
from collections import Counter, deque
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import freeze_support

import cv2
import numpy as np


# ============================================================
# CONFIG
# ============================================================

# ===== user paths =====
BASE_HINT = Path(
    r"D:\V2X\Public-V2X-Datasets\V2X-Seq (CVPR2023)\Sequential-Perception-Dataset\Full Dataset (train & val)\V2X-Seq-SPD\V2X-Seq-SPD\V2X-Seq-SPD"
)
PAIR_ROOT = Path(r"D:\V2X\pair_V2X")

INFRA_IMG_DIR = PAIR_ROOT / "infra"
META_DIR = PAIR_ROOT / "meta"

# ===== outputs =====
# Phase-1: raw (per-pair) results
OUT_RAW_ROOT = PAIR_ROOT / "ego_projection_raw_parallel"
OUT_RAW_JSON_DIR = OUT_RAW_ROOT / "json"         # raw json (できれば全pair)
OUT_RAW_IMG_DIR = OUT_RAW_ROOT / "images_hit"    # raw hit 画像（任意）

# Phase-2: smoothed results (W=7 majority vote on track_id)
OUT_SMOOTH_ROOT = PAIR_ROOT / "ego_projection_smoothed_W7"
OUT_SMOOTH_JSON_DIR = OUT_SMOOTH_ROOT / "json"
OUT_SMOOTH_IMG_DIR = OUT_SMOOTH_ROOT / "images"

# ===== speed / behavior =====
JPEG_QUALITY = 85
MARGIN_PX = 12.0

# 重要: track_id 多数決をやるなら「全pairのraw json」がある方が安定する
# True: picked=False のpairも raw json を保存（おすすめ）
# False: hit のみ保存（最速だが smoothing の窓が歯抜けになりやすい）
SAVE_RAW_JSON_FOR_ALL_PAIRS = True

# raw hit のときに画像を保存するか（あとで smoothed で描き直すなら False でもOK）
SAVE_RAW_HIT_IMAGES = True

# resume（既に出力があればスキップ）
RESUME_SKIP_IF_RAW_JSON_EXISTS = True
RESUME_SKIP_IF_SMOOTH_JSON_EXISTS = True

# ===== selection rules (raw pick) =====
TYPE_BIAS_ENABLE = True
TYPE_BIAS_CAR = -1e6
TYPE_BIAS_TRUCK_BUS = +1e6
AREA_MIN_PX2: float | None = None   # 例: 1200.0 を入れると小さい bbox を除外
SCORE_DIST2_WEIGHT = 0.5            # score = area + w*dist2 + type_bias

# ===== parallel =====
MAX_WORKERS = 2
CHUNKSIZE = 50
RETRY = 3
RETRY_SLEEP_SEC = 2.0

# ===== smoothing =====
SMOOTH_W = 7
SMOOTH_TIE_BREAK = "recent"   # "recent" or "keep_last"
SMOOTH_STICKY = True          # True: 直前の smoothed が窓内にあればそれを優先（切替抑制）

# ===== meta selection =====
PAIR_META_RE = re.compile(r"^pair_(\d{6})$")  # pair_000123.json only


# ============================================================
# IO HELPERS
# ============================================================

def load_json(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))


def save_json(p: Path, obj: dict) -> None:
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def _dir_has_any_json(d: Path) -> bool:
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


def _to_int_track_id(x) -> int | None:
    """
    track_id が "001463" (str) / 1463 (int) の両方ありえるので int に正規化
    """
    if x is None:
        return None
    try:
        s = str(x).strip()
        if s == "":
            return None
        return int(s)
    except Exception:
        return None


# ============================================================
# BASE / CALIB RESOLVER
# ============================================================

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


# ============================================================
# MATH HELPERS
# ============================================================

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


# ============================================================
# PHASE-1: RAW PICK
# ============================================================

def _type_bias(obj_type: str) -> float:
    if not TYPE_BIAS_ENABLE:
        return 0.0
    t = (obj_type or "").lower()
    if "car" in t:
        return TYPE_BIAS_CAR
    if "truck" in t or "bus" in t:
        return TYPE_BIAS_TRUCK_BUS
    return 0.0


def process_one_pair_raw(
    base: Path,
    pair_idx: int,
    meta_path: Path,
    out_raw_json_dir: Path,
    out_raw_img_dir: Path,
    calib_dirs: dict[str, Path],
) -> tuple[bool, str]:
    """
    returns: (ok, reason)
      ok=True: 成功扱い（pickedでもno_hitでも）
      ok=False: エラー
    """

    out_json = out_raw_json_dir / f"pair_{pair_idx:06d}.json"
    if RESUME_SKIP_IF_RAW_JSON_EXISTS and out_json.exists():
        return True, "already_done"

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
        if SAVE_RAW_JSON_FOR_ALL_PAIRS:
            save_json(
                out_json,
                {
                    "pair_index": pair_idx,
                    "vehicle_frame": veh_frame,
                    "infrastructure_frame": inf_frame,
                    "ego_world_xyz": p_world_xyz.tolist(),
                    "ego_infra_uv": None,
                    "picked": False,
                    "picked_reason": "projection_failed_or_behind_camera",
                    "meta_path": str(meta_path),
                },
            )
        return True, "projection_failed_or_behind_camera"
    u, v = uv

    infra_label_path = base / "infrastructure-side" / "label" / "camera" / f"{inf_frame}.json"
    if not infra_label_path.exists():
        return False, f"infra_label_missing:{infra_label_path}"

    inf_label = load_json(infra_label_path)

    cands = []
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

        if AREA_MIN_PX2 is not None and area < float(AREA_MIN_PX2):
            continue

        cx = 0.5 * (xmin + xmax)
        cy = 0.5 * (ymin + ymax)
        dist2 = (u - cx) ** 2 + (v - cy) ** 2

        bias = _type_bias(str(obj.get("type") or ""))
        score = area + SCORE_DIST2_WEIGHT * dist2 + bias

        cands.append((score, area, dist2, obj, (int(xmin), int(ymin), int(xmax), int(ymax))))

    if not cands:
        if SAVE_RAW_JSON_FOR_ALL_PAIRS:
            save_json(
                out_json,
                {
                    "pair_index": pair_idx,
                    "vehicle_frame": veh_frame,
                    "infrastructure_frame": inf_frame,
                    "ego_world_xyz": p_world_xyz.tolist(),
                    "ego_infra_uv": [float(u), float(v)],
                    "picked": False,
                    "picked_reason": f"no_bbox_contains_point(margin={MARGIN_PX})",
                    "meta_path": str(meta_path),
                    "input_infra_label": str(infra_label_path),
                },
            )
        return True, "no_hit"

    cands.sort(key=lambda x: x[0])
    score, area, dist2, picked_obj, picked_bbox = cands[0]
    xmin, ymin, xmax, ymax = picked_bbox

    raw_record = {
        "pair_index": pair_idx,
        "vehicle_frame": veh_frame,
        "infrastructure_frame": inf_frame,
        "ego_world_xyz": p_world_xyz.tolist(),
        "ego_infra_uv": [float(u), float(v)],
        "picked": True,
        "picked_reason": f"best_candidate(score=area+{SCORE_DIST2_WEIGHT}*dist2+type_bias, margin={MARGIN_PX})",
        "picked_obj_track_id": picked_obj.get("track_id"),
        "picked_obj_type": picked_obj.get("type"),
        "picked_bbox_xyxy": [xmin, ymin, xmax, ymax],
        "picked_score": float(score),
        "picked_area": float(area),
        "picked_dist2": float(dist2),
        "meta_path": str(meta_path),
        "input_infra_label": str(infra_label_path),
        "input_infra_image": str(INFRA_IMG_DIR / f"pair_{pair_idx:06d}.jpg"),
    }
    save_json(out_json, raw_record)

    if SAVE_RAW_HIT_IMAGES:
        img_path = INFRA_IMG_DIR / f"pair_{pair_idx:06d}.jpg"
        img = cv2.imread(str(img_path))
        if img is None:
            return False, f"infra_image_missing:{img_path}"
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 3)
        out_img = out_raw_img_dir / f"pair_{pair_idx:06d}.jpg"
        ok = cv2.imwrite(str(out_img), img, [int(cv2.IMWRITE_JPEG_QUALITY), int(JPEG_QUALITY)])
        if not ok:
            return False, f"cv2_imwrite_failed:{out_img}"

    return True, "hit"


def _should_retry_oserror(e: OSError) -> bool:
    winerror = getattr(e, "winerror", None)
    if winerror == 1450:
        return True
    if getattr(e, "errno", None) == 22:
        return True
    return False


def _worker_one_pair_with_retry(args) -> tuple[int, bool, str]:
    (
        base_str,
        pair_idx,
        meta_path_str,
        out_raw_json_dir_str,
        out_raw_img_dir_str,
        calib_dirs_str,
        retry,
        retry_sleep_sec,
    ) = args

    base = Path(base_str)
    meta_path = Path(meta_path_str)
    out_raw_json_dir = Path(out_raw_json_dir_str)
    out_raw_img_dir = Path(out_raw_img_dir_str)
    calib_dirs = {k: Path(v) for k, v in calib_dirs_str.items()}

    for attempt in range(retry + 1):
        try:
            ok, reason = process_one_pair_raw(
                base=base,
                pair_idx=pair_idx,
                meta_path=meta_path,
                out_raw_json_dir=out_raw_json_dir,
                out_raw_img_dir=out_raw_img_dir,
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


def phase1_run_raw_parallel(
    start_pair: int = 0,
    limit: int | None = None,
):
    OUT_RAW_JSON_DIR.mkdir(parents=True, exist_ok=True)
    OUT_RAW_IMG_DIR.mkdir(parents=True, exist_ok=True)

    base = resolve_base(BASE_HINT)
    print(f"[INFO] BASE = {base}")
    print(f"[INFO] outputs raw -> {OUT_RAW_ROOT}")

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
    print(f"[INFO] pairs: {total} (start_pair={start_pair}, limit={limit})")
    print(f"[INFO] parallel: workers={MAX_WORKERS}, chunksize={CHUNKSIZE}")
    print(f"[INFO] SAVE_RAW_JSON_FOR_ALL_PAIRS={SAVE_RAW_JSON_FOR_ALL_PAIRS}, SAVE_RAW_HIT_IMAGES={SAVE_RAW_HIT_IMAGES}")

    base_str = str(base)
    calib_dirs_str = {k: str(v) for k, v in calib_dirs.items()}

    tasks = [
        (
            base_str,
            pair_idx,
            str(meta_path),
            str(OUT_RAW_JSON_DIR),
            str(OUT_RAW_IMG_DIR),
            calib_dirs_str,
            RETRY,
            RETRY_SLEEP_SEC,
        )
        for (pair_idx, meta_path) in meta_files
    ]

    hit = 0
    no_hit = 0
    skipped = 0
    err = 0
    done_n = 0

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = []
        for k in range(0, len(tasks), CHUNKSIZE):
            batch = tasks[k : k + CHUNKSIZE]
            for t in batch:
                futures.append(ex.submit(_worker_one_pair_with_retry, t))

            for fut in as_completed(futures):
                pair_idx, ok, reason = fut.result()
                done_n += 1

                if not ok:
                    err += 1
                    print(f"[ERR] pair {pair_idx:06d}: {reason}")
                else:
                    if reason == "hit":
                        hit += 1
                    elif reason == "no_hit" or reason.startswith("projection_failed"):
                        no_hit += 1
                    elif reason == "already_done":
                        skipped += 1

                if done_n % 500 == 0:
                    print(f"[PROG] {done_n}/{total} hit={hit} no_hit={no_hit} skipped={skipped} err={err}")

            futures.clear()

    summary = {
        "phase": "raw",
        "start_pair": start_pair,
        "limit": limit,
        "total_pairs": total,
        "hit": hit,
        "no_hit": no_hit,
        "skipped": skipped,
        "error": err,
        "out_raw_root": str(OUT_RAW_ROOT),
        "out_raw_json": str(OUT_RAW_JSON_DIR),
        "out_raw_images_hit": str(OUT_RAW_IMG_DIR),
        "resolved_BASE": str(base),
        "margin_px": MARGIN_PX,
        "jpeg_quality": JPEG_QUALITY,
        "save_raw_json_for_all_pairs": SAVE_RAW_JSON_FOR_ALL_PAIRS,
        "save_raw_hit_images": SAVE_RAW_HIT_IMAGES,
        "type_bias_enable": TYPE_BIAS_ENABLE,
        "area_min_px2": AREA_MIN_PX2,
        "score_dist2_weight": SCORE_DIST2_WEIGHT,
        "max_workers": MAX_WORKERS,
        "chunksize": CHUNKSIZE,
        "retry": RETRY,
        "retry_sleep_sec": RETRY_SLEEP_SEC,
    }
    save_json(OUT_RAW_ROOT / "summary_raw.json", summary)
    print("[DONE][RAW] summary:", OUT_RAW_ROOT / "summary_raw.json")


# ============================================================
# PHASE-2: SMOOTH (W=7 majority vote on track_id) + REDRAW
# ============================================================

def _mode_with_tie_break(history: list[int], last_smoothed: int | None) -> int:
    c = Counter(history)
    max_cnt = max(c.values())
    candidates = {k for k, v in c.items() if v == max_cnt}

    if SMOOTH_TIE_BREAK == "keep_last" and last_smoothed is not None and last_smoothed in candidates:
        return last_smoothed

    for tid in reversed(history):
        if tid in candidates:
            return tid
    return history[-1]


def _find_bbox_by_track_id(inf_label: list[dict], track_id: int):
    target = int(track_id)
    for obj in inf_label:
        obj_tid = _to_int_track_id(obj.get("track_id"))
        if obj_tid is None or obj_tid != target:
            continue
        b = obj.get("2d_box")
        if not isinstance(b, dict):
            continue
        if not all(k in b for k in ("xmin", "ymin", "xmax", "ymax")):
            continue
        xmin, ymin, xmax, ymax = map(int, map(float, (b["xmin"], b["ymin"], b["xmax"], b["ymax"])))
        return obj, (xmin, ymin, xmax, ymax)
    return None, None


def phase2_smooth_and_redraw():
    OUT_SMOOTH_JSON_DIR.mkdir(parents=True, exist_ok=True)
    OUT_SMOOTH_IMG_DIR.mkdir(parents=True, exist_ok=True)

    base = resolve_base(BASE_HINT)
    infra_label_dir = base / "infrastructure-side" / "label" / "camera"

    raw_files = sorted(OUT_RAW_JSON_DIR.glob("pair_*.json"))
    pairs = []
    for p in raw_files:
        try:
            idx = int(p.stem.split("_")[1])
            pairs.append((idx, p))
        except Exception:
            continue
    pairs.sort(key=lambda x: x[0])

    print(f"[INFO][SMOOTH] input raw json: {OUT_RAW_JSON_DIR} ({len(pairs)} files)")
    print(f"[INFO][SMOOTH] outputs -> {OUT_SMOOTH_ROOT}")
    print(f"[INFO][SMOOTH] W={SMOOTH_W}, sticky={SMOOTH_STICKY}, tie_break={SMOOTH_TIE_BREAK}")

    win = deque(maxlen=SMOOTH_W)
    last_smoothed: int | None = None

    done_n = 0
    redrawn_n = 0
    skipped_n = 0
    err_n = 0

    for pair_idx, p in pairs:
        out_json = OUT_SMOOTH_JSON_DIR / f"pair_{pair_idx:06d}.json"
        if RESUME_SKIP_IF_SMOOTH_JSON_EXISTS and out_json.exists():
            skipped_n += 1
            continue

        try:
            j = load_json(p)

            picked = bool(j.get("picked", False))
            raw_tid = _to_int_track_id(j.get("picked_obj_track_id", None))
            inf_frame = str(j.get("infrastructure_frame")).zfill(6)

            # vote update (picked=True のときだけ)
            if picked and raw_tid is not None:
                win.append(raw_tid)

            smoothed_tid: int | None = None
            if len(win) > 0:
                smoothed_tid = _mode_with_tie_break(list(win), last_smoothed)

                if SMOOTH_STICKY and last_smoothed is not None and last_smoothed in win:
                    smoothed_tid = last_smoothed

            redrawn = False
            smoothed_bbox = None
            smoothed_type = None
            out_img_path = OUT_SMOOTH_IMG_DIR / f"pair_{pair_idx:06d}.jpg"

            if smoothed_tid is not None:
                label_path = infra_label_dir / f"{inf_frame}.json"
                if label_path.exists():
                    inf_label = load_json(label_path)
                    obj, bbox = _find_bbox_by_track_id(inf_label, smoothed_tid)
                    if bbox is not None:
                        img_path = Path(j.get("input_infra_image", str(INFRA_IMG_DIR / f"pair_{pair_idx:06d}.jpg")))
                        img = cv2.imread(str(img_path))
                        if img is not None:
                            x1, y1, x2, y2 = bbox
                            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
                            ok = cv2.imwrite(str(out_img_path), img, [int(cv2.IMWRITE_JPEG_QUALITY), int(JPEG_QUALITY)])
                            if ok:
                                redrawn = True
                                smoothed_bbox = [x1, y1, x2, y2]
                                smoothed_type = obj.get("type")

            out = dict(j)
            out["smoothed_window_W"] = SMOOTH_W
            out["smoothed_track_id"] = smoothed_tid
            out["smoothed_bbox_xyxy"] = smoothed_bbox
            out["smoothed_obj_type"] = smoothed_type
            out["smoothed_redrawn"] = redrawn
            if redrawn:
                out["output_image_smoothed"] = str(out_img_path)

            save_json(out_json, out)

            done_n += 1
            if redrawn:
                redrawn_n += 1
            last_smoothed = smoothed_tid

            if done_n % 500 == 0:
                print(f"[PROG][SMOOTH] {done_n}/{len(pairs)} redrawn={redrawn_n} skipped={skipped_n} err={err_n}")

        except Exception as e:
            err_n += 1
            print(f"[ERR][SMOOTH] pair {pair_idx:06d}: {type(e).__name__}: {e}")

    summary = {
        "phase": "smooth",
        "input_raw_json_dir": str(OUT_RAW_JSON_DIR),
        "total_raw_json": len(pairs),
        "processed": done_n,
        "redrawn": redrawn_n,
        "skipped": skipped_n,
        "error": err_n,
        "out_smooth_root": str(OUT_SMOOTH_ROOT),
        "out_smooth_json": str(OUT_SMOOTH_JSON_DIR),
        "out_smooth_images": str(OUT_SMOOTH_IMG_DIR),
        "W": SMOOTH_W,
        "sticky": SMOOTH_STICKY,
        "tie_break": SMOOTH_TIE_BREAK,
        "jpeg_quality": JPEG_QUALITY,
    }
    save_json(OUT_SMOOTH_ROOT / "summary_smooth.json", summary)
    print("[DONE][SMOOTH] summary:", OUT_SMOOTH_ROOT / "summary_smooth.json")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    freeze_support()

    # Phase-1: 最初から全部（raw json を作る）
    phase1_run_raw_parallel(
        start_pair=0,
        limit=None,
    )

    # Phase-2: W=7 の track_id 多数決で smoothed bbox を描き直す
    phase2_smooth_and_redraw()