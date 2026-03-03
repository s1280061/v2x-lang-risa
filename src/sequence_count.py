from __future__ import annotations

from pathlib import Path
import json
import re
import csv
from collections import Counter, defaultdict

# ===== user paths =====
PAIRS_BOXL_JSONL = Path(r"D:\V2X\pair_V2X\exports\pairs_boxes.jsonl")
HITS_JSON_DIR = Path(r"D:\V2X\pair_V2X\ego_projection_hits_fast\json")

OUT_CSV = Path(r"D:\V2X\pair_V2X\exports\sequence_frame_counts.csv")
OUT_SUMMARY_TXT = Path(r"D:\V2X\pair_V2X\exports\sequence_frame_counts_summary.txt")

PAIR_NAME_RE = re.compile(r"pair_(\d+)\.json$", re.IGNORECASE)


def collect_hit_pair_indices(hits_dir: Path) -> set[int]:
    """Read pair_XXXXXX.json files and collect pair_index as int."""
    if not hits_dir.exists():
        raise FileNotFoundError(f"hits json dir not found: {hits_dir}")

    pair_indices: set[int] = set()
    for p in hits_dir.glob("pair_*.json"):
        m = PAIR_NAME_RE.search(p.name)
        if m:
            pair_indices.add(int(m.group(1)))
            continue

        # fallback: read json to get pair_index
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
            if "pair_index" in d:
                pair_indices.add(int(d["pair_index"]))
        except Exception:
            pass

    if not pair_indices:
        raise RuntimeError(f"No pair indices found in: {hits_dir}")
    return pair_indices


def stream_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                # skip broken line, but keep going
                print(f"[WARN] JSON decode error at line {i}: {e}")
                continue


def main():
    if not PAIRS_BOXL_JSONL.exists():
        raise FileNotFoundError(f"pairs_boxes.jsonl not found: {PAIRS_BOXL_JSONL}")

    hit_pairs = collect_hit_pair_indices(HITS_JSON_DIR)
    print(f"[INFO] hit pair count (from hits_fast/json): {len(hit_pairs):,}")

    # sequence -> count
    seq_counter = Counter()

    # (optional) also keep example pair indices per seq
    seq_examples: dict[str, list[int]] = defaultdict(list)

    total_rows = 0
    kept_rows = 0

    for obj in stream_jsonl(PAIRS_BOXL_JSONL):
        total_rows += 1

        pair_index = obj.get("pair_index", None)
        if pair_index is None:
            continue

        try:
            pair_index = int(pair_index)
        except Exception:
            continue

        if pair_index not in hit_pairs:
            continue

        kept_rows += 1

        # prefer vehicle_sequence; fallback to infrastructure_sequence
        seq = obj.get("vehicle_sequence") or obj.get("infrastructure_sequence") or "UNKNOWN"
        seq = str(seq)

        seq_counter[seq] += 1
        if len(seq_examples[seq]) < 5:
            seq_examples[seq].append(pair_index)

    print(f"[INFO] pairs_boxes.jsonl rows: total={total_rows:,} kept(hit)={kept_rows:,}")
    print(f"[INFO] unique sequences (hit only): {len(seq_counter):,}")

    # write CSV
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    rows = sorted(seq_counter.items(), key=lambda x: (int(x[0]) if x[0].isdigit() else 10**9, x[0]))

    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["sequence", "frames", "example_pair_indices"])
        for seq, cnt in rows:
            ex = " ".join(map(str, seq_examples.get(seq, [])))
            w.writerow([seq, cnt, ex])

    # write summary
    counts = list(seq_counter.values())
    counts_sorted = sorted(counts)
    min_c = counts_sorted[0] if counts_sorted else 0
    max_c = counts_sorted[-1] if counts_sorted else 0
    mean_c = (sum(counts_sorted) / len(counts_sorted)) if counts_sorted else 0.0
    median_c = counts_sorted[len(counts_sorted)//2] if counts_sorted else 0

    top10 = sorted(seq_counter.items(), key=lambda x: x[1], reverse=True)[:10]

    summary_lines = [
        f"hit pairs (hits_fast/json): {len(hit_pairs):,}",
        f"pairs_boxes.jsonl total rows: {total_rows:,}",
        f"pairs kept (hit only): {kept_rows:,}",
        f"unique sequences (hit only): {len(seq_counter):,}",
        "",
        f"frames per sequence: min={min_c}, median={median_c}, mean={mean_c:.2f}, max={max_c}",
        "",
        "top10 sequences by frames:",
    ]
    for seq, cnt in top10:
        summary_lines.append(f"  seq {seq}: {cnt} frames (examples: {seq_examples.get(seq, [])})")

    OUT_SUMMARY_TXT.write_text("\n".join(summary_lines), encoding="utf-8")

    print(f"[DONE] wrote: {OUT_CSV}")
    print(f"[DONE] wrote: {OUT_SUMMARY_TXT}")


if __name__ == "__main__":
    main()