#!/usr/bin/env python3
"""
Convert Twist2 motion recordings (JSON) to LeRobot v2 format for GR00T training.

Vision data is not included in this conversion and will be integrated later.

Usage:
    uv run python scripts/convert_twist2_to_lerobot.py \
        --input-dir episodes-twist2 \
        --output-dir datasets/twist2_lerobot \
        --task-description "twist2 motion demonstration" \
        --fps 50 \
        --episode-length 200

Input JSON format (per frame):
    {
        "timestamp": float,
        "body_dof_pos": [float x 29],   -> observation.state
        "target_dof_pos": [float x 29], -> action
        "tau": [float x 29],
        "voltage": [float x 29],
        "temperature": [[float, float] x 29]
    }
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm


DOF_COUNT = 29
CHUNK_SIZE = 100  # episodes per chunk


def load_json_recording(path: Path) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def split_into_episodes(frames: list[dict], episode_length: int, max_gap_sec: float = 1.0) -> list[list[dict]]:
    """Split a flat list of frames into episodes.

    Splits on:
    1. Time gap > max_gap_sec between consecutive frames (natural boundaries).
    2. Fixed episode_length if > 0.
    """
    if not frames:
        return []

    episodes = []
    current = [frames[0]]

    for i in range(1, len(frames)):
        dt = frames[i]["timestamp"] - frames[i - 1]["timestamp"]
        if dt > max_gap_sec:
            # Natural boundary
            if len(current) > 0:
                episodes.append(current)
            current = [frames[i]]
        else:
            current.append(frames[i])

    if current:
        episodes.append(current)

    # Further split by episode_length
    if episode_length > 0:
        split = []
        for ep in episodes:
            for start in range(0, len(ep), episode_length):
                chunk = ep[start : start + episode_length]
                if len(chunk) > 0:
                    split.append(chunk)
        episodes = split

    return episodes


def frames_to_parquet_rows(frames: list[dict], episode_index: int, global_start_index: int, task_index: int) -> list[dict]:
    t0 = frames[0]["timestamp"]
    rows = []
    for local_i, frame in enumerate(frames):
        is_last = local_i == len(frames) - 1
        rows.append({
            "observation.state": np.array(frame["body_dof_pos"], dtype=np.float64),
            "action": np.array(frame["target_dof_pos"], dtype=np.float64),
            "timestamp": frame["timestamp"] - t0,
            "annotation.human.action.task_description": task_index,
            "task_index": task_index,
            "annotation.human.validity": 1,
            "episode_index": episode_index,
            "index": global_start_index + local_i,
            "next.reward": 0.0,
            "next.done": is_last,
        })
    return rows


def compute_stats(parquet_paths: list[Path]) -> dict:
    dfs = [pd.read_parquet(p) for p in tqdm(parquet_paths, desc="Computing stats")]
    combined = pd.concat(dfs, axis=0, ignore_index=True)
    features = ["observation.state", "action", "timestamp", "next.reward"]
    stats = {}
    for feat in features:
        arr = np.vstack([np.atleast_1d(np.asarray(x, dtype=np.float64)) for x in combined[feat]])
        stats[feat] = {
            "mean": arr.mean(axis=0).tolist(),
            "std": arr.std(axis=0).tolist(),
            "min": arr.min(axis=0).tolist(),
            "max": arr.max(axis=0).tolist(),
            "q01": np.quantile(arr, 0.01, axis=0).tolist(),
            "q99": np.quantile(arr, 0.99, axis=0).tolist(),
        }
    return stats


def build_info_json(total_episodes: int, total_frames: int, fps: float, task_count: int) -> dict:
    motor_names = [f"motor_{i}" for i in range(DOF_COUNT)]
    return {
        "codebase_version": "v2.0",
        "robot_type": "Twist2",
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "total_tasks": task_count,
        "total_videos": 0,
        "total_chunks": 0,
        "chunks_size": CHUNK_SIZE,
        "fps": fps,
        "splits": {"train": f"0:{total_episodes}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": {
            "observation.state": {
                "dtype": "float64",
                "shape": [DOF_COUNT],
                "names": motor_names,
            },
            "action": {
                "dtype": "float64",
                "shape": [DOF_COUNT],
                "names": motor_names,
            },
            "timestamp": {"dtype": "float64", "shape": [1]},
            "annotation.human.action.task_description": {"dtype": "int64", "shape": [1]},
            "task_index": {"dtype": "int64", "shape": [1]},
            "annotation.human.validity": {"dtype": "int64", "shape": [1]},
            "episode_index": {"dtype": "int64", "shape": [1]},
            "index": {"dtype": "int64", "shape": [1]},
            "next.reward": {"dtype": "float64", "shape": [1]},
            "next.done": {"dtype": "bool", "shape": [1]},
        },
    }


def build_modality_json() -> dict:
    """Modality mapping for Twist2 (no vision yet)."""
    groups = {
        "body": {"start": 0, "end": DOF_COUNT},
    }
    return {
        "state": {"body": {"start": 0, "end": DOF_COUNT}},
        "action": {"body": {"start": 0, "end": DOF_COUNT}},
        "video": {},  # empty until vision is integrated
        "annotation": {
            "human.action.task_description": {},
            "human.validity": {},
        },
    }


def convert(
    input_dir: Path,
    output_dir: Path,
    task_description: str,
    fps: float,
    episode_length: int,
    max_gap_sec: float,
    min_episode_frames: int,
):
    json_files = sorted(input_dir.rglob("*.json"))
    if not json_files:
        print(f"No JSON files found in {input_dir}")
        return

    print(f"Found {len(json_files)} JSON file(s)")

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "meta").mkdir(exist_ok=True)

    all_episodes: list[list[dict]] = []
    for jf in json_files:
        print(f"  Loading {jf.name} ...")
        frames = load_json_recording(jf)
        print(f"    {len(frames)} frames loaded")
        episodes = split_into_episodes(frames, episode_length, max_gap_sec)
        # Filter out too-short episodes
        episodes = [ep for ep in episodes if len(ep) >= min_episode_frames]
        print(f"    -> {len(episodes)} episode(s)")
        all_episodes.extend(episodes)

    if not all_episodes:
        print("No valid episodes found.")
        return

    print(f"\nTotal episodes: {len(all_episodes)}")

    # Write parquet files
    parquet_paths = []
    episode_meta = []
    global_index = 0

    for ep_idx, episode in enumerate(tqdm(all_episodes, desc="Writing parquet")):
        chunk_idx = ep_idx // CHUNK_SIZE
        chunk_dir = output_dir / "data" / f"chunk-{chunk_idx:03d}"
        chunk_dir.mkdir(parents=True, exist_ok=True)

        rows = frames_to_parquet_rows(episode, ep_idx, global_index, task_index=0)
        df = pd.DataFrame(rows)
        parquet_path = chunk_dir / f"episode_{ep_idx:06d}.parquet"
        df.to_parquet(parquet_path, index=False)
        parquet_paths.append(parquet_path)

        episode_meta.append({
            "episode_index": ep_idx,
            "tasks": [task_description, "valid"],
            "length": len(episode),
        })
        global_index += len(episode)

    total_frames = global_index

    # meta/episodes.jsonl
    with open(output_dir / "meta" / "episodes.jsonl", "w") as f:
        for em in episode_meta:
            f.write(json.dumps(em) + "\n")

    # meta/tasks.jsonl
    with open(output_dir / "meta" / "tasks.jsonl", "w") as f:
        f.write(json.dumps({"task_index": 0, "task": task_description}) + "\n")
        f.write(json.dumps({"task_index": 1, "task": "valid"}) + "\n")

    # meta/info.json
    info = build_info_json(len(all_episodes), total_frames, fps, task_count=2)
    with open(output_dir / "meta" / "info.json", "w") as f:
        json.dump(info, f, indent=4)

    # meta/modality.json
    with open(output_dir / "meta" / "modality.json", "w") as f:
        json.dump(build_modality_json(), f, indent=4)

    # meta/stats.json
    print("\nComputing dataset statistics...")
    stats = compute_stats(parquet_paths)
    with open(output_dir / "meta" / "stats.json", "w") as f:
        json.dump(stats, f, indent=4)

    # meta/relative_stats.json (empty for now, not needed without action_configs)
    with open(output_dir / "meta" / "relative_stats.json", "w") as f:
        json.dump({}, f, indent=4)

    print(f"\nDone! Dataset written to: {output_dir}")
    print(f"  Episodes : {len(all_episodes)}")
    print(f"  Frames   : {total_frames}")
    print(f"  FPS      : {fps}")
    print(f"  DOF      : {DOF_COUNT}")
    print()
    print("Next steps:")
    print("  1. Create a modality config Python file (see configs/data/twist2_modality_config.py)")
    print("  2. Fine-tune with:")
    print(f"     uv run python gr00t/experiment/launch_finetune.py \\")
    print(f"       --base-model-path nvidia/GR00T-N1.6-3B \\")
    print(f"       --dataset-path {output_dir} \\")
    print(f"       --embodiment-tag NEW_EMBODIMENT \\")
    print(f"       --modality-config-path configs/data/twist2_modality_config.py \\")
    print(f"       --num-gpus 1 --output-dir <OUTPUT_PATH>")


def main():
    parser = argparse.ArgumentParser(description="Convert Twist2 JSON recordings to LeRobot v2 format")
    parser.add_argument("--input-dir", type=Path, default=Path("episodes-twist2"),
                        help="Directory containing Twist2 JSON recording files")
    parser.add_argument("--output-dir", type=Path, default=Path("datasets/twist2_lerobot"),
                        help="Output directory for the converted LeRobot v2 dataset")
    parser.add_argument("--task-description", type=str, default="twist2 motion demonstration",
                        help="Task description string for the dataset")
    parser.add_argument("--fps", type=float, default=50.0,
                        help="Recording FPS (default: 50)")
    parser.add_argument("--episode-length", type=int, default=0,
                        help="Max frames per episode (0 = no splitting by length)")
    parser.add_argument("--max-gap-sec", type=float, default=1.0,
                        help="Max time gap in seconds to allow within an episode before splitting")
    parser.add_argument("--min-episode-frames", type=int, default=10,
                        help="Minimum number of frames required to keep an episode")
    args = parser.parse_args()

    convert(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        task_description=args.task_description,
        fps=args.fps,
        episode_length=args.episode_length,
        max_gap_sec=args.max_gap_sec,
        min_episode_frames=args.min_episode_frames,
    )


if __name__ == "__main__":
    main()
