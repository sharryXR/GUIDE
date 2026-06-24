#!/usr/bin/env python3
"""Build a WindowsAgentArena video-knowledge index seeded from OSWorld data."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List


def canonical_task_id(task_id: str) -> str:
    return re.sub(r"(?i)-wos$", "", (task_id or "").strip())


def task_web(example: Dict[str, Any], fallback_domain: str) -> str:
    related_apps = example.get("related_apps") or []
    if related_apps:
        return " + ".join(str(app) for app in related_apps)
    return fallback_domain


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def iter_waa_tasks(client_dir: Path, meta_path: Path, difficulty: str) -> Iterable[Dict[str, Any]]:
    meta = load_json(meta_path)
    examples_dir = "examples" if difficulty == "normal" else "examples_noctxt"

    for domain, ids in meta.items():
        for example_id in ids:
            example_path = client_dir / "evaluation_examples_windows" / examples_dir / domain / f"{example_id}.json"
            example = load_json(example_path)
            yield {
                "id": example_id,
                "canonical_id": canonical_task_id(example_id),
                "domain": domain,
                "web": task_web(example, domain),
                "instruction": example.get("instruction", ""),
            }


def index_osworld(entries: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    by_canonical: Dict[str, Dict[str, Any]] = {}
    for entry in entries:
        entry_id = str(entry.get("id", ""))
        key = canonical_task_id(str(entry.get("canonical_id") or entry_id))
        if key and key not in by_canonical:
            by_canonical[key] = entry
    return by_canonical


def build_index(args: argparse.Namespace) -> List[Dict[str, Any]]:
    client_dir = Path(args.waa_client_dir).resolve()
    meta_path = Path(args.test_all_meta_path)
    if not meta_path.is_absolute():
        meta_path = client_dir / meta_path

    osworld_entries = load_json(Path(args.osworld_converted).resolve())
    osworld_by_canonical = index_osworld(osworld_entries)

    output: List[Dict[str, Any]] = []
    matched = 0
    with_video = 0

    for task in iter_waa_tasks(client_dir, meta_path, args.difficulty):
        source = osworld_by_canonical.get(task["canonical_id"])
        if source:
            matched += 1
            video_count = int(source.get("video_count") or 0)
            converted_video_count = int(source.get("converted_video_count") or 0)
            planning_results = source.get("planning_results", "")
            grounding_results = source.get("grounding_results", "")
            query = source.get("query", "")
            source_label = "osworld_canonical_id"
            if video_count > 0 and (planning_results or grounding_results):
                with_video += 1
        else:
            video_count = 0
            converted_video_count = 0
            planning_results = ""
            grounding_results = ""
            query = ""
            source_label = "missing"

        output.append(
            {
                "id": task["id"],
                "canonical_id": task["canonical_id"],
                "domain": task["domain"],
                "web": task["web"],
                "instruction": task["instruction"],
                "query": query,
                "video_count": video_count,
                "converted_video_count": converted_video_count,
                "planning_results": planning_results,
                "grounding_results": grounding_results,
                "source": source_label,
            }
        )

    print(
        "Built WAA video index: "
        f"tasks={len(output)} matched_osworld={matched} with_video={with_video}"
    )
    return output


def config() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--waa-client-dir",
        default=".",
        help="WindowsAgentArena client directory. Defaults to the current directory.",
    )
    parser.add_argument(
        "--test-all-meta-path",
        default="evaluation_examples_windows/test_all.json",
        help="Task meta JSON relative to the WAA client dir or absolute path.",
    )
    parser.add_argument(
        "--difficulty",
        choices=["normal", "hard"],
        default="normal",
        help="Which WAA example directory to read.",
    )
    parser.add_argument(
        "--osworld-converted",
        required=True,
        help="OSWorld converted video-knowledge JSON.",
    )
    parser.add_argument(
        "--output",
        default="evaluation_examples_windows/test_all_queries_with_videos_with_converted.json",
        help="Output WAA video-knowledge JSON. Relative paths are resolved from the current directory.",
    )
    return parser.parse_args()


def main() -> None:
    args = config()
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = Path(args.waa_client_dir).resolve() / output_path
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = build_index(args)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
