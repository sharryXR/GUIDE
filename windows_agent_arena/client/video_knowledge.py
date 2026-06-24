import json
import logging
import os
import re
from typing import Any, Dict, Optional, Tuple


logger = logging.getLogger("desktopenv.video_knowledge")


def canonical_task_id(task_id: Optional[str]) -> str:
    value = (task_id or "").strip()
    return re.sub(r"(?i)-wos$", "", value)


def task_web(example: Dict[str, Any], fallback_domain: str = "") -> str:
    related_apps = example.get("related_apps") or []
    if related_apps:
        return " + ".join(str(app) for app in related_apps)
    return fallback_domain


def truncate_grounding(grounding: Optional[str], max_k: int) -> Optional[str]:
    if not grounding or max_k <= 0:
        return grounding

    parts = re.split(r"(The grounding trajectory of Demo \d+:[^\n]*\n)", grounding)
    result = []
    for part in parts:
        if re.match(r"The grounding trajectory of Demo \d+:", part):
            result.append(part)
            continue

        matches = list(re.finditer(r"^\d+\.\s", part, flags=re.MULTILINE))
        if len(matches) > max_k:
            result.append(part[: matches[max_k].start()].rstrip() + "\n")
        else:
            result.append(part)
    return "".join(result).strip()


def _load_entries(video_json_path: str):
    if not video_json_path or not os.path.exists(video_json_path):
        logger.warning("Video knowledge JSON not found: %s", video_json_path)
        return []
    with open(video_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Video knowledge JSON must be a list: {video_json_path}")
    return data


def _has_video(entry: Dict[str, Any]) -> bool:
    return int(entry.get("video_count") or 0) > 0 and (
        bool(entry.get("planning_results")) or bool(entry.get("grounding_results"))
    )


def find_video_entry(
    video_json_path: str,
    example_id: str,
    web: str,
    instruction: str,
) -> Optional[Dict[str, Any]]:
    entries = _load_entries(video_json_path)
    if not entries:
        return None

    target_id = (example_id or "").strip()
    target_canonical_id = canonical_task_id(target_id)

    def entry_canonical(entry: Dict[str, Any]) -> str:
        return canonical_task_id(str(entry.get("canonical_id") or entry.get("id") or ""))

    for entry in entries:
        if str(entry.get("id", "")).strip() == target_id and _has_video(entry):
            return entry

    for entry in entries:
        if entry_canonical(entry) == target_canonical_id and _has_video(entry):
            return entry

    for entry in entries:
        if entry.get("web") == web and entry.get("instruction") == instruction and _has_video(entry):
            return entry

    return None


def get_task_video_knowledge(
    video_json_path: str,
    example_id: str,
    domain: str,
    example: Dict[str, Any],
    grounding_max_k: int = 0,
) -> Tuple[Optional[str], Optional[str], Dict[str, Any]]:
    web = task_web(example, domain)
    instruction = example.get("instruction", "")
    entry = find_video_entry(video_json_path, example_id, web, instruction)
    if not entry:
        return None, None, {}

    planning = entry.get("planning_results") or None
    grounding = truncate_grounding(entry.get("grounding_results") or None, grounding_max_k)
    meta = {
        "id": entry.get("id", ""),
        "canonical_id": entry.get("canonical_id") or canonical_task_id(entry.get("id", "")),
        "source": entry.get("source", ""),
        "video_count": entry.get("video_count", 0),
        "converted_video_count": entry.get("converted_video_count", 0),
    }
    return planning, grounding, meta
