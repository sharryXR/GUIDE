#!/usr/bin/env python3
"""Run resumable WAA video crawling and GPT-5.2 annotation for missing tasks."""

from __future__ import annotations

import argparse
import json
import os
import queue
import re
import shutil
import shlex
import subprocess
import sys
import threading
import time
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


DEFAULT_INDEX = "evaluation_examples_windows/test_all_queries_with_videos_with_similarity.json"
DEFAULT_OUTPUT = "evaluation_examples_windows/test_all_queries_with_videos_with_similarity_gpt52.json"
DEFAULT_REPORT = "evaluation_examples_windows/test_all_queries_with_videos_with_similarity_gpt52.report.json"
DEFAULT_STATUS = "tools/video_knowledge/waa_video_annotation_status.json"
DEFAULT_MANIFEST = "tools/video_knowledge/waa_video_annotation_pending.json"
DEFAULT_ENV = ".env"
DEFAULT_VIDEO_ROOT = "."

STAGE_ORDER = ["query", "download", "cmd1", "cmd2", "cmd3"]
VIDEO_SUFFIXES = (".mp4", ".avi", ".mkv", ".mov")


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_task_id(task_id: Optional[str]) -> str:
    return re.sub(r"(?i)-wos$", "", (task_id or "").strip())


def safe_component(value: str, fallback: str = "task") -> str:
    value = re.sub(r"\s+", "_", (value or "").strip())
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value)
    value = value.strip("._-")
    return value or fallback


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json_atomic(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")
    tmp.replace(path)


def has_complete_video_annotation(entry: Dict[str, Any]) -> bool:
    return (
        int(entry.get("video_count") or 0) > 0
        and bool(str(entry.get("planning_results") or "").strip())
        and bool(str(entry.get("grounding_results") or "").strip())
    )


def load_env_file(path: Path) -> Dict[str, str]:
    env: Dict[str, str] = {}
    if not path.exists():
        return env
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            env[key] = value
    return env


def build_env(args: argparse.Namespace, task: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
    env = os.environ.copy()
    env.update(load_env_file(Path(args.env_file)))
    if env.get("OPENAI_API_KEY_2"):
        env["OPENAI_API_KEY"] = env["OPENAI_API_KEY_2"]
    env["VIDEO_LLM_MODEL"] = args.model
    env["VIDEO_SEARCH_MODEL"] = args.model
    env["VIDEO_GUI_MODEL"] = args.model
    env["VIDEO_ANNOTATION_MODEL"] = args.model
    env["PYTHONUNBUFFERED"] = "1"
    node_path = shutil.which("node")
    env.setdefault("YTDLP_JS_RUNTIME", f"node:{node_path}" if node_path else "node")
    env.setdefault("YTDLP_EXTRACTOR_ARGS", "youtube:player_client=default,android,web")
    env.setdefault("YTDLP_VIDEO_FORMAT", "bestvideo*[height<=1080]+bestaudio/best[height<=1080]/best")
    env.setdefault("YTDLP_SOCKET_TIMEOUT", "30")
    env.setdefault("YTDLP_CONCURRENT_FRAGMENTS", "4")
    env.setdefault("YTDLP_RETRIES", "2")
    env.setdefault("YTDLP_FRAGMENT_RETRIES", "2")
    env.setdefault("YTDLP_EXTRACTOR_RETRIES", "2")
    env.setdefault("YTDLP_DOWNLOADER", "aria2c")
    env.setdefault("YTDLP_DOWNLOADER_ARGS", "aria2c:-x 8 -s 8 -k 1M")
    env.setdefault("YTDLP_REMOTE_COMPONENTS", "")
    env.setdefault("YTDLP_METADATA_TIMEOUT", "35")
    env.setdefault("YTDLP_SUBTITLE_TIMEOUT", "30")
    env.setdefault("YTDLP_VIDEO_TIMEOUT", "1800")
    env.setdefault("VIDEO_LLM_TIMEOUT", "90")
    env.setdefault("VIDEO_LLM_MAX_RETRIES", "2")
    env.setdefault("VIDEO_METADATA_SCAN_LIMIT", "5")
    env.setdefault("VIDEO_DOWNLOAD_CANDIDATES", "3")
    env.setdefault("VIDEO_SUCCESS_TARGET", "1")
    env.setdefault("VIDEO_SKIP_AUDIO_DOWNLOAD", "1")
    env.setdefault("VIDEO_SKIP_SUBTITLE_SELECT", "1")
    env.setdefault("VIDEO_SKIP_GUI_FILTER", "0")
    env.setdefault("VIDEO_MIN_TOP_SCORE", "0.35")
    env.setdefault("VIDEO_MIN_EXTRA_SCORE", "0.5")
    env.setdefault("VIDEO_ALLOW_METADATA_FALLBACK", "0")
    env["PYTHONPATH"] = (
        f"{args.video_root}:{env.get('PYTHONPATH', '')}"
        if env.get("PYTHONPATH")
        else args.video_root
    )

    cookie_file = choose_cookie_file(Path(args.video_root))
    if cookie_file:
        env["VIDEO_COOKIE_FILE"] = str(cookie_file)

    if task:
        env["VIDEO_WEB"] = task["web"]
        env["VIDEO_QUERY"] = task["query"]
        env["VIDEO_TASK_DIR"] = task["task_dir"]
        env["VIDEO_URL_STEM"] = task["url_stem"]
        env["VIDEO_MAX_RESULTS"] = str(args.max_results)
        env["VIDEO_MAX_SELECTED"] = str(args.max_selected)
        env["VIDEO_SEARCH_METHOD"] = args.search_method
    return env


def choose_cookie_file(video_root: Path) -> Optional[Path]:
    candidates = [
        video_root / "cookies.txt",
        video_root / "cookies2.txt",
        video_root / "cookies3.txt",
        video_root / "cookies4.txt",
    ]
    for path in candidates:
        if path.exists() and path.stat().st_size > 0:
            return path
    return None


def make_task_dir(entry: Dict[str, Any]) -> str:
    canonical = canonical_task_id(str(entry.get("canonical_id") or entry.get("id") or ""))
    domain = str(entry.get("domain") or "waa")
    prefix = canonical[:8] if canonical else str(entry.get("id") or "")[:8]
    return safe_component(f"{domain}_{prefix}")


def output_paths(video_root: Path, task: Dict[str, Any], model: str) -> Dict[str, str]:
    base = video_root / "videos" / task["web"] / task["task_dir"]
    labeled = base / f"Labeled_{model}"
    return {
        "base": str(base),
        "video": str(base / "video"),
        "audio": str(base / "audio"),
        "subtitle": str(base / "subtitle"),
        "audios_text": str(base / "audios_text"),
        "keyframes_sumvideo_time": str(base / "keyframes_sumvideo_time"),
        "omniparser": str(base / "OmniParser_Pic"),
        "labeled": str(labeled),
        "planning": str(labeled / "devided" / "planning"),
        "grounding": str(labeled / "devided" / "grounding"),
    }


def make_record(entry: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    task_dir = make_task_dir(entry)
    query = str(entry.get("query") or "").strip()
    task = {
        "id": str(entry.get("id") or ""),
        "canonical_id": canonical_task_id(str(entry.get("canonical_id") or entry.get("id") or "")),
        "domain": str(entry.get("domain") or ""),
        "web": str(entry.get("web") or entry.get("domain") or ""),
        "instruction": str(entry.get("instruction") or ""),
        "query": query,
        "task_dir": task_dir,
        "url_stem": task_dir,
        "source": str(entry.get("source") or ""),
        "status": "pending",
        "stage_status": {},
        "retries": {},
        "error": "",
        "updated_at": now_iso(),
        "counts": {},
        "video_urls": [],
        "output_paths": {},
    }
    task["output_paths"] = output_paths(Path(args.video_root), task, args.model)
    return task


def load_tasks(args: argparse.Namespace) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    index = load_json(Path(args.index))
    pending_entries = [entry for entry in index if not has_complete_video_annotation(entry)]

    existing_by_id: Dict[str, Dict[str, Any]] = {}
    if not args.no_resume and Path(args.status).exists():
        status = load_json(Path(args.status))
        existing_by_id = {str(task.get("id")): task for task in status.get("tasks", [])}

    tasks = []
    for entry in pending_entries:
        fresh = make_record(entry, args)
        existing = existing_by_id.get(fresh["id"])
        if existing:
            merged = deepcopy(existing)
            for key in ("canonical_id", "domain", "web", "instruction", "source", "task_dir", "url_stem"):
                merged[key] = fresh[key]
            if fresh["query"] and not merged.get("query"):
                merged["query"] = fresh["query"]
            merged["output_paths"] = output_paths(Path(args.video_root), merged, args.model)
            tasks.append(merged)
        else:
            tasks.append(fresh)

    if args.task_id:
        wanted = set(args.task_id)
        tasks = [task for task in tasks if task["id"] in wanted or task["canonical_id"] in wanted]
    if args.limit:
        tasks = tasks[: args.limit]
    return index, tasks


def write_status(args: argparse.Namespace, tasks: Sequence[Dict[str, Any]]) -> None:
    if (args.task_id or args.limit) and Path(args.status).exists():
        try:
            existing_status = load_json(Path(args.status))
            existing_tasks = existing_status.get("tasks", [])
        except Exception:
            existing_tasks = []
        task_by_id = {str(task.get("id")): task for task in existing_tasks}
        for task in tasks:
            task_by_id[str(task.get("id"))] = task
        tasks = list(task_by_id.values())

    payload = {
        "source_index": args.index,
        "model": args.model,
        "updated_at": now_iso(),
        "tasks": list(tasks),
    }
    write_json_atomic(Path(args.status), payload)
    manifest = [
        {
            "id": task["id"],
            "domain": task["domain"],
            "web": task["web"],
            "instruction": task["instruction"],
            "query": task.get("query", ""),
            "task_dir": task.get("task_dir", ""),
            "url_stem": task.get("url_stem", ""),
            "status": task.get("status", ""),
            "stage_status": task.get("stage_status", {}),
            "error": task.get("error", ""),
            "output_paths": task.get("output_paths", {}),
        }
        for task in tasks
    ]
    write_json_atomic(Path(args.manifest), manifest)


def stage_done(task: Dict[str, Any], stage: str) -> bool:
    return task.get("stage_status", {}).get(stage) == "completed"


def mark_stage(task: Dict[str, Any], stage: str, status: str, error: str = "") -> None:
    task.setdefault("stage_status", {})[stage] = status
    if status == "completed":
        task["status"] = "completed" if all(stage_done(task, s) for s in STAGE_ORDER) else "in_progress"
    else:
        task["status"] = status
    task["error"] = error
    task["updated_at"] = now_iso()
    if status == "failed":
        retries = task.setdefault("retries", {})
        retries[stage] = int(retries.get(stage) or 0) + 1


def should_skip(args: argparse.Namespace, task: Dict[str, Any], stage: str) -> bool:
    if stage in args.force_stage:
        return False
    if stage_done(task, stage):
        return True
    if task.get("stage_status", {}).get(stage) == "failed" and not args.retry_failed:
        return True
    return False


def fallback_query(task: Dict[str, Any]) -> str:
    web = task["web"].replace("_", " ").replace("-", " ")
    instruction = re.sub(r"\s+", " ", task["instruction"]).strip()
    instruction = re.sub(r"[\"'`]", "", instruction)
    words = instruction.split()
    concise = " ".join(words[:14])
    return f"{web} Windows tutorial {concise}".strip()


def clean_query(text: str) -> str:
    text = (text or "").strip()
    text = re.sub(r"^```(?:text)?|```$", "", text).strip()
    text = text.splitlines()[0].strip()
    text = text.strip(" \"'`")
    text = re.sub(r"\s+", " ", text)
    return text[:180]


def call_gpt_query(args: argparse.Namespace, env: Dict[str, str], task: Dict[str, Any]) -> str:
    try:
        from openai import OpenAI
    except Exception as exc:
        raise RuntimeError(f"openai package unavailable: {exc}") from exc

    api_key = env.get("OPENAI_API_KEY_2") or env.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY_2/OPENAI_API_KEY is missing")

    client = OpenAI(api_key=api_key, base_url=env.get("OPENAI_BASE_URL") or None)
    messages = [
        {
            "role": "system",
            "content": (
                "Create concise English YouTube search queries for Windows GUI "
                "software tutorials. Return one query only, no quotes."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Application/domain: {task['web']} / {task['domain']}\n"
                f"Instruction: {task['instruction']}\n"
                "Write a 6-14 word YouTube search query that would find a practical GUI tutorial. "
                "Ignore benchmark-specific placeholder names such as sample users, file names, "
                "folder names, or exact labels; keep the general Windows/app operation."
            ),
        },
    ]
    try:
        response = client.chat.completions.create(
            model=args.model,
            messages=messages,
            max_completion_tokens=80,
        )
    except Exception:
        response = client.chat.completions.create(
            model=args.model,
            messages=messages,
            max_tokens=80,
        )
    return clean_query(response.choices[0].message.content or "")


def run_query_stage(args: argparse.Namespace, task: Dict[str, Any]) -> None:
    if task.get("query") and "query" not in args.force_stage:
        mark_stage(task, "query", "completed")
        return
    env = build_env(args, task)
    try:
        query = call_gpt_query(args, env, task)
    except Exception as exc:
        query = ""
        task["query_generation_error"] = str(exc)
    if not query:
        query = fallback_query(task)
        task["query_generated_by"] = "fallback"
    else:
        task["query_generated_by"] = args.model
    task["query"] = query
    mark_stage(task, "query", "completed")


def count_files(path: Path, suffixes: Optional[Sequence[str]] = None) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        if suffixes is None or path.name.lower().endswith(tuple(suffixes)):
            return 1
        return 0
    count = 0
    for child in path.rglob("*"):
        if child.is_file() and (suffixes is None or child.name.lower().endswith(tuple(suffixes))):
            count += 1
    return count


def update_counts(args: argparse.Namespace, task: Dict[str, Any]) -> None:
    paths = output_paths(Path(args.video_root), task, args.model)
    task["output_paths"] = paths
    task["counts"] = {
        "video": count_files(Path(paths["video"]), VIDEO_SUFFIXES),
        "audios_text": count_files(Path(paths["audios_text"]), (".txt", ".vtt")),
        "keyframes_sumvideo_time": count_files(Path(paths["keyframes_sumvideo_time"]), (".png", ".jpg", ".jpeg")),
        "omniparser": count_files(Path(paths["omniparser"]), (".png", ".txt")),
        "planning": count_files(Path(paths["planning"]), ("_planning.txt",)),
        "grounding": count_files(Path(paths["grounding"]), ("_grounding.txt",)),
    }


def command_log_path(args: argparse.Namespace, task: Dict[str, Any], stage: str) -> Path:
    log_dir = Path(args.status).resolve().parent / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir / f"{task['task_dir']}.{stage}.log"


def tail(path: Path, max_chars: int = 4000) -> str:
    if not path.exists():
        return ""
    data = path.read_text(encoding="utf-8", errors="replace")
    return data[-max_chars:]


def run_shell(args: argparse.Namespace, task: Dict[str, Any], stage: str, command: str, timeout: int) -> None:
    log_path = command_log_path(args, task, stage)
    env = build_env(args, task)
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n\n===== {now_iso()} stage={stage} task={task['id']} =====\n")
        try:
            result = subprocess.run(
                command,
                shell=True,
                executable="/bin/bash",
                cwd=args.video_root,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(f"{stage} timed out after {timeout}s; log={log_path}") from exc
    if result.returncode != 0:
        raise RuntimeError(f"{stage} failed with exit code {result.returncode}; log tail:\n{tail(log_path)}")


def conda_prefix(args: argparse.Namespace, env_name: str) -> str:
    conda_sh = Path(args.conda_base).expanduser() / "etc/profile.d/conda.sh"
    return f"source {shlex.quote(str(conda_sh))} && conda activate {shlex.quote(env_name)}"


def run_download_stage(
    args: argparse.Namespace,
    task: Dict[str, Any],
    previous_status: Optional[str] = None,
) -> None:
    if not task.get("query"):
        raise RuntimeError("download requires a query; run query stage first")
    refresh_download = (
        "download" in args.force_stage
        or (
            args.retry_failed
            and previous_status in {"failed", "running"}
        )
    )
    refresh_media = "download" in args.force_stage or (
        args.retry_failed and previous_status == "failed"
    )
    command = f"""{conda_prefix(args, "video_self_learning")} && cd {shlex.quote(args.video_root)} && python -u - <<'PY'
import os
import shutil
from pathlib import Path
from video_path_utils import get_url_stem
from youtube import save_video_urls, select_video_urls, run_get_video, llm3

web = os.environ["VIDEO_WEB"]
query = os.environ["VIDEO_QUERY"]
max_results = int(os.environ.get("VIDEO_MAX_RESULTS", "50"))
max_selected = int(os.environ.get("VIDEO_MAX_SELECTED", "3"))
method = os.environ.get("VIDEO_SEARCH_METHOD", "ytdlp")
cookies = os.environ.get("VIDEO_COOKIE_FILE", "cookies.txt")
url_dir = os.path.join("./urls", web)
os.makedirs(url_dir, exist_ok=True)
url_file = os.path.join(url_dir, f"{{get_url_stem(query)}}.txt")
selected_file = os.path.join(url_dir, f"{{get_url_stem(query)}}_selected.txt")
if {repr(refresh_download)}:
    for stale_file in (url_file, selected_file):
        if os.path.exists(stale_file):
            os.remove(stale_file)
if {repr(refresh_media)}:
    base = Path("./videos") / web / os.environ["VIDEO_TASK_DIR"]
    root = Path(".").resolve()
    for stale_dir in ("video", "audio", "subtitle"):
        target = (base / stale_dir).resolve()
        if root in target.parents and target.exists():
            shutil.rmtree(target)

save_video_urls(web, query, max_results=max_results, method=method)
select_video_urls(llm3, web, query, url_file, os.path.join("./videos", web, os.environ["VIDEO_TASK_DIR"]), cookies, max_results=max_selected)
selected_count = run_get_video(web, query)
print(f"VIDEO_SELECTED_COUNT:{{selected_count}}")
PY"""
    run_shell(args, task, "download", command, args.download_timeout)
    update_counts(args, task)
    selected_file = Path(args.video_root) / "urls" / task["web"] / f"{task['url_stem']}_selected.txt"
    if selected_file.exists():
        task["video_urls"] = [line.strip() for line in selected_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    if int(task["counts"].get("video") or 0) <= 0:
        raise RuntimeError("download completed but no video files were found")


def run_cmd1_stage(args: argparse.Namespace, task: Dict[str, Any]) -> None:
    if "cmd1" in args.force_stage:
        base = Path(args.video_root) / "videos" / task["web"] / task["task_dir"]
        for stale_dir in ("audios_text", "keyframes", "keyframes_time", "keyframes_sumvideo", "keyframes_sumvideo_time"):
            target = (base / stale_dir).resolve()
            if Path(args.video_root).resolve() in target.parents and target.exists():
                shutil.rmtree(target)
    command = (
        f"{conda_prefix(args, 'video_self_learning')} && "
        f"cd {shlex.quote(args.video_root)} && "
        'python -u auto_catch.py --web "$VIDEO_WEB" --query "$VIDEO_QUERY" --task-dir "$VIDEO_TASK_DIR"'
    )
    run_shell(args, task, "cmd1", command, args.cmd1_timeout)
    update_counts(args, task)
    if int(task["counts"].get("keyframes_sumvideo_time") or 0) <= 0:
        raise RuntimeError("cmd1 completed but no sumvideo keyframes were found")


def run_cmd2_stage(args: argparse.Namespace, task: Dict[str, Any]) -> None:
    if "cmd2" in args.force_stage:
        target = (
            Path(args.video_root)
            / "videos"
            / task["web"]
            / task["task_dir"]
            / "OmniParser_Pic"
        ).resolve()
        if Path(args.video_root).resolve() in target.parents and target.exists():
            shutil.rmtree(target)
    command = (
        f"{conda_prefix(args, 'omni')} && "
        f"cd {shlex.quote(str(Path(args.video_root) / 'OmniParser'))} && "
        'python -u auto_omni.py --web "$VIDEO_WEB" --query "$VIDEO_QUERY" --task-dir "$VIDEO_TASK_DIR"'
    )
    run_shell(args, task, "cmd2", command, args.cmd2_timeout)
    update_counts(args, task)
    if int(task["counts"].get("omniparser") or 0) <= 0:
        raise RuntimeError("cmd2 completed but no OmniParser outputs were found")


def run_cmd3_stage(args: argparse.Namespace, task: Dict[str, Any]) -> None:
    if "cmd3" in args.force_stage:
        target = (
            Path(args.video_root)
            / "videos"
            / task["web"]
            / task["task_dir"]
            / f"Labeled_{args.model}"
        ).resolve()
        if Path(args.video_root).resolve() in target.parents and target.exists():
            shutil.rmtree(target)
    command = (
        f"{conda_prefix(args, 'video_self_learning')} && "
        f"cd {shlex.quote(args.video_root)} && "
        'python -u action_annotation4.py --web "$VIDEO_WEB" --query "$VIDEO_QUERY" '
        f"--model_name {shlex.quote(args.model)} --task-dir " + '"$VIDEO_TASK_DIR"'
    )
    run_shell(args, task, "cmd3", command, args.cmd3_timeout)
    update_counts(args, task)
    if int(task["counts"].get("planning") or 0) <= 0 or int(task["counts"].get("grounding") or 0) <= 0:
        raise RuntimeError("cmd3 completed but planning/grounding files were not found")


def run_stage(
    args: argparse.Namespace,
    task: Dict[str, Any],
    stage: str,
    on_update: Optional[Any] = None,
) -> None:
    if should_skip(args, task, stage):
        return
    previous_status = task.get("stage_status", {}).get(stage)
    mark_stage(task, stage, "running")
    if on_update:
        on_update()
    try:
        if stage == "query":
            run_query_stage(args, task)
        elif stage == "download":
            run_download_stage(args, task, previous_status=previous_status)
        elif stage == "cmd1":
            run_cmd1_stage(args, task)
        elif stage == "cmd2":
            run_cmd2_stage(args, task)
        elif stage == "cmd3":
            run_cmd3_stage(args, task)
        else:
            raise ValueError(f"unknown stage: {stage}")
    except Exception as exc:
        mark_stage(task, stage, "failed", str(exc))
        if on_update:
            on_update()
        raise
    else:
        update_counts(args, task)
        mark_stage(task, stage, "completed")
        if on_update:
            on_update()


def next_pipeline_stage(
    args: argparse.Namespace,
    task: Dict[str, Any],
    forced_done: set[Tuple[str, str]],
) -> Optional[str]:
    for stage in STAGE_ORDER:
        if not stage_done(task, stage):
            if task.get("stage_status", {}).get(stage) == "failed" and not args.retry_failed:
                return None
            return stage
        force_key = (task["id"], stage)
        if stage in args.force_stage and force_key not in forced_done:
            return stage
    return None


def run_pipeline(args: argparse.Namespace, tasks: Sequence[Dict[str, Any]]) -> int:
    worker_counts = {
        "query": args.query_workers,
        "download": args.download_workers,
        "cmd1": args.cmd1_workers,
        "cmd2": args.cmd2_workers,
        "cmd3": args.cmd3_workers,
    }
    queues = {stage: queue.Queue() for stage in STAGE_ORDER}
    forced_done: set[Tuple[str, str]] = set()
    failures: List[Tuple[str, str, str]] = []
    status_lock = threading.Lock()
    state_lock = threading.Lock()
    print_lock = threading.Lock()
    active = {"count": 0}

    def save_status() -> None:
        with status_lock:
            write_status(args, tasks)

    def log(message: str, *, stderr: bool = False) -> None:
        with print_lock:
            print(message, file=sys.stderr if stderr else sys.stdout, flush=True)

    def worker(stage: str, worker_index: int) -> None:
        while True:
            item = queues[stage].get()
            if item is None:
                queues[stage].task_done()
                break
            task = item
            with state_lock:
                active["count"] += 1
            try:
                log(f"[pipeline:{stage}:{worker_index}] start {task['id']}")
                run_stage(args, task, stage, on_update=save_status)
                if stage in args.force_stage:
                    forced_done.add((task["id"], stage))
                update_counts(args, task)
                next_stage = next_pipeline_stage(args, task, forced_done)
                if next_stage:
                    queues[next_stage].put(task)
                    log(f"[pipeline:{stage}:{worker_index}] completed {task['id']} -> {next_stage}")
                else:
                    log(f"[pipeline:{stage}:{worker_index}] completed {task['id']}")
            except Exception as exc:
                failures.append((task["id"], stage, str(exc)))
                log(f"[pipeline:{stage}:{worker_index}] failed {task['id']}: {exc}", stderr=True)
            finally:
                save_status()
                with state_lock:
                    active["count"] -= 1
                queues[stage].task_done()

    threads = []
    for stage, count in worker_counts.items():
        for worker_index in range(max(0, count)):
            thread = threading.Thread(
                target=worker,
                args=(stage, worker_index + 1),
                name=f"waa-{stage}-{worker_index + 1}",
                daemon=True,
            )
            thread.start()
            threads.append(thread)

    enqueued = 0
    for task in tasks:
        update_counts(args, task)
        stage = next_pipeline_stage(args, task, forced_done)
        if stage:
            queues[stage].put(task)
            enqueued += 1
    save_status()
    log(
        "Pipeline started with "
        + ", ".join(f"{stage}={worker_counts[stage]}" for stage in STAGE_ORDER)
        + f"; enqueued {enqueued} task(s)."
    )

    while True:
        time.sleep(max(1, args.pipeline_status_interval))
        queued = sum(q.unfinished_tasks for q in queues.values())
        with state_lock:
            active_count = active["count"]
        status_counter = {}
        for task in tasks:
            status_counter[task.get("status", "")] = status_counter.get(task.get("status", ""), 0) + 1
        log(
            f"[pipeline] queued={queued} active={active_count} "
            f"completed={status_counter.get('completed', 0)} "
            f"in_progress={status_counter.get('in_progress', 0)} "
            f"pending={status_counter.get('pending', 0)} "
            f"failed={status_counter.get('failed', 0)}"
        )
        if queued == 0 and active_count == 0:
            break

    for q in queues.values():
        for _ in threads:
            q.put(None)
    for thread in threads:
        thread.join(timeout=5)

    if failures:
        log(f"Pipeline finished with {len(failures)} failed stage(s).", stderr=True)
        for task_id, stage, error in failures[:20]:
            log(f"  {task_id} {stage}: {error[:300]}", stderr=True)
    else:
        log("Pipeline finished without stage failures.")
    return len(failures)


def read_annotation_dir(path: Path, suffix: str, label: str) -> Tuple[str, int]:
    if not path.exists():
        return "", 0
    chunks = []
    files = sorted(file for file in path.iterdir() if file.is_file() and file.name.endswith(suffix))
    for index, file in enumerate(files, 1):
        content = file.read_text(encoding="utf-8", errors="replace").strip()
        if not content:
            continue
        task_name = file.name.replace(suffix, "").split("~~")[0]
        chunks.append(f"The {label} trajectory of Demo {index}: {task_name}:\n{content}\n")
    return "".join(chunks), len(chunks)


def merge_results(args: argparse.Namespace, index: Sequence[Dict[str, Any]], tasks: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    tasks_by_id = {task["id"]: task for task in tasks}
    output_path = Path(args.output)
    if output_path.exists():
        existing_output = load_json(output_path)
        output = deepcopy(existing_output) if len(existing_output) == len(index) else deepcopy(list(index))
    else:
        output = deepcopy(list(index))
    updated = 0
    failed = []

    for entry in output:
        task = tasks_by_id.get(str(entry.get("id") or ""))
        if not task:
            continue
        update_counts(args, task)
        paths = output_paths(Path(args.video_root), task, args.model)
        planning, planning_count = read_annotation_dir(Path(paths["planning"]), "_planning.txt", "planning")
        grounding, grounding_count = read_annotation_dir(Path(paths["grounding"]), "_grounding.txt", "grounding")
        video_count = int(task.get("counts", {}).get("video") or 0)
        if video_count > 0 and planning and grounding:
            entry["query"] = task["query"]
            entry["video_count"] = video_count
            entry["converted_video_count"] = min(video_count, planning_count, grounding_count)
            entry["planning_results"] = planning
            entry["grounding_results"] = grounding
            entry["previous_source"] = entry.get("source", "")
            entry["source"] = "waa_gpt52_video_crawl"
            entry["annotation_model"] = args.model
            entry["video_task_dir"] = task["task_dir"]
            entry["video_url_stem"] = task["url_stem"]
            updated += 1
        else:
            failed.append(
                {
                    "id": task["id"],
                    "stage_status": task.get("stage_status", {}),
                    "error": task.get("error", ""),
                    "counts": task.get("counts", {}),
                }
            )

    write_json_atomic(Path(args.output), output)
    complete = sum(1 for entry in output if has_complete_video_annotation(entry))
    report = {
        "source_index": args.index,
        "output": args.output,
        "model": args.model,
        "total": len(output),
        "complete_after_merge": complete,
        "newly_updated": updated,
        "remaining_in_selected_tasks": failed,
        "updated_at": now_iso(),
    }
    write_json_atomic(Path(args.report), report)
    return report


def api_smoke(args: argparse.Namespace) -> None:
    env = build_env(args)
    api_key = env.get("OPENAI_API_KEY_2") or env.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY_2/OPENAI_API_KEY is missing")
    from openai import OpenAI

    client = OpenAI(api_key=api_key, base_url=env.get("OPENAI_BASE_URL") or None)
    try:
        response = client.chat.completions.create(
            model=args.model,
            messages=[{"role": "user", "content": "Return exactly: ok"}],
            max_completion_tokens=8,
        )
    except Exception:
        response = client.chat.completions.create(
            model=args.model,
            messages=[{"role": "user", "content": "Return exactly: ok"}],
            max_tokens=8,
        )
    print(f"API smoke response: {response.choices[0].message.content}")


def config() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", default=DEFAULT_INDEX)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--report", default=DEFAULT_REPORT)
    parser.add_argument("--status", default=DEFAULT_STATUS)
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--env-file", default=DEFAULT_ENV)
    parser.add_argument("--video-root", default=DEFAULT_VIDEO_ROOT)
    parser.add_argument("--conda-base", default="~/anaconda3")
    parser.add_argument("--model", default="gpt-5.2")
    parser.add_argument("--stage", choices=STAGE_ORDER + ["merge", "all", "api-smoke"], default="all")
    parser.add_argument("--task-id", action="append", help="Limit to a WAA id or canonical id. Repeatable.")
    parser.add_argument("--limit", type=int, help="Limit selected pending tasks.")
    parser.add_argument("--max-results", type=int, default=50)
    parser.add_argument("--max-selected", type=int, default=3)
    parser.add_argument("--search-method", choices=["ytdlp", "google"], default="ytdlp")
    parser.add_argument("--retry-failed", action="store_true", help="Rerun tasks whose selected stage failed.")
    parser.add_argument("--force-stage", action="append", default=[], choices=STAGE_ORDER, help="Rerun a completed stage.")
    parser.add_argument("--no-resume", action="store_true", help="Ignore existing status and rebuild task records.")
    parser.add_argument("--pipeline", action="store_true", help="Run stage workers concurrently with task dependencies.")
    parser.add_argument("--query-workers", type=int, default=1)
    parser.add_argument("--download-workers", type=int, default=1)
    parser.add_argument("--cmd1-workers", type=int, default=1)
    parser.add_argument("--cmd2-workers", type=int, default=1)
    parser.add_argument("--cmd3-workers", type=int, default=1)
    parser.add_argument("--pipeline-status-interval", type=int, default=60)
    parser.add_argument("--download-timeout", type=int, default=3600)
    parser.add_argument("--cmd1-timeout", type=int, default=900)
    parser.add_argument("--cmd2-timeout", type=int, default=2400)
    parser.add_argument("--cmd3-timeout", type=int, default=2400)
    return parser.parse_args()


def main() -> int:
    args = config()
    if args.stage == "api-smoke":
        api_smoke(args)
        return 0

    index, tasks = load_tasks(args)
    write_status(args, tasks)
    print(f"Loaded {len(index)} WAA entries; selected {len(tasks)} pending task(s).")

    if args.stage == "merge":
        report = merge_results(args, index, tasks)
        write_status(args, tasks)
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0

    stages = STAGE_ORDER if args.stage == "all" else [args.stage]
    failures = 0
    if args.pipeline and args.stage == "all":
        failures = run_pipeline(args, tasks)
    else:
        if args.pipeline:
            print("--pipeline is only used with --stage all; falling back to serial stage execution.", file=sys.stderr)
        for task in tasks:
            print(f"\n=== Task {task['id']} ({task['domain']}) ===")
            for stage in stages:
                try:
                    run_stage(args, task, stage)
                    write_status(args, tasks)
                    print(f"[{stage}] completed")
                except Exception as exc:
                    failures += 1
                    write_status(args, tasks)
                    print(f"[{stage}] failed: {exc}", file=sys.stderr)
                    break

    if args.stage == "all":
        report = merge_results(args, index, tasks)
        write_status(args, tasks)
        print(json.dumps(report, indent=2, ensure_ascii=False))

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
