#!/usr/bin/env python3
"""Install GUIDE's WindowsAgentArena adapter files into a WAA client checkout."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PACKAGE_DIR = Path(__file__).resolve().parent


def copy_file(src: Path, dst: Path, overwrite: bool) -> None:
    if dst.exists() and not overwrite:
        print(f"skip existing file: {dst}")
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    print(f"installed file: {dst}")


def copy_tree(src: Path, dst: Path, overwrite: bool) -> None:
    if dst.exists():
        if not overwrite:
            print(f"skip existing directory: {dst}")
            return
        shutil.rmtree(dst)
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    print(f"installed directory: {dst}")


def replace_once(text: str, old: str, new: str, label: str) -> str:
    if new in text:
        return text
    if old not in text:
        raise RuntimeError(f"Could not find patch anchor for {label}")
    return text.replace(old, new, 1)


def normalize_lines(text: str) -> str:
    return "\n".join(line.rstrip() for line in text.splitlines()) + "\n"


def patch_run_py(path: Path) -> None:
    text = normalize_lines(path.read_text(encoding="utf-8"))

    text = replace_once(
        text,
        '    parser.add_argument("--agent_name", type=str, default="navi")',
        '    parser.add_argument("--agent_name", "--agent", dest="agent_name", type=str, default="navi")',
        "run.py agent alias",
    )
    text = replace_once(
        text,
        '    parser.add_argument("--max_trajectory_length", type=int, default=3)',
        '    parser.add_argument("--max_trajectory_length", type=int, default=8)',
        "run.py trajectory length",
    )
    text = replace_once(
        text,
        '    parser.add_argument("--stop_token", type=str, default=None)\n',
        '''    parser.add_argument("--stop_token", type=str, default=None)
    parser.add_argument("--coord", type=str, choices=["absolute", "relative"], default="relative")
    parser.add_argument("--api_backend", type=str, choices=["openai", "dashscope", "local"], default="openai")
    parser.add_argument("--enable_thinking", action="store_true", default=False)
    parser.add_argument("--thinking_budget", type=int, default=8192)
    parser.add_argument("--video_json", type=str, default=None)
    parser.add_argument("--enable_planning", action="store_true", default=False)
    parser.add_argument("--enable_grounding", action="store_true", default=False)
    parser.add_argument("--grounding_max_k", type=int, default=0)
''',
        "run.py GUIDE args",
    )
    text = replace_once(
        text,
        '        "stop_token": args.stop_token,\n',
        '''        "stop_token": args.stop_token,
        "coord": args.coord,
        "api_backend": args.api_backend,
        "enable_thinking": args.enable_thinking,
        "thinking_budget": args.thinking_budget,
        "video_json": args.video_json,
        "enable_planning": args.enable_planning,
        "enable_grounding": args.enable_grounding,
        "grounding_max_k": args.grounding_max_k,
''',
        "run.py cfg args",
    )
    text = replace_once(
        text,
        '    if cfg_args["agent_name"] == "navi":\n',
        '    agent = None\n    agent_action_space = None\n\n    if cfg_args["agent_name"] == "navi":\n',
        "run.py agent action-space init",
    )
    text = replace_once(
        text,
        '''        agent = NaviAgent(
            server="oai",
            model=args.model,
            som_config=som_config,
            som_origin=args.som_origin,
            temperature=args.temperature
        )
    elif cfg_args["agent_name"] == "claude":
        from mm_agents.claude.agent import ClaudeAgent
        agent = ClaudeAgent()
    else:
        raise ValueError(f"Unknown agent name: {cfg_args['agent_name']}")
''',
        '''        agent = NaviAgent(
            server="oai",
            model=args.model,
            som_config=som_config,
            som_origin=args.som_origin,
            temperature=args.temperature
        )
        agent_action_space = agent.action_space
    elif cfg_args["agent_name"] == "claude":
        from mm_agents.claude.agent import ClaudeAgent
        agent = ClaudeAgent()
        agent_action_space = agent.action_space
    elif cfg_args["agent_name"] in ["qwen3vl", "qwen3vl_vrag"]:
        from mm_agents.qwen3vl_vrag import Qwen3VLVragAgent

        agent = Qwen3VLVragAgent(
            platform="windows",
            model=args.model,
            max_tokens=args.max_tokens,
            top_p=args.top_p,
            temperature=args.temperature,
            action_space="pyautogui",
            observation_type="screenshot",
            coordinate_type=args.coord,
            api_backend=args.api_backend,
            enable_thinking=args.enable_thinking,
            thinking_budget=args.thinking_budget,
            video_json=args.video_json,
        )
        agent_action_space = agent.action_space
    elif cfg_args["agent_name"] == "agent_s3":
        agent_action_space = "pyautogui"
    else:
        raise ValueError(f"Unknown agent name: {cfg_args['agent_name']}")
''',
        "run.py GUIDE agents",
    )
    text = replace_once(
        text,
        "        action_space=agent.action_space,\n",
        "        action_space=agent_action_space,\n",
        "run.py env action-space",
    )
    text = replace_once(
        text,
        '''    for domain in tqdm(test_all_meta, desc="Domain"):
''',
        '''    if cfg_args["agent_name"] == "agent_s3":
        from mm_agents.agent_s3 import AgentS3WaaAgent

        agent = AgentS3WaaAgent(
            env=env,
            model=args.model,
            base_url=os.getenv("OPENAI_BASE_URL"),
            api_key=os.getenv("OPENAI_API_KEY_2") or os.getenv("OPENAI_API_KEY"),
            temperature=args.temperature,
            max_trajectory_length=args.max_trajectory_length,
            screen_width=args.screen_width,
            screen_height=args.screen_height,
            grounding_width=args.screen_width,
            grounding_height=args.screen_height,
            video_json=args.video_json,
            enable_reflection=True,
        )

    for domain in tqdm(test_all_meta, desc="Domain"):
''',
        "run.py AgentS3 construction",
    )
    text = replace_once(
        text,
        '''            logger.info(f"[Instruction]: {instruction}")
''',
        '''            logger.info(f"[Instruction]: {instruction}")
            if hasattr(agent, "set_task_context"):
                agent.set_task_context(domain=domain, example_id=example_id, example=example, args=args)
''',
        "run.py task context",
    )

    path.write_text(text, encoding="utf-8")
    print(f"patched file: {path}")


def patch_start_client(path: Path) -> None:
    text = normalize_lines(path.read_text(encoding="utf-8"))
    text = replace_once(
        text,
        'diff_lvl="normal"\n',
        '''diff_lvl="normal"
max_tokens="8192"
temperature="1.0"
max_steps="15"
coord="relative"
api_backend="openai"
enable_thinking=false
thinking_budget="8192"
video_json=""
enable_planning=false
enable_grounding=false
grounding_max_k="0"
''',
        "start_client defaults",
    )
    text = replace_once(
        text,
        '''        --diff-lvl)
            diff_lvl=$2
            shift 2
            ;;
''',
        '''        --diff-lvl)
            diff_lvl=$2
            shift 2
            ;;
        --max-tokens)
            max_tokens=$2
            shift 2
            ;;
        --temperature)
            temperature=$2
            shift 2
            ;;
        --max-steps)
            max_steps=$2
            shift 2
            ;;
        --coord)
            coord=$2
            shift 2
            ;;
        --api-backend)
            api_backend=$2
            shift 2
            ;;
        --enable-thinking)
            enable_thinking=$2
            shift 2
            ;;
        --thinking-budget)
            thinking_budget=$2
            shift 2
            ;;
        --video-json)
            video_json=$2
            shift 2
            ;;
        --enable-planning)
            enable_planning=$2
            shift 2
            ;;
        --enable-grounding)
            enable_grounding=$2
            shift 2
            ;;
        --grounding-max-k)
            grounding_max_k=$2
            shift 2
            ;;
''',
        "start_client args",
    )
    text = replace_once(
        text,
        '''python run.py --agent "$agent" --model "$model" --som_origin "$som_origin" --a11y_backend "$a11y_backend" --worker_id "$worker_id" --num_workers "$num_workers" --result_dir "$result_dir" --test_all_meta_path "$json_name" --diff_lvl "$diff_lvl"
''',
        '''run_args=(
    python run.py
    --agent_name "$agent"
    --model "$model"
    --som_origin "$som_origin"
    --a11y_backend "$a11y_backend"
    --worker_id "$worker_id"
    --num_workers "$num_workers"
    --result_dir "$result_dir"
    --test_all_meta_path "$json_name"
    --diff_lvl "$diff_lvl"
    --max_tokens "$max_tokens"
    --temperature "$temperature"
    --max_steps "$max_steps"
    --coord "$coord"
    --api_backend "$api_backend"
    --thinking_budget "$thinking_budget"
    --grounding_max_k "$grounding_max_k"
)

if [ -n "$video_json" ]; then
    run_args+=(--video_json "$video_json")
fi
if [ "$enable_planning" = true ]; then
    run_args+=(--enable_planning)
fi
if [ "$enable_grounding" = true ]; then
    run_args+=(--enable_grounding)
fi
if [ "$enable_thinking" = true ]; then
    run_args+=(--enable_thinking)
fi

"${run_args[@]}"
''',
        "start_client run command",
    )

    path.write_text(text, encoding="utf-8")
    print(f"patched file: {path}")


def install(args: argparse.Namespace) -> None:
    waa_client = Path(args.waa_client_dir).resolve()
    if not (waa_client / "run.py").exists():
        raise FileNotFoundError(f"{waa_client} does not look like a WAA client directory")

    source_client = PACKAGE_DIR / "client"
    copy_file(source_client / "video_knowledge.py", waa_client / "video_knowledge.py", args.overwrite)
    copy_tree(
        source_client / "mm_agents" / "qwen3vl_vrag",
        waa_client / "mm_agents" / "qwen3vl_vrag",
        args.overwrite,
    )
    copy_tree(
        source_client / "mm_agents" / "agent_s3",
        waa_client / "mm_agents" / "agent_s3",
        args.overwrite,
    )

    if args.install_agent_s3_backend:
        source_backend = ROOT / "osworld" / "new_gui_agents_with_video"
        copy_tree(source_backend, waa_client / "new_gui_agents_with_video", args.overwrite)

    if args.patch_entrypoints:
        patch_run_py(waa_client / "run.py")
        start_client = waa_client.parent / "start_client.sh"
        if start_client.exists():
            patch_start_client(start_client)
        else:
            print(f"start_client.sh not found, skipped: {start_client}")
    else:
        print()
        print("Next step: re-run with --patch-entrypoints or manually pass GUIDE flags to run.py.")


def config() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--waa-client-dir",
        required=True,
        help="Path to WindowsAgentArena/src/win-arena-container/client.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing adapter directories/files in the WAA client.",
    )
    parser.add_argument(
        "--install-agent-s3-backend",
        action="store_true",
        help="Also copy GUIDE's AgentS3 backend package needed by the AgentS3 adapter.",
    )
    parser.add_argument(
        "--patch-entrypoints",
        action="store_true",
        help="Patch WAA run.py and start_client.sh to register GUIDE agents and CLI flags.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    install(config())
