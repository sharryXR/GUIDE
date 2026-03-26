"""Script to run end-to-end evaluation on the benchmark with AgentS3.
Utils and basic architecture credit to https://github.com/web-arena-x/webarena/blob/main/run.py.
"""

from __future__ import annotations
import argparse
import datetime
import json
import logging
import os
import re
import signal
import sys
import time
import threading
from typing import List
from multiprocessing import Process, Manager
from multiprocessing import current_process
import requests

import lib_run_single_s3_with_video as lib_run_single
from desktop_env.desktop_env import DesktopEnv
from new_gui_agents_with_video.s3.agents.agent_s import AgentS3
from new_gui_agents_with_video.s3.agents.grounding import OSWorldACI

from dotenv import load_dotenv
from video_self_convert import get_converted_results
load_dotenv()

os.environ["OSWORLD_TOKEN"] = os.getenv("OSWORLD_TOKEN", "YOUR_TOKEN")
os.environ["OSWORLD_BASE_URL"] = 'http://YOUR_SERVER_IP:YOUR_PORT'

# Global variables for signal handling
active_environments = []
processes = []
is_terminating = False

# Watchdog: max seconds a worker can be idle before being killed and restarted
TASK_TIMEOUT_SECONDS = 1800  # 30 minutes

# Budget monitor: check interval in seconds
BUDGET_CHECK_INTERVAL = 120  # every 2 minutes


def _check_gpt_alive(api_key: str) -> bool:
    """Return True if GPT key responds normally; False only on quota/auth/credit errors."""
    base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/")
    try:
        resp = requests.post(
            f"{base_url.rstrip('/')}/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={"model": "gpt-5.2-2025-12-11", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 1},
            timeout=20,
        )
        if resp.status_code == 200:
            return True
        err = resp.text.lower()
        # Only treat as dead on clear budget-exhaustion signals, NOT transient rate-limit errors
        if any(kw in err for kw in ("quota exceeded", "insufficient funds", "no credit", "budget exceeded", "has been exceeded", "budget_exceeded", "billing")):
            return False
        return True  # rate-limit (429), 5xx, or other transient errors — don't kill
    except Exception:
        return True  # network/timeout errors — don't kill


def _check_doubao_alive(api_key: str, api_url: str) -> bool:
    """Return False if doubao key is exhausted (quota error), True otherwise."""
    try:
        from openai import OpenAI
        OpenAI(api_key=api_key, base_url=api_url).chat.completions.create(
            model="doubao-seed-1-8-251228",
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=1,
        )
        return True
    except Exception as e:
        err = str(e).lower()
        if any(kw in err for kw in ("quota", "insufficient", "balance", "credit", "limit")):
            return False
        return True  # other errors (network etc.) — don't kill the experiment


def budget_monitor(stop_event: threading.Event) -> None:
    """Background thread: stop only if ALL GPT keys dead OR seed key dead."""
    gpt_keys = {
        "OPENAI_API_KEY":   os.getenv("OPENAI_API_KEY", ""),
        "OPENAI_API_KEY_2": os.getenv("OPENAI_API_KEY_2", ""),
    }
    doubao_key = os.getenv("DOUBAO_API_KEY", "")
    doubao_url = os.getenv("DOUBAO_API_URL", "https://ark.cn-beijing.volces.com/api/v3")

    while not stop_event.is_set():
        stop_event.wait(BUDGET_CHECK_INTERVAL)
        if stop_event.is_set():
            break

        # Check GPT keys — at least one must be alive to continue
        gpt_alive_any = False
        active_gpt_keys = [(name, key) for name, key in gpt_keys.items() if key]
        for key_name, api_key in active_gpt_keys:
            alive = _check_gpt_alive(api_key)
            logger.info(f"[BudgetMonitor] {key_name}: {'✅ 可用' if alive else '❌ 不可用'}")
            if alive:
                gpt_alive_any = True

        if active_gpt_keys and not gpt_alive_any:
            logger.error(
                "[BudgetMonitor] 所有 GPT key 均无法响应，"
                "终止全部实验以避免浪费 Seed 资源。"
            )
            signal_handler(signal.SIGTERM, None)
            return

        # Check Doubao/seed key
        if doubao_key:
            alive = _check_doubao_alive(doubao_key, doubao_url)
            logger.info(f"[BudgetMonitor] DOUBAO_API_KEY: {'✅ 可用' if alive else '❌ 不可用'}")
            if not alive:
                logger.error(
                    "[BudgetMonitor] DOUBAO_API_KEY 余额耗尽，"
                    "终止全部实验以避免浪费 GPT 资源。"
                )
                signal_handler(signal.SIGTERM, None)
                return


def config() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run end-to-end evaluation on the benchmark (AgentS3)"
    )

    # environment config
    parser.add_argument("--path_to_vm", type=str, default=None)
    parser.add_argument(
        "--headless", action="store_true", help="Run in headless machine"
    )
    parser.add_argument(
        "--action_space", type=str, default="pyautogui", help="Action type"
    )
    parser.add_argument(
        "--observation_type",
        choices=["screenshot", "a11y_tree", "screenshot_a11y_tree", "som"],
        default="screenshot",
        help="Observation type",
    )
    parser.add_argument("--screen_width", type=int, default=1920)
    parser.add_argument("--screen_height", type=int, default=1080)
    parser.add_argument("--sleep_after_execution", type=float, default=3.0)
    parser.add_argument("--max_steps", type=int, default=30)

    # agent config
    parser.add_argument("--max_trajectory_length", type=int, default=3)
    parser.add_argument(
        "--test_config_base_dir", type=str, default="evaluation_examples"
    )

    # lm config
    parser.add_argument("--model", type=str, default="agentS3_withvideo")
    parser.add_argument("--temperature", type=float, default=1.0)

    # video knowledge config
    parser.add_argument(
        "--enable_planning", action="store_true", default=False,
        help="Enable video planning knowledge"
    )
    parser.add_argument(
        "--enable_grounding", action="store_true", default=False,
        help="Enable video grounding knowledge"
    )
    parser.add_argument(
        "--grounding_max_k", type=int, default=0,
        help="Max number of grounding elements per demo section (0=no limit)"
    )

    # example config
    parser.add_argument("--domain", type=str, default="all")
    parser.add_argument(
        "--test_all_meta_path", type=str, default="evaluation_examples/test_nogdrive.json"
    )

    # logging / runner config
    parser.add_argument("--result_dir", type=str, default="./results")
    parser.add_argument(
        "--run_name", type=str, default=None,
        help="Descriptive name for result directory (defaults to --model if not set)"
    )
    parser.add_argument(
        "--num_envs", type=int, default=1, help="Number of environments to run in parallel"
    )
    parser.add_argument(
        "--log_level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Set the logging level",
    )
    args = parser.parse_args()
    return args


args = config()  # parse args first so log_level is available for logger setup

logger = logging.getLogger()
log_level = getattr(logging, args.log_level.upper())
logger.setLevel(log_level)

datetime_str: str = datetime.datetime.now().strftime("%Y%m%d@%H%M%S")

file_handler = logging.FileHandler(
    os.path.join("logs", "normal-{:}.log".format(datetime_str)), encoding="utf-8"
)
debug_handler = logging.FileHandler(
    os.path.join("logs", "debug-{:}.log".format(datetime_str)), encoding="utf-8"
)
stdout_handler = logging.StreamHandler(sys.stdout)

file_handler.setLevel(logging.INFO)
debug_handler.setLevel(logging.DEBUG)
stdout_handler.setLevel(log_level)

formatter = logging.Formatter(
    fmt=(
        "\x1b[1;33m[%(asctime)s \x1b[31m%(levelname)s "
        "\x1b[32m%(module)s/%(lineno)d-%(processName)s\x1b[1;33m] "
        "\x1b[0m%(message)s"
    )
)
file_handler.setFormatter(formatter)
debug_handler.setFormatter(formatter)
stdout_handler.setFormatter(formatter)

stdout_handler.addFilter(logging.Filter("desktopenv"))

logger.addHandler(file_handler)
logger.addHandler(debug_handler)
logger.addHandler(stdout_handler)

logger = logging.getLogger("desktopenv.experiment")


def truncate_grounding(grounding: str, max_k: int) -> str:
    """Truncate each demo section in grounding to keep only the first max_k numbered elements."""
    if not grounding or max_k <= 0:
        return grounding
    parts = re.split(r'(The grounding trajectory of Demo \d+:[^\n]*\n)', grounding)
    result = []
    for part in parts:
        if re.match(r'The grounding trajectory of Demo \d+:', part):
            result.append(part)
        else:
            pattern = re.compile(r'^\d+\.\s', re.MULTILINE)
            matches = list(pattern.finditer(part))
            if len(matches) > max_k:
                truncated = part[:matches[max_k].start()].rstrip() + '\n'
                result.append(truncated)
            else:
                result.append(part)
    return ''.join(result).strip()


def distribute_tasks(test_all_meta: dict) -> List[tuple]:
    all_tasks = []
    for domain, examples in test_all_meta.items():
        for example_id in examples:
            all_tasks.append((domain, example_id))
    return all_tasks


def run_env_tasks(task_queue, args: argparse.Namespace, shared_scores: list, heartbeats: dict = None):
    env = None
    proc_name = current_process().name

    # Engine params for AgentS3 (use KEY_2 as KEY_1 is exhausted)
    engine_params = {
        "engine_type": "openai",
        "model": "gpt-5.2-2025-12-11",
        "api_key": os.getenv("OPENAI_API_KEY_2"),
        "base_url": os.getenv("OPENAI_BASE_URL"),
        "temperature": args.temperature,
        "timeout": 120,
    }
    grounding_engine_params = {
        "engine_type": "openai",
        "model": "doubao-seed-1-8-251228",
        "api_key": os.getenv("DOUBAO_API_KEY"),
        "base_url": os.getenv("DOUBAO_API_URL"),
        "grounding_width": args.screen_width,
        "grounding_height": args.screen_height,
    }
    # Code agent uses seed model for code generation (no grounding-specific keys needed)
    code_agent_engine_params = {
        "engine_type": "openai",
        "model": "doubao-seed-1-8-251228",
        "api_key": os.getenv("DOUBAO_API_KEY"),
        "base_url": os.getenv("DOUBAO_API_URL"),
        "timeout": 120,
    }

    def update_heartbeat():
        if heartbeats is not None:
            heartbeats[proc_name] = time.time()
            # Track current emulator_id so watchdog can release it on crash
            if env is not None and hasattr(env, 'provider') and hasattr(env.provider, 'emulator_id') and env.provider.emulator_id:
                heartbeats[proc_name + "_emu"] = env.provider.emulator_id

    try:
        env = DesktopEnv(
            action_space="pyautogui",
            provider_name="docker_server",
            os_type='Ubuntu',
        )
        grounding_agent = OSWorldACI(
            env=env,
            platform="linux",
            engine_params_for_generation=engine_params,
            engine_params_for_grounding=grounding_engine_params,
            code_agent_engine_params=code_agent_engine_params,
            width=args.screen_width,
            height=args.screen_height,
        )
        agent = AgentS3(
            engine_params,
            grounding_agent,
            platform="linux",
            max_trajectory_length=args.max_trajectory_length,
        )
        # Patch env.reset so that each task's new emulator_id is tracked immediately
        _original_env_reset = env.reset
        def _patched_env_reset(*args, **kwargs):
            result = _original_env_reset(*args, **kwargs)
            update_heartbeat()  # capture new emulator_id after reset
            return result
        env.reset = _patched_env_reset

        logger.info(f"Process {proc_name} started.")
        update_heartbeat()

        while True:
            try:
                item = task_queue.get(timeout=5)
            except Exception:
                break
            domain, example_id = item
            update_heartbeat()
            try:
                config_file = os.path.join(
                    args.test_config_base_dir, f"examples/{domain}/{example_id}.json"
                )
                with open(config_file, "r", encoding="utf-8") as f:
                    example = json.load(f)

                logger.info(f"[{proc_name}][Domain]: {domain}")
                logger.info(f"[{proc_name}][Example ID]: {example_id}")

                related_apps = example['related_apps']
                web = ' + '.join(related_apps)
                instruction = example["instruction"]
                logger.info(f"[{proc_name}][Instruction]: {instruction}")

                # Load pre-converted planning/grounding annotations
                try:
                    result = get_converted_results(web, instruction)
                    if result is None:
                        logger.warning(f"[{proc_name}] No converted results for web='{web}'. Using empty planning.")
                        video_planning = ""
                        video_grounding = ""
                    else:
                        video_planning, video_grounding = result
                    logger.info(f"[{proc_name}][Video Planning]: {video_planning}")
                    logger.info(f"[{proc_name}][Video Grounding]: {video_grounding}")
                except Exception as e:
                    logger.error(f"[{proc_name}] Error fetching converted results for {web}. Error: {e}")
                    video_planning = ""
                    video_grounding = ""

                # Apply enable_planning / enable_grounding flags
                if not args.enable_planning:
                    video_planning = None
                if not args.enable_grounding:
                    video_grounding = None
                elif args.grounding_max_k > 0 and video_grounding:
                    video_grounding = truncate_grounding(video_grounding, args.grounding_max_k)

                example_result_dir = os.path.join(
                    args.result_dir,
                    args.action_space,
                    args.observation_type,
                    args.run_name or args.model,
                    domain,
                    example_id,
                )
                os.makedirs(example_result_dir, exist_ok=True)

                try:
                    lib_run_single.run_single_example(
                        agent,
                        env,
                        example,
                        args.max_steps,
                        instruction,
                        args,
                        example_result_dir,
                        shared_scores,
                        video_planning=video_planning,
                        video_grounding=video_grounding,
                    )
                    update_heartbeat()
                except Exception as e:
                    import traceback
                    logger.error(f"Exception in {proc_name} {domain}/{example_id}: {e}")
                    logger.error(traceback.format_exc())
                    try:
                        if hasattr(env, "controller") and env.controller is not None:
                            env.controller.end_recording(
                                os.path.join(example_result_dir, "recording.mp4")
                            )
                    except Exception as rec_e:
                        logger.error(f"Failed to end recording: {rec_e}")
                    with open(os.path.join(example_result_dir, "traj.jsonl"), "a") as f:
                        f.write(json.dumps({"Error": f"{domain}/{example_id} - {e}"}))
                        f.write("\n")
                    update_heartbeat()
            except Exception as e:
                logger.error(f"Task-level error in {proc_name}: {e}")
                import traceback
                logger.error(traceback.format_exc())
    except Exception as e:
        logger.error(f"Process-level error in {proc_name}: {e}")
        import traceback
        logger.error(traceback.format_exc())
    finally:
        logger.info(f"{proc_name} cleaning up environment...")
        try:
            if env:
                env.close()
                logger.info(f"{proc_name} environment closed successfully")
        except Exception as e:
            logger.error(f"{proc_name} error during environment cleanup: {e}")
        # Clear emulator tracking so watchdog won't double-release
        if heartbeats is not None:
            heartbeats.pop(proc_name + "_emu", None)


def signal_handler(signum, _):
    global is_terminating, active_environments, processes
    if is_terminating:
        return
    is_terminating = True
    logger.info(f"Received signal {signum}. Gracefully shutting down...")
    for env in active_environments:
        try:
            env.close()
        except Exception as e:
            logger.error(f"Error closing environment: {e}")
    for p in processes:
        if p.is_alive():
            try:
                p.terminate()
            except Exception as e:
                logger.error(f"Error terminating process {p.name}: {e}")
    time.sleep(1)
    for p in processes:
        if p.is_alive():
            try:
                os.kill(p.pid, signal.SIGKILL)
            except Exception as e:
                logger.error(f"Error force-killing process {p.name}: {e}")
    logger.info("Shutdown complete. Exiting.")
    sys.exit(0)


def _release_emulator(proc_name: str, heartbeats: dict) -> None:
    """Release the docker container owned by a dead process to free up quota."""
    emu_id = heartbeats.pop(proc_name + "_emu", None)
    if not emu_id:
        return
    try:
        base_url = os.environ.get("OSWORLD_BASE_URL", "http://YOUR_SERVER_IP:YOUR_PORT")
        token = os.environ.get("OSWORLD_TOKEN", "")
        resp = requests.delete(
            f"{base_url}/emulators/{emu_id}",
            headers={"Authorization": f"Bearer {token}"},
            timeout=10,
        )
        logger.info(f"Released orphaned emulator {emu_id} for {proc_name}: HTTP {resp.status_code}")
    except Exception as e:
        logger.warning(f"Failed to release emulator {emu_id} for {proc_name}: {e}")


def test(args: argparse.Namespace, test_all_meta: dict) -> None:
    global processes
    logger.info("Args: %s", args)
    all_tasks = distribute_tasks(test_all_meta)
    logger.info(f"Total tasks: {len(all_tasks)}")

    # Start budget monitor thread
    _budget_stop = threading.Event()
    _budget_thread = threading.Thread(
        target=budget_monitor, args=(_budget_stop,), daemon=True, name="BudgetMonitor"
    )
    _budget_thread.start()
    logger.info("BudgetMonitor started (check interval: %ds)", BUDGET_CHECK_INTERVAL)

    with Manager() as manager:
        shared_scores = manager.list()
        task_queue = manager.Queue()
        heartbeats = manager.dict()
        for item in all_tasks:
            task_queue.put(item)

        num_envs = args.num_envs
        processes = []
        for i in range(num_envs):
            p = Process(
                target=run_env_tasks,
                args=(task_queue, args, shared_scores, heartbeats),
                name=f"EnvProcess-{i+1}"
            )
            p.daemon = True
            p.start()
            processes.append(p)
            logger.info(f"Started process {p.name} with PID {p.pid}")

        try:
            while True:
                alive_count = 0
                now = time.time()
                for idx, p in enumerate(processes):
                    if not p.is_alive():
                        logger.warning(f"Process {p.name} died, restarting...")
                        _release_emulator(p.name, heartbeats)
                        new_p = Process(
                            target=run_env_tasks,
                            args=(task_queue, args, shared_scores, heartbeats),
                            name=f"EnvProcess-Restart-{idx+1}"
                        )
                        new_p.daemon = True
                        new_p.start()
                        processes[idx] = new_p
                        logger.info(f"Restarted process {new_p.name} with PID {new_p.pid}")
                    else:
                        alive_count += 1
                        # Watchdog: kill and restart hung processes
                        last_hb = heartbeats.get(p.name)
                        if last_hb is not None:
                            idle_secs = now - last_hb
                            if idle_secs > TASK_TIMEOUT_SECONDS:
                                logger.warning(
                                    f"Watchdog: Process {p.name} (PID {p.pid}) idle for "
                                    f"{idle_secs:.0f}s (>{TASK_TIMEOUT_SECONDS}s), killing..."
                                )
                                try:
                                    p.terminate()
                                    p.join(timeout=5)
                                    if p.is_alive():
                                        os.kill(p.pid, signal.SIGKILL)
                                except Exception as kill_e:
                                    logger.error(f"Watchdog: Error killing {p.name}: {kill_e}")
                                heartbeats.pop(p.name, None)
                                _release_emulator(p.name, heartbeats)
                                new_p = Process(
                                    target=run_env_tasks,
                                    args=(task_queue, args, shared_scores, heartbeats),
                                    name=f"EnvProcess-Watchdog-{idx+1}"
                                )
                                new_p.daemon = True
                                new_p.start()
                                processes[idx] = new_p
                                logger.info(f"Watchdog: Restarted as {new_p.name} PID {new_p.pid}")
                if task_queue.empty():
                    logger.info("All tasks finished.")
                    break
                if alive_count == 0:
                    logger.error("All processes died, exiting.")
                    break
                time.sleep(5)

            for p in processes:
                p.join()
        except KeyboardInterrupt:
            logger.info("Main process received KeyboardInterrupt. Initiating graceful shutdown...")
            raise
        except Exception as e:
            logger.error(f"Unexpected error while waiting for processes: {e}", exc_info=True)
            for p in processes:
                if p.is_alive():
                    try:
                        p.terminate()
                    except Exception as term_e:
                        logger.error(f"Error terminating process {p.name}: {term_e}")
            raise

        scores = list(shared_scores)

    _budget_stop.set()
    logger.info(f"Average score: {sum(scores) / len(scores) if scores else 0}")


def get_unfinished(
    action_space, use_model, observation_type, result_dir, total_file_json
):
    target_dir = os.path.join(result_dir, action_space, observation_type, use_model)

    if not os.path.exists(target_dir):
        return total_file_json

    finished = {}
    for domain in os.listdir(target_dir):
        finished[domain] = []
        domain_path = os.path.join(target_dir, domain)
        if os.path.isdir(domain_path):
            for example_id in os.listdir(domain_path):
                if example_id == "onboard":
                    continue
                example_path = os.path.join(domain_path, example_id)
                if os.path.isdir(example_path):
                    if "result.txt" not in os.listdir(example_path):
                        for file in os.listdir(example_path):
                            os.remove(os.path.join(example_path, file))
                    else:
                        finished[domain].append(example_id)

    if not finished:
        return total_file_json

    for domain, examples in finished.items():
        if domain in total_file_json:
            total_file_json[domain] = [
                x for x in total_file_json[domain] if x not in examples
            ]

    return total_file_json


def get_result(action_space, use_model, observation_type, result_dir, total_file_json):
    target_dir = os.path.join(result_dir, action_space, observation_type, use_model)
    if not os.path.exists(target_dir):
        print("New experiment, no result yet.")
        return None

    all_result = []

    for domain in os.listdir(target_dir):
        domain_path = os.path.join(target_dir, domain)
        if os.path.isdir(domain_path):
            for example_id in os.listdir(domain_path):
                example_path = os.path.join(domain_path, example_id)
                if os.path.isdir(example_path):
                    if "result.txt" in os.listdir(example_path):
                        try:
                            all_result.append(
                                float(
                                    open(
                                        os.path.join(example_path, "result.txt"), "r"
                                    ).read()
                                )
                            )
                        except Exception:
                            all_result.append(0.0)

    if not all_result:
        print("New experiment, no result yet.")
        return None
    else:
        print("Current Success Rate:", sum(all_result) / len(all_result) * 100, "%")
        return all_result


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    try:
        run_name = args.run_name or args.model
        path_to_args = os.path.join(
            args.result_dir,
            args.action_space,
            args.observation_type,
            run_name,
            "args.json",
        )
        os.makedirs(os.path.dirname(path_to_args), exist_ok=True)
        with open(path_to_args, "w", encoding="utf-8") as f:
            json.dump(vars(args), f, indent=4)

        with open(args.test_all_meta_path, "r", encoding="utf-8") as f:
            test_all_meta = json.load(f)

        if args.domain != "all":
            test_all_meta = {args.domain: test_all_meta[args.domain]}

        test_file_list = get_unfinished(
            args.action_space,
            run_name,
            args.observation_type,
            args.result_dir,
            test_all_meta,
        )
        left_info = ""
        for domain in test_file_list:
            left_info += f"{domain}: {len(test_file_list[domain])}\n"
        logger.info(f"Left tasks:\n{left_info}")

        get_result(
            args.action_space,
            run_name,
            args.observation_type,
            args.result_dir,
            test_all_meta,
        )
        test(args, test_file_list)
    except KeyboardInterrupt:
        logger.info("Main process received KeyboardInterrupt.")
    except Exception as e:
        logger.error(f"Unexpected error in main process: {e}", exc_info=True)
        signal_handler(signal.SIGTERM, None)
    finally:
        logger.info("Main process final cleanup...")
        for env in active_environments:
            if env is not None:
                try:
                    env.close()
                except Exception as e:
                    logger.error(f"Error during final environment cleanup: {e}")
        for p in processes:
            if p is not None and p.is_alive():
                try:
                    p.terminate()
                except Exception as e:
                    logger.error(f"Error terminating process: {e}")
        time.sleep(1)
        for p in processes:
            if p is not None and p.is_alive():
                try:
                    os.kill(p.pid, signal.SIGKILL)
                except Exception as e:
                    logger.error(f"Error force killing process: {e}")
