from __future__ import annotations
import argparse
import datetime
import json
import logging
import os
import sys
import re
import signal
import time
from typing import List
from multiprocessing import Process, Manager
from multiprocessing import current_process
import lib_run_single
from mm_agents.seed_agent import SeedAgent

from desktop_env.desktop_env import DesktopEnv

from dotenv import load_dotenv
from video_self_convert import get_converted_results
load_dotenv()

os.environ["OSWORLD_TOKEN"] = os.getenv("OSWORLD_TOKEN", "YOUR_TOKEN")
os.environ["OSWORLD_BASE_URL"] = 'http://YOUR_SERVER_IP:YOUR_PORT'
# -------------------------------------------------------------------
# Seed1.5-VL (Doubao) 服务配置
# DOUBAO_API_KEY 和 DOUBAO_API_URL 需在 .env 文件或环境变量中设置
# -------------------------------------------------------------------

# Global variables for signal handling
active_environments = []
processes = []
is_terminating = False

# Watchdog: max seconds a worker can be idle before being killed and restarted
TASK_TIMEOUT_SECONDS = 1800  # 30 minutes

# load the environment variables from .env file
if os.path.exists(".env"):
    from dotenv import load_dotenv
    load_dotenv()


def config() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run end-to-end evaluation on the benchmark (Seed1.5-VL / Doubao)"
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
    parser.add_argument("--sleep_after_execution", type=float, default=0.0)
    parser.add_argument("--max_steps", type=int, default=50)

    # agent config
    parser.add_argument("--max_trajectory_length", type=int, default=3)
    parser.add_argument("--history_n", type=int, default=5, help="Number of history images to keep")
    parser.add_argument(
        "--test_config_base_dir", type=str, default="evaluation_examples"
    )

    # lm config
    parser.add_argument("--model", type=str, default="doubao-seed-1-8-251228")
    parser.add_argument("--model_type", type=str, default="seed", help="Model type identifier")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--max_tokens", type=int, default=20480)
    parser.add_argument("--stop_token", type=str, default=None)
    parser.add_argument(
        "--use_thinking",
        action="store_true",
        default=True,
        help="Enable thinking mode for the model",
    )
    parser.add_argument(
        "--resize_image",
        action="store_true",
        default=False,
        help="Resize input images before sending to model",
    )
    parser.add_argument("--resized_image_width", type=int, default=1920)
    parser.add_argument("--resized_image_height", type=int, default=1080)

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
    parser.add_argument(
        "--annotation_json", type=str, default=None,
        help="Path to annotation JSON file for video knowledge (overrides default converted JSON)"
    )

    # example config
    parser.add_argument("--domain", type=str, default="all")
    parser.add_argument(
        "--test_all_meta_path", type=str, default="evaluation_examples/test_nogdrive.json"
    )

    # logging related
    parser.add_argument("--result_dir", type=str, default="./results")
    parser.add_argument(
        "--run_name", type=str, default=None,
        help="Descriptive name used for result directory (defaults to --model if not set)"
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

    # provider config
    parser.add_argument(
        "--region", type=str, default="us-east-1", help="AWS region for the VM"
    )
    parser.add_argument(
        "--screen_width", type=int, default=1920, help="Screen width"
    )
    parser.add_argument(
        "--screen_height", type=int, default=1080, help="Screen height"
    )
    args = parser.parse_args()
    return args


args = config()  # Get command line arguments first

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
    active_environments = []
    env = None
    proc_name = current_process().name

    def update_heartbeat():
        if heartbeats is not None:
            heartbeats[proc_name] = time.time()

    try:
        env = DesktopEnv(
            action_space="pyautogui",
            provider_name="docker_server",
            os_type='Ubuntu',
        )
        active_environments.append(env)
        agent = SeedAgent(
            model=args.model,
            model_type=args.model_type,
            max_tokens=args.max_tokens,
            top_p=args.top_p,
            temperature=args.temperature,
            max_trajectory_length=args.max_trajectory_length,
            history_n=args.history_n,
            max_steps=args.max_steps,
            use_thinking=args.use_thinking,
            resize_image=args.resize_image,
            resized_image_width=args.resized_image_width,
            resized_image_height=args.resized_image_height,
        )
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

                logger.info(f"[{current_process().name}][Domain]: {domain}")
                logger.info(f"[{current_process().name}][Example ID]: {example_id}")

                # Get web from related_apps
                related_apps = example['related_apps']
                web = ''
                for app in related_apps:
                    if app != related_apps[-1]:
                        web += app + ' + '
                    else:
                        web += app
                instruction = example["instruction"]

                logger.info(f"[{current_process().name}][Instruction]: {instruction}")

                # Generate video planning and grounding
                try:
                    kwargs = {}
                    if args.annotation_json:
                        kwargs['json_file_path'] = args.annotation_json
                    result = get_converted_results(web, instruction, **kwargs)
                    if result is None:
                        logger.warning(f"[{current_process().name}] No converted results found for web='{web}', instruction='{instruction}'. Using empty planning.")
                        video_planning = ""
                        video_grounding = ""
                    else:
                        video_planning, video_grounding = result
                    logger.info(f"[{current_process().name}][Video Planning]: {video_planning}")
                    logger.info(f"[{current_process().name}][Video Grounding]: {video_grounding}")
                except Exception as e:
                    logger.error(f"[{current_process().name}] Error in video conversion for {web} with query. Error: {e}")
                    video_planning = ""
                    video_grounding = ""

                # Update agent with video knowledge based on flags
                if not args.enable_planning:
                    video_planning = None
                if not args.enable_grounding:
                    video_grounding = None
                elif args.grounding_max_k > 0 and video_grounding:
                    video_grounding = truncate_grounding(video_grounding, args.grounding_max_k)
                agent.video_planning = video_planning
                agent.video_grounding = video_grounding

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
        logger.error(f"Process-level error in {current_process().name}: {e}")
        import traceback
        logger.error(traceback.format_exc())
    finally:
        logger.info(f"{current_process().name} cleaning up environment...")
        try:
            if env:
                env.close()
                logger.info(f"{current_process().name} environment closed successfully")
        except Exception as e:
            logger.error(f"{current_process().name} error during environment cleanup: {e}")


def signal_handler(signum, _):
    global is_terminating, active_environments, processes
    if is_terminating:
        return
    is_terminating = True
    logger.info(f"Received signal {signum}. Gracefully shutting down...")
    for env in active_environments:
        try:
            logger.info(f"Closing environment...")
            env.close()
            logger.info(f"Environment closed successfully")
        except Exception as e:
            logger.error(f"Error closing environment: {e}")
    for p in processes:
        if p.is_alive():
            try:
                logger.info(f"Sending termination signal to process {p.name}...")
                p.terminate()
            except Exception as e:
                logger.error(f"Error sending termination signal to process: {e}")
    time.sleep(1)
    for p in processes:
        if p.is_alive():
            try:
                logger.info(f"Forcefully terminating process {p.name}...")
                import signal as sig
                os.kill(p.pid, sig.SIGKILL)
            except Exception as e:
                logger.error(f"Error forcefully terminating process: {e}")
    logger.info("Shutdown complete. Exiting.")
    sys.exit(0)


def test(args: argparse.Namespace, test_all_meta: dict) -> None:
    global processes
    logger.info("Args: %s", args)
    all_tasks = distribute_tasks(test_all_meta)
    logger.info(f"Total tasks: {len(all_tasks)}")
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
                        # Watchdog: check if process is hung
                        last_hb = heartbeats.get(p.name)
                        if last_hb is not None:
                            idle_secs = now - last_hb
                            if idle_secs > TASK_TIMEOUT_SECONDS:
                                logger.warning(
                                    f"Watchdog: Process {p.name} (PID {p.pid}) has been idle for "
                                    f"{idle_secs:.0f}s (>{TASK_TIMEOUT_SECONDS}s), killing and restarting..."
                                )
                                try:
                                    p.terminate()
                                    p.join(timeout=5)
                                    if p.is_alive():
                                        os.kill(p.pid, signal.SIGKILL)
                                        logger.warning(f"Watchdog: Force-killed process {p.name} (PID {p.pid})")
                                except Exception as kill_e:
                                    logger.error(f"Watchdog: Error killing process {p.name}: {kill_e}")
                                # Remove stale heartbeat
                                heartbeats.pop(p.name, None)
                                # Start replacement
                                new_p = Process(
                                    target=run_env_tasks,
                                    args=(task_queue, args, shared_scores, heartbeats),
                                    name=f"EnvProcess-Watchdog-{idx+1}"
                                )
                                new_p.daemon = True
                                new_p.start()
                                processes[idx] = new_p
                                logger.info(f"Watchdog: Restarted as {new_p.name} with PID {new_p.pid}")
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
                        logger.info(f"Terminating process {p.name} due to error...")
                        p.terminate()
                    except Exception as term_e:
                        logger.error(f"Error terminating process {p.name}: {term_e}")
            raise
        scores = list(shared_scores)
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
                            value_str = open(
                                os.path.join(example_path, "result.txt"), "r"
                            ).read()
                            all_result.append(float(value_str))
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
        args = config()
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
                    logger.info("Closing environment in final cleanup...")
                    env.close()
                    logger.info("Environment closed successfully in final cleanup")
                except Exception as e:
                    logger.error(f"Error during final environment cleanup: {e}")
        for p in processes:
            if p is not None and p.is_alive():
                try:
                    logger.info(f"Terminating process {p.name}...")
                    p.terminate()
                except Exception as e:
                    logger.error(f"Error terminating process: {e}")
        time.sleep(1)
        for p in processes:
            if p is not None and p.is_alive():
                try:
                    logger.info(f"Force killing process {p.name}...")
                    os.kill(p.pid, signal.SIGKILL)
                    logger.info(f"Process {p.name} force killed")
                except Exception as e:
                    logger.error(f"Error force killing process: {e}")
