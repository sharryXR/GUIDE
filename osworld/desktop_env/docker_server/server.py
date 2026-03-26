from flask import Flask, request, jsonify, send_file, abort, render_template
import uuid
import logging
from desktop_env.providers.docker.provider import DockerProvider
from dataclasses import dataclass, field
from typing import Dict, Optional
import time
import os
import threading
import psutil
import docker
from collections import deque
import json
from datetime import datetime

from desktop_env.utils import load_config

logger = logging.getLogger("desktopenv.docker_server")
logger.setLevel(logging.INFO)

# Load config with safe defaults
_cfg = None
try:
    _cfg = load_config()
except Exception as e:
    logger.warning(f"Failed to load config.yaml, using defaults. Error: {e}")

def _cfg_get(path: str, default=None):
    try:
        cur = _cfg
        for part in path.split("."):
            if cur is None:
                return default
            cur = cur.get(part) if hasattr(cur, "get") else getattr(cur, part)
        return cur if cur is not None else default
    except Exception:
        return default

# Token/auth configuration
TOKEN_LIMITS = dict(_cfg_get("tokens", {}) or {})
REQUIRE_TOKEN = bool(_cfg_get("auth.require_token", False))
HEADER_NAME = _cfg_get("auth.header_name", "Authorization")
BEARER_PREFIX = _cfg_get("auth.bearer_prefix", "Bearer ")

WAIT_TIME = 3
RETRY_INTERVAL = 1
LOCK_TIMEOUT = 10
# VM path from config (with default for fallback)
DEFAULT_VM_PATH = "/home/shichenrui/TongGUI/ubuntu_env/desktop_env/Ubuntu.qcow2"
PATH_TO_VM = _cfg_get("remote_docker_server.path_to_vm", DEFAULT_VM_PATH)

# Base directory for resolving script paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# Flask app (templates under ./templates next to this file)
app = Flask(__name__)

# Request logging setup
LOG_PATH = os.path.join("logs", "requests.jsonl")
REQUEST_LOG_MAX = 1000
request_logs = deque(maxlen=REQUEST_LOG_MAX)

def _append_request_log(entry):
    request_logs.append(entry)
    try:
        os.makedirs("logs", exist_ok=True)
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as e:
        logger.warning(f"Failed to persist request log: {e}")

@app.before_request
def _capture_request():
    try:
        payload = request.get_json(silent=True)
    except Exception:
        payload = None
    entry = {
        "ts": datetime.utcnow().isoformat() + "Z",
        "method": request.method,
        "path": request.path,
        "token": _extract_token(request),
        "json": payload,
    }
    _append_request_log(entry)

@app.route("/request_logs", methods=["GET"])
def request_logs_api():
    try:
        limit = int(request.args.get("limit", 200))
    except Exception:
        limit = 200
    if limit < 0:
        limit = 0
    items = list(request_logs)[-limit:] if limit else list(request_logs)
    return jsonify(items)

# Visualization window route (inlined)
@app.route("/window", methods=["GET"])
def emulator_window():
    return render_template("window.html")

server_started_at = time.time()
lock = threading.Lock()

@dataclass
class Emulator:
    provider: DockerProvider
    emulator_id: str
    token: Optional[str] = None
    start_time: float = field(default_factory=time.time)
    
    def stop_emulator(self):
        # keep original path for compatibility
        self.provider.stop_emulator(path_to_vm=PATH_TO_VM)
    
    def duration(self) -> int:
        # convert to minutes
        return int((time.time() - self.start_time) / 60)

# emulator_id -> Emulator
emulators: Dict[str, Emulator] = dict()
# token usage/current state
token_usage: Dict[str, Dict] = dict()  # token -> {"current": int, "limit": int, "emulator_ids": set()}

def _sum_network_bytes(networks):
    rx = 0
    tx = 0
    if isinstance(networks, dict):
        for v in networks.values():
            try:
                rx += int(v.get("rx_bytes", 0) or 0)
                tx += int(v.get("tx_bytes", 0) or 0)
            except Exception:
                pass
    return rx, tx

def _sum_blkio_bytes(blkio):
    read = 0
    write = 0
    if isinstance(blkio, dict):
        entries = blkio.get("io_service_bytes_recursive") or []
        for e in entries:
            try:
                op = str(e.get("op", "")).lower()
                val = int(e.get("value", 0) or 0)
                if op == "read":
                    read += val
                elif op == "write":
                    write += val
            except Exception:
                pass
    return read, write

def _calc_cpu_percent_from_frames(prev, cur):
    try:
        prev_cpu_total = (prev.get("cpu_stats", {}).get("cpu_usage", {}).get("total_usage", 0) or 0)
        cur_cpu_total = (cur.get("cpu_stats", {}).get("cpu_usage", {}).get("total_usage", 0) or 0)
        prev_system = (prev.get("cpu_stats", {}).get("system_cpu_usage", 0) or 0)
        cur_system = (cur.get("cpu_stats", {}).get("system_cpu_usage", 0) or 0)
        cpu_delta = cur_cpu_total - prev_cpu_total
        system_delta = cur_system - prev_system
        ncpu = (cur.get("cpu_stats", {}).get("online_cpus")
                or len(cur.get("cpu_stats", {}).get("cpu_usage", {}).get("percpu_usage") or [])
                or 1)
        if system_delta > 0 and cpu_delta >= 0:
            return (cpu_delta / system_delta) * ncpu * 100.0
    except Exception:
        pass
    return None

def _calc_cpu_percent(container):
    # Prefer streaming two frames for accurate CPU percentage
    try:
        gen = container.stats(stream=True, decode=True)
        prev = next(gen)
        cur = next(gen)
        return _calc_cpu_percent_from_frames(prev, cur)
    except Exception:
        # Fallback to single-frame precpu calculation if available
        try:
            st = container.stats(stream=False, decode=True)
            cpu_stats = st.get("cpu_stats", {}) or {}
            precpu_stats = st.get("precpu_stats", {}) or {}
            if cpu_stats and precpu_stats:
                return _calc_cpu_percent_from_frames({"cpu_stats": precpu_stats}, {"cpu_stats": cpu_stats})
        except Exception:
            pass
    return None

def parse_container_stats(container):
    try:
        st = container.stats(stream=False, decode=True)
    except Exception as e:
        return {"error": str(e)}

    # Memory usage and limit
    mem_stats = st.get("memory_stats", {}) or {}
    mem_usage = int(mem_stats.get("usage", 0) or 0)
    try:
        cache = int((mem_stats.get("stats", {}) or {}).get("cache", 0) or 0)
        mem_usage = max(mem_usage - cache, 0)
    except Exception:
        pass
    mem_limit = int(mem_stats.get("limit", 0) or 0)
    mem_percent = (mem_usage / mem_limit * 100.0) if mem_limit else None

    # CPU percent
    cpu_percent = _calc_cpu_percent(container)

    # Network bytes
    rx_bytes, tx_bytes = _sum_network_bytes(st.get("networks", {}) or {})

    # Block IO bytes
    blk_read, blk_write = _sum_blkio_bytes(st.get("blkio_stats", {}) or {})

    return {
        "cpu_percent": cpu_percent,
        "memory_usage_bytes": mem_usage,
        "memory_limit_bytes": mem_limit,
        "memory_percent": mem_percent,
        "net_rx_bytes": rx_bytes,
        "net_tx_bytes": tx_bytes,
        "blk_read_bytes": blk_read,
        "blk_write_bytes": blk_write,
    }

def _extract_token(req: request) -> Optional[str]:
    # Priority 1: header
    auth_val = req.headers.get(HEADER_NAME)
    if auth_val and auth_val.startswith(BEARER_PREFIX):
        return auth_val[len(BEARER_PREFIX):].strip()
    # Priority 2: JSON body
    if req.is_json:
        body = req.get_json(silent=True) or {}
        if "token" in body:
            return str(body.get("token") or "").strip()
    # Priority 3: query string
    if "token" in req.args:
        return req.args.get("token", "").strip()
    return None

def _get_token_limit(token: str) -> Optional[int]:
    try:
        return int(TOKEN_LIMITS.get(token)) if token in TOKEN_LIMITS else None
    except Exception:
        return None

def _ensure_token_initialized(token: str) -> bool:
    if token not in TOKEN_LIMITS:
        return False
    lim = _get_token_limit(token)
    if token not in token_usage:
        token_usage[token] = {
            "current": 0,
            "limit": lim if lim is not None else 0,
            "emulator_ids": set(),
        }
    else:
        token_usage[token]["limit"] = lim if lim is not None else 0
    return True

@app.route("/ping", methods=["GET"])
def read_root():
    return {"message": "Hello, World!"}

@app.route("/start_emulator", methods=["GET", "POST"])
def start_emulator():
    print("start_emulator")
    token = _extract_token(request)
    if not token:
        return jsonify({"message": "Token required", "code": 401}), 401
    if token not in TOKEN_LIMITS:
        return jsonify({"message": f"Unknown or unauthorized token '{token}'", "code": 403}), 403

    # Atomic quota check and pre-allocation
    emulator_id = str(uuid.uuid4())
    with lock:
        ok = _ensure_token_initialized(token)
        if not ok:
            return jsonify({"message": f"Unknown or unauthorized token '{token}'", "code": 403}), 403
        current = token_usage[token]["current"]
        limit = token_usage[token]["limit"]
        if current >= limit:
            return jsonify({"message": f"Token quota exceeded for '{token}': {current}/{limit}", "code": 429}), 429
        
        # Immediately pre-allocate quota (atomic operation)
        token_usage[token]["current"] += 1
        token_usage[token]["emulator_ids"].add(emulator_id)
        logger.info(f"Pre-allocated quota for token '{token}': {current+1}/{limit}, emulator_id={emulator_id}")

    # Start emulator outside the lock
    provider = DockerProvider(region="")
    try:
        # Label container with token and emulator_id for observability
        labels = {"osworld.emulator_id": emulator_id}
        if token:
            labels["osworld.token"] = token
        try:
            provider.start_emulator(
                path_to_vm=PATH_TO_VM,
                headless=True,
                os_type="linux",
                labels=labels
            )
        except TypeError as te:
            # Backward compatibility: older DockerProvider.start_emulator may not accept 'labels'
            if "labels" in str(te):
                provider.start_emulator(
                    path_to_vm=PATH_TO_VM,
                    headless=True,
                    os_type="linux"
                )
            else:
                raise
        
        # Create emulator object
        emu = Emulator(
            provider=provider,
            emulator_id=emulator_id,
            token=token,
        )
        emulators[emulator_id] = emu

        logger.info(f"Successfully started emulator {emulator_id} for token '{token}'")
        return jsonify({
            "message": "Emulator started successfully",
            "code": 0,
            "data": {
                "emulator_id": emulator_id,
                "vnc_port": provider.vnc_port,
                "chromium_port": provider.chromium_port,
                "vlc_port": provider.vlc_port,
                "server_port": provider.server_port,
                "token": token
            }
        })
    except Exception as e:
        # Rollback quota on failure
        logger.error(f"Failed to start emulator {emulator_id}: {e}")
        with lock:
            token_usage[token]["current"] -= 1
            token_usage[token]["emulator_ids"].discard(emulator_id)
            logger.info(f"Rolled back quota for token '{token}': {token_usage[token]['current']}/{limit}")
        try:
            if provider and provider.container:
                provider.stop_emulator(path_to_vm=PATH_TO_VM)
        except Exception:
            pass
        return jsonify({"message": f"Failed to start emulator: {e}", "code": 500}), 500

@app.route("/stop_emulator", methods=["POST"])
def stop_emulator():
    emulator_id = None
    if request.is_json:
        emulator_id = (request.json or {}).get("emulator_id")
    if not emulator_id:
        return jsonify({"message": "emulator_id is required", "code": 400}), 400

    if emulator_id not in emulators:
        return jsonify({"message": "Emulator not found", "code": 404}), 404

    emu = emulators[emulator_id]
    token = emu.token

    # Validate token ownership if required
    requester_token = _extract_token(request)
    if REQUIRE_TOKEN:
        if not requester_token:
            return jsonify({"message": "Token required", "code": 401}), 401
        if emu.token and emu.token != requester_token:
            return jsonify({"message": "Forbidden: token does not own this emulator", "code": 403}), 403

    try:
        emu.stop_emulator()
    except Exception as e:
        logger.error(f"Error stopping emulator {emulator_id}: {e}")

    # Update state
    with lock:
        try:
            if token in token_usage:
                if emulator_id in token_usage[token]["emulator_ids"]:
                    token_usage[token]["emulator_ids"].remove(emulator_id)
                if token_usage[token]["current"] > 0:
                    token_usage[token]["current"] -= 1
        finally:
            # Remove emulator record
            emulators.pop(emulator_id, None)

    return jsonify({"message": "Emulator stopped successfully", "code": 0})

@app.route("/status", methods=["GET"])
def status():
    # Overall system stats
    try:
        cpu_percent = psutil.cpu_percent(interval=0.1)
        mem = psutil.virtual_memory()
        memory_percent = mem.percent
    except Exception:
        cpu_percent = None
        memory_percent = None

    # Docker stats
    total_image_containers = None
    try:
        dclient = docker.from_env()
        total_image_containers = sum(
            1 for c in dclient.containers.list()
            if c.image and getattr(c.image, "tags", []) and any("happysixd/osworld-docker" in t for t in c.image.tags)
        )
    except Exception:
        total_image_containers = None

    # Tokens snapshot
    with lock:
        tokens_view = [
            {
                "token": t,
                "current": v["current"],
                "limit": v["limit"],
                "available": max(v["limit"] - v["current"], 0),
            }
            for t, v in token_usage.items()
        ]

    return jsonify({
        "uptime_seconds": int(time.time() - server_started_at),
        "cpu_percent": cpu_percent,
        "memory_percent": memory_percent,
        "total_containers_for_image": total_image_containers,
        "total_emulators": len(emulators),
        "tokens": tokens_view
    })

@app.route("/tokens", methods=["GET"])
def tokens():
    with lock:
        result = {
            t: {
                "current": v["current"],
                "limit": v["limit"],
                "emulator_count": len(v["emulator_ids"])
            }
            for t, v in token_usage.items()
        }
    return jsonify(result)

@app.route("/emulators", methods=["GET"])
def list_emulators():
    token_q = request.args.get("token")
    items = []
    for eid, emu in emulators.items():
        if token_q and emu.token != token_q:
            continue
        item = {
            "emulator_id": eid,
            "token": emu.token,
            "vnc_port": emu.provider.vnc_port,
            "chromium_port": emu.provider.chromium_port,
            "vlc_port": emu.provider.vlc_port,
            "server_port": emu.provider.server_port,
            "start_time": emu.start_time,
            "duration_minutes": emu.duration()
        }
        container = getattr(emu.provider, "container", None)
        item["container_id"] = container.id if container else None
        items.append(item)
    return jsonify(items)

@app.route("/emulator_resources", methods=["GET"])
def emulator_resources():
    token_q = request.args.get("token")
    items = []
    for eid, emu in emulators.items():
        if token_q and emu.token != token_q:
            continue
        item = {
            "emulator_id": eid,
            "token": emu.token,
            "vnc_port": emu.provider.vnc_port,
            "chromium_port": emu.provider.chromium_port,
            "vlc_port": emu.provider.vlc_port,
            "server_port": emu.provider.server_port,
            "start_time": emu.start_time,
            "duration_minutes": emu.duration()
        }
        try:
            container = getattr(emu.provider, "container", None)
            if container:
                item["container_id"] = container.id
                item["resources"] = parse_container_stats(container)
            else:
                item["container_id"] = None
                item["resources"] = {"error": "container not available"}
        except Exception as e:
            item["resources"] = {"error": str(e)}
        items.append(item)
    return jsonify(items)

@app.route("/emulator_resources/<emulator_id>", methods=["GET"])
def emulator_resource_single(emulator_id):
    emu = emulators.get(emulator_id)
    if not emu:
        return jsonify({"message": "Emulator not found", "code": 404}), 404
    item = {
        "emulator_id": emulator_id,
        "token": emu.token,
        "vnc_port": emu.provider.vnc_port,
        "chromium_port": emu.provider.chromium_port,
        "vlc_port": emu.provider.vlc_port,
        "server_port": emu.provider.server_port,
        "start_time": emu.start_time,
        "duration_minutes": emu.duration()
    }
    container = getattr(emu.provider, "container", None)
    if container:
        item["container_id"] = container.id
        item["resources"] = parse_container_stats(container)
    else:
        item["container_id"] = None
        item["resources"] = {"error": "container not available"}
    return jsonify(item)

@app.route("/emulators_by_token", methods=["GET"])
def emulators_by_token():
    """
    Convenience endpoint: GET /emulators_by_token?token=xxx
    Returns only emulator_id list for given token.
    """
    token_q = request.args.get("token")
    if not token_q:
        return jsonify({"message": "token is required", "code": 400}), 400
    ids = [eid for eid, emu in emulators.items() if emu.token == token_q]
    return jsonify(ids)

@app.route("/emulator_status/<emulator_id>", methods=["GET"])
def emulator_status(emulator_id):
    """
    Explicit status endpoint. Returns running if emulator exists, else 404 with not_found code.
    """
    emu = emulators.get(emulator_id)
    if not emu:
        return jsonify({"message": "Emulator not found", "code": 404, "status": "not_found"}), 404
    container = getattr(emu.provider, "container", None)
    return jsonify({
        "emulator_id": emulator_id,
        "status": "running",
        "token": emu.token,
        "container_id": getattr(container, "id", None),
        "vnc_port": emu.provider.vnc_port,
        "chromium_port": emu.provider.chromium_port,
        "vlc_port": emu.provider.vlc_port,
        "server_port": emu.provider.server_port,
        "duration_minutes": emu.duration()
    })

@app.route("/emulators/<emulator_id>", methods=["DELETE"])
def delete_emulator(emulator_id):
    """
    RESTful delete to stop a given emulator by id. Honors REQUIRE_TOKEN ownership check.
    """
    emu = emulators.get(emulator_id)
    if not emu:
        return jsonify({"message": "Emulator not found", "code": 404}), 404

    requester_token = _extract_token(request)
    if REQUIRE_TOKEN:
        if not requester_token:
            return jsonify({"message": "Token required", "code": 401}), 401
        if emu.token and emu.token != requester_token:
            return jsonify({"message": "Forbidden: token does not own this emulator", "code": 403}), 403

    try:
        emu.stop_emulator()
    except Exception as e:
        logger.error(f"Error stopping emulator {emulator_id}: {e}")

    with lock:
        token = emu.token
        try:
            if token in token_usage:
                if emulator_id in token_usage[token]["emulator_ids"]:
                    token_usage[token]["emulator_ids"].remove(emulator_id)
                if token_usage[token]["current"] > 0:
                    token_usage[token]["current"] -= 1
        finally:
            emulators.pop(emulator_id, None)

    return jsonify({"message": "Emulator stopped successfully", "code": 0})

@app.route("/set_token_limit", methods=["POST"])
def set_token_limit():
    if not request.is_json:
        return jsonify({"message": "JSON body required", "code": 400}), 400
    data = request.get_json(silent=True) or {}
    token = str(data.get("token", "")).strip()
    limit = data.get("limit")
    try:
        limit = int(limit)
    except Exception:
        return jsonify({"message": "Valid integer 'limit' is required", "code": 400}), 400
    if not token:
        return jsonify({"message": "'token' is required", "code": 400}), 400
    if token not in TOKEN_LIMITS:
        return jsonify({"message": f"Unknown token '{token}'. Token creation is disabled.", "code": 400}), 400

    with lock:
        TOKEN_LIMITS[token] = limit
        _ensure_token_initialized(token)
        token_usage[token]["limit"] = limit
    return jsonify({"message": f"Token '{token}' limit set to {limit}", "code": 0})

@app.route("/stop_all_emulators", methods=["POST"])
def stop_all_emulators():
    """
    Stop all emulators currently in memory.
    Honors REQUIRE_TOKEN - only stops emulators owned by the requester's token.
    """
    requester_token = _extract_token(request)
    if REQUIRE_TOKEN and not requester_token:
        return jsonify({"message": "Token required", "code": 401}), 401
    
    stopped_count = 0
    failed_count = 0
    stopped_ids = []
    
    # Get list of emulators to stop (filter by token if REQUIRE_TOKEN)
    with lock:
        emus_to_stop = []
        for eid, emu in list(emulators.items()):
            if REQUIRE_TOKEN:
                if emu.token == requester_token:
                    emus_to_stop.append((eid, emu))
            else:
                emus_to_stop.append((eid, emu))
    
    # Stop each emulator
    for eid, emu in emus_to_stop:
        try:
            emu.stop_emulator()
            stopped_ids.append(eid)
            stopped_count += 1
            
            # Update state
            with lock:
                token = emu.token
                try:
                    if token in token_usage:
                        if eid in token_usage[token]["emulator_ids"]:
                            token_usage[token]["emulator_ids"].remove(eid)
                        if token_usage[token]["current"] > 0:
                            token_usage[token]["current"] -= 1
                finally:
                    emulators.pop(eid, None)
        except Exception as e:
            logger.error(f"Failed to stop emulator {eid}: {e}")
            failed_count += 1
    
    return jsonify({
        "message": f"Stopped {stopped_count} emulator(s), {failed_count} failed",
        "code": 0,
        "stopped_count": stopped_count,
        "failed_count": failed_count,
        "stopped_ids": stopped_ids
    })

@app.route("/remove_all_containers", methods=["POST"])
def remove_all_containers():
    """
    Remove Docker containers using Docker SDK with filters.
    Supports min_age (minutes), image filter, and dry_run mode.
    """
    requester_token = _extract_token(request)
    if REQUIRE_TOKEN and not requester_token:
        return jsonify({"message": "Token required", "code": 401}), 401
    
    # Parse parameters
    data = request.get_json(silent=True) or {}
    min_age = int(data.get("min_age", 0))
    image_name = str(data.get("image", "happysixd/osworld-docker")).strip()
    dry_run = bool(data.get("dry_run", False))
    
    try:
        dclient = docker.from_env()
    except Exception as e:
        logger.error(f"Failed to connect to Docker: {e}")
        return jsonify({"message": f"Failed to connect to Docker: {e}", "code": 500}), 500
    
    removed_count = 0
    failed_count = 0
    removed_ids = []
    
    try:
        # Get all containers (including stopped ones)
        all_containers = dclient.containers.list(all=True)
        
        # Filter containers by image
        matching_containers = []
        for container in all_containers:
            try:
                image_tags = container.image.tags if container.image else []
                if any(image_name in tag for tag in image_tags):
                    matching_containers.append(container)
            except Exception as e:
                logger.warning(f"Failed to check image for container {container.id[:12]}: {e}")
                continue
        
        if not matching_containers:
            return jsonify({
                "message": f"No containers found for image '{image_name}'",
                "code": 0,
                "removed_count": 0,
                "failed_count": 0,
                "removed_ids": []
            })
        
        # Process each matching container
        for container in matching_containers:
            try:
                container_id = container.id[:12]
                
                # Calculate age
                created_str = container.attrs['Created']
                created_dt = datetime.fromisoformat(created_str.replace('Z', '+00:00'))
                now = datetime.now(datetime.timezone.utc)
                age_minutes = int((now - created_dt).total_seconds() / 60)
                
                # Check if container is old enough
                if age_minutes >= min_age:
                    if dry_run:
                        logger.info(f"[DRY RUN] Would remove container {container_id}")
                        removed_ids.append(container_id)
                        removed_count += 1
                    else:
                        try:
                            container.remove(force=True)
                            logger.info(f"Successfully removed container {container_id}")
                            removed_ids.append(container_id)
                            removed_count += 1
                        except Exception as e:
                            logger.error(f"Failed to remove container {container_id}: {e}")
                            failed_count += 1
                else:
                    logger.info(f"Container {container_id} is too young (age: {age_minutes} min, threshold: {min_age} min)")
            except Exception as e:
                logger.error(f"Error processing container: {e}")
                failed_count += 1
                continue
        
    except Exception as e:
        logger.error(f"Error listing containers: {e}")
        return jsonify({"message": f"Error listing containers: {e}", "code": 500}), 500
    
    return jsonify({
        "message": f"{'[DRY RUN] Would remove' if dry_run else 'Removed'} {removed_count} container(s), {failed_count} failed",
        "code": 0,
        "removed_count": removed_count,
        "failed_count": failed_count,
        "removed_ids": removed_ids,
        "dry_run": dry_run
    })

@app.route("/cleanup", methods=["POST"])
def cleanup():
    """
    Legacy cleanup endpoint - now uses absolute path resolution.
    """
    try:
        script_path = os.path.join(BASE_DIR, "scripts", "remove_all.sh")
        if not os.path.exists(script_path):
            return jsonify({"message": f"Script not found at {script_path}", "code": 404}), 404
        
        ret = os.system(f"bash {script_path}")
        return jsonify({"message": "Cleanup executed", "code": 0, "ret": ret})
    except Exception as e:
        logger.error(f"Cleanup error: {e}")
        return jsonify({"message": f"Cleanup error: {e}", "code": 500}), 500

@app.route("/dashboard", methods=["GET"])
def dashboard():
    # Simple HTML page rendered from template
    return render_template("dashboard.html")

# Static assets or screenshots passthrough if ever needed
@app.route("/screenshot/<path:filename>", methods=["GET"])
def get_screenshot(filename):
    # This endpoint is kept for compatibility with old readiness checks
    # There is no actual screenshot served here by this server; return 404.
    return abort(404)

if __name__ == '__main__':
    # Initialize tokens from config.yaml
    with lock:
        for t, lim in (TOKEN_LIMITS or {}).items():
            if t not in token_usage:
                token_usage[t] = {"current": 0, "limit": int(lim), "emulator_ids": set()}
            else:
                token_usage[t]["limit"] = int(lim)
    port = int(os.getenv("OSWORLD_SERVER_PORT", _cfg_get("remote_docker_server.port", 50003)))
    app.run(debug=True, host="0.0.0.0", port=port)
