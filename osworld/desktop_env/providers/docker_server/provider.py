import logging
import os
import platform
import time
import docker
import psutil
import requests
from filelock import FileLock
from pathlib import Path
from urllib.parse import urlparse

from desktop_env.providers.base import Provider
from desktop_env.utils import load_config

config = load_config()
logger = logging.getLogger("desktopenv.providers.docker.DockerProvider")
logger.setLevel(logging.INFO)

WAIT_TIME = 3
RETRY_INTERVAL = 1
LOCK_TIMEOUT = 10


class PortAllocationError(Exception):
    pass


class RemoteDockerProvider(Provider):
    def __init__(self, region: str, remote_docker_server_ip: str = config.remote_docker_server.ip, remote_docker_server_port: int = config.remote_docker_server.port):
        # self.client = docker.from_env()
        # remote_docker_server_ip = 'YOUR_SERVER_IP'
        print('remote docker server ip',remote_docker_server_ip)
        self.server_port = None
        self.vnc_port = None
        self.chromium_port = None
        self.vlc_port = None
        self.container = None
        self.emulator_id = None
        self.environment = {"DISK_SIZE": "2G", "RAM_SIZE": "2G", "CPU_CORES": "2"}  # Modify if needed
        
        # Parse URL list from environment variable
        self.server_urls = []  # List of (ip, port) tuples
        base_url = os.getenv("OSWORLD_BASE_URL")
        
        if base_url:
            # Support comma-separated URL list
            url_list = [url.strip() for url in base_url.split(',')]
            for url in url_list:
                try:
                    u = urlparse(url)
                    if not u.scheme:
                        u = urlparse("http://" + url)
                    host = u.hostname or remote_docker_server_ip
                    port = u.port or remote_docker_server_port
                    self.server_urls.append((host, int(port)))
                except Exception as e:
                    logger.warning(f"Failed to parse URL '{url}': {e}")
        else:
            # Fallback to individual env vars or config defaults
            ip_env = os.getenv("OSWORLD_REMOTE_DOCKER_IP")
            port_env = os.getenv("OSWORLD_REMOTE_DOCKER_PORT")
            host = ip_env or remote_docker_server_ip
            port = int(port_env) if port_env else remote_docker_server_port
            self.server_urls.append((host, port))
        
        if not self.server_urls:
            raise ValueError("No valid server URLs configured")
        
        # Set initial server (will be updated when emulator starts)
        self.remote_docker_server_ip = self.server_urls[0][0]
        self.remote_docker_server_port = self.server_urls[0][1]
        
        # token for quota/auth; prefer env OSWORLD_TOKEN, fallback to config.client_token if present
        self.token = os.getenv("OSWORLD_TOKEN")
        temp_dir = Path(os.getenv('TEMP') if platform.system() == 'Windows' else '/tmp')
        self.lock_file = temp_dir / "docker_port_allocation.lck"
        self.lock_file.parent.mkdir(parents=True, exist_ok=True)

    def _get_used_ports(self):
        """Get all currently used ports (both system and Docker)."""
        # Get system ports
        system_ports = set(conn.laddr.port for conn in psutil.net_connections())
        
        # Get Docker container ports
        docker_ports = set()
        for container in self.client.containers.list():
            ports = container.attrs['NetworkSettings']['Ports']
            if ports:
                for port_mappings in ports.values():
                    if port_mappings:
                        docker_ports.update(int(p['HostPort']) for p in port_mappings)
        
        return system_ports | docker_ports

    def _get_available_port(self, start_port: int) -> int:
        """Find next available port starting from start_port."""
        used_ports = self._get_used_ports()
        port = start_port
        while port < 65354:
            if port not in used_ports:
                return port
            port += 1
        raise PortAllocationError(f"No available ports found starting from {start_port}")

    def _wait_for_vm_ready(self, timeout: int = 300):
        """Wait for VM to be ready by checking screenshot endpoint."""
        start_time = time.time()
        
        def check_screenshot():
            try:
                response = requests.get(
                    f"http://localhost:{self.server_port}/screenshot",
                    timeout=(10, 10)
                )
                return response.status_code == 200
            except Exception:
                return False

        while time.time() - start_time < timeout:
            if check_screenshot():
                return True
            logger.info("Checking if virtual machine is ready...")
            time.sleep(RETRY_INTERVAL)
        
        raise TimeoutError("VM failed to become ready within timeout period")

    def check_quota(self, server_ip=None, server_port=None):
        """
        Check current token quota status for a specific server.
        Returns dict with quota information or raises RuntimeError if check fails.
        """
        token = self.token or os.getenv("OSWORLD_TOKEN")
        if not token:
            raise RuntimeError("Token is required for quota check")
        
        # Use provided server or current server
        ip = server_ip or self.remote_docker_server_ip
        port = server_port or self.remote_docker_server_port
        
        url = f"http://{ip}:{port}/tokens"
        headers = {"Authorization": f"Bearer {token}"}
        
        try:
            resp = requests.get(url, headers=headers, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            
            if token not in data:
                raise RuntimeError(f"Token '{token}' not found or unauthorized")
            
            token_info = data[token]
            quota_info = {
                "token": token,
                "server": f"{ip}:{port}",
                "current": token_info["current"],
                "limit": token_info["limit"],
                "available": token_info["limit"] - token_info["current"],
                "can_start": token_info["current"] < token_info["limit"]
            }
            return quota_info
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Failed to check quota on {ip}:{port}: {e}")

    def start_emulator(self, path_to_vm: str, headless: bool, os_type: str,
                       max_quota_retries: int = 30, quota_retry_interval: int = 30):
        """
        Start emulator via remote docker server with round-robin polling across multiple servers.
        Tries each server in the list until one succeeds or all fail.
        When all servers are at quota, retries with backoff instead of failing immediately.

        Args:
            max_quota_retries: Max number of retry rounds when all servers are at quota (default 30 = ~15 min).
            quota_retry_interval: Seconds to wait between retry rounds (default 30).
        """
        token = self.token or os.getenv("OSWORLD_TOKEN")

        for retry_round in range(max_quota_retries + 1):
            errors = []
            all_quota_full = True  # Track if every server failed only due to quota

            for idx, (server_ip, server_port) in enumerate(self.server_urls, 1):
                logger.info(f"Trying server {idx}/{len(self.server_urls)}: {server_ip}:{server_port}")

                try:
                    # Check quota for this specific server
                    if token:
                        quota = self.check_quota(server_ip, server_port)
                        if not quota["can_start"]:
                            error_msg = (
                                f"Server {server_ip}:{server_port} quota exceeded: "
                                f"{quota['current']}/{quota['limit']} VMs running"
                            )
                            logger.warning(error_msg)
                            errors.append(error_msg)
                            continue  # Try next server

                        logger.info(
                            f"Quota check passed for server {server_ip}:{server_port}: "
                            f"{quota['current']}/{quota['limit']} VMs in use, {quota['available']} available"
                        )

                    all_quota_full = False  # At least one server has quota available

                    # Try to start emulator on this server
                    url = f"http://{server_ip}:{server_port}/start_emulator"
                    if token:
                        headers = {"Authorization": f"Bearer {token}"}
                        resp = requests.post(url, json={"token": str(token)}, headers=headers, timeout=30)
                    else:
                        resp = requests.get(url, timeout=30)

                    data = resp.json()

                    # Validate response
                    if resp.status_code != 200 or data.get("code") != 0 or "data" not in data:
                        error_msg = f"Server {server_ip}:{server_port} failed: status={resp.status_code}, payload={data}"
                        logger.warning(error_msg)
                        errors.append(error_msg)
                        all_quota_full = False
                        continue  # Try next server

                    # Success! Update current server info and return
                    self.remote_docker_server_ip = server_ip
                    self.remote_docker_server_port = server_port
                    self.emulator_id = data["data"]["emulator_id"]
                    self.server_port = data["data"]["server_port"]
                    self.vnc_port = data["data"]["vnc_port"]
                    self.chromium_port = data["data"]["chromium_port"]
                    self.vlc_port = data["data"]["vlc_port"]
                    logger.info(
                        f"Successfully started emulator {self.emulator_id} on server {server_ip}:{server_port}"
                    )
                    return  # Success, exit function

                except Exception as e:
                    error_msg = f"Server {server_ip}:{server_port} error: {str(e)}"
                    logger.warning(error_msg)
                    errors.append(error_msg)
                    all_quota_full = False
                    continue  # Try next server

            # All servers tried this round — decide whether to retry
            if all_quota_full and retry_round < max_quota_retries:
                logger.info(
                    f"All servers at quota capacity. Waiting {quota_retry_interval}s before retry "
                    f"({retry_round + 1}/{max_quota_retries})..."
                )
                time.sleep(quota_retry_interval)
                continue  # Retry the whole server list
            else:
                # Non-quota failure or retries exhausted — give up
                break

        error_summary = "\n".join([f"  - {err}" for err in errors])
        raise RuntimeError(
            f"Failed to start emulator on all {len(self.server_urls)} servers:\n{error_summary}"
        )

    def get_ip_address(self, path_to_vm: str) -> str:
        if not all([self.server_port, self.chromium_port, self.vnc_port, self.vlc_port]):
            raise RuntimeError("VM not started - ports not allocated")
        return f"{self.remote_docker_server_ip}:{self.server_port}:{self.chromium_port}:{self.vnc_port}:{self.vlc_port}"

    def save_state(self, path_to_vm: str, snapshot_name: str):
        raise NotImplementedError("Snapshots not available for Docker provider")

    def revert_to_snapshot(self, path_to_vm: str, snapshot_name: str):
        self.stop_emulator(path_to_vm)

    def stop_emulator(self, path_to_vm: str):
        if self.emulator_id:
            logger.info(f"Stopping VM {self.emulator_id}...")
            headers = {}
            token = self.token or os.getenv("OSWORLD_TOKEN")
            if token:
                headers["Authorization"] = f"Bearer {token}"
            try:
                # Prefer RESTful DELETE endpoint if available
                del_url = f"http://{self.remote_docker_server_ip}:{self.remote_docker_server_port}/emulators/{self.emulator_id}"
                resp = requests.delete(del_url, headers=headers, timeout=20)
                logger.info(f"Stop VM {self.emulator_id} response: {resp.status_code}")
                if resp.status_code in (404, 405):
                    # Fallback to legacy POST /stop_emulator
                    post_url = f"http://{self.remote_docker_server_ip}:{self.remote_docker_server_port}/stop_emulator"
                    resp2 = requests.post(post_url, json={"emulator_id": self.emulator_id}, headers=headers, timeout=20)
                    logger.info(f"Stop VM {self.emulator_id} fallback response: {resp2.status_code}")
                # Brief wait for server to finish cleanup
                time.sleep(2)
            except Exception as e:
                logger.error(f"Error stopping container {self.emulator_id}: {e}")
            finally:
                self.container = None
                self.server_port = None
                self.vnc_port = None
                self.chromium_port = None
                self.vlc_port = None
