import logging
import os
from typing import Any, Dict, Optional

from video_knowledge import get_task_video_knowledge

try:
    from gui_agents.s3.agents.agent_s import AgentS3
    from gui_agents.s3.agents.grounding import OSWorldACI
except ImportError:
    from new_gui_agents_with_video.s3.agents.agent_s import AgentS3
    from new_gui_agents_with_video.s3.agents.grounding import OSWorldACI


logger = logging.getLogger("desktopenv.agent_s3")


def _clean_env_value(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    cleaned = value.strip().strip('"').strip("'").strip()
    return cleaned or None


def _openai_api_key() -> Optional[str]:
    return _clean_env_value(os.getenv("OPENAI_API_KEY_2")) or _clean_env_value(
        os.getenv("OPENAI_API_KEY")
    )


class AgentS3WaaAgent:
    """WindowsAgentArena adapter for AgentS3 with optional GUIDE knowledge."""

    action_space = "pyautogui"

    def __init__(
        self,
        env: Any,
        model: str,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: Optional[float] = None,
        max_trajectory_length: int = 8,
        screen_width: int = 1920,
        screen_height: int = 1200,
        grounding_width: Optional[int] = None,
        grounding_height: Optional[int] = None,
        video_json: Optional[str] = None,
        enable_reflection: bool = True,
    ) -> None:
        self.env = env
        self.model = model
        self.video_json = video_json
        self.current_video_planning: Optional[str] = None
        self.current_video_grounding: Optional[str] = None
        self.current_video_meta: Dict[str, Any] = {}
        self.current_domain = ""
        self.current_example_id = ""

        engine_params = {
            "engine_type": "openai",
            "model": model,
            "base_url": _clean_env_value(base_url) or _clean_env_value(os.getenv("OPENAI_BASE_URL")),
            "api_key": _clean_env_value(api_key) or _openai_api_key(),
            "temperature": temperature,
            "timeout": float(os.getenv("AGENT_S3_OPENAI_TIMEOUT", "180")),
        }
        grounding_params = {
            **engine_params,
            "grounding_width": grounding_width or screen_width,
            "grounding_height": grounding_height or screen_height,
        }

        self.grounding_agent = OSWorldACI(
            env=env,
            platform="windows",
            engine_params_for_generation=engine_params,
            engine_params_for_grounding=grounding_params,
            width=screen_width,
            height=screen_height,
        )
        self.agent = AgentS3(
            engine_params,
            self.grounding_agent,
            platform="windows",
            max_trajectory_length=max_trajectory_length,
            enable_reflection=enable_reflection,
        )

    def set_task_context(
        self, domain: str, example_id: str, example: Dict[str, Any], args: Any
    ) -> None:
        self.current_video_planning = None
        self.current_video_grounding = None
        self.current_video_meta = {}
        self.current_domain = domain
        self.current_example_id = example_id

        video_json = getattr(args, "video_json", None) or self.video_json
        if not video_json:
            logger.info("No video JSON configured for %s/%s", domain, example_id)
            return

        planning, grounding, meta = get_task_video_knowledge(
            video_json_path=video_json,
            example_id=example_id,
            domain=domain,
            example=example,
            grounding_max_k=getattr(args, "grounding_max_k", 0),
        )

        if not getattr(args, "enable_planning", False):
            planning = None
        if not getattr(args, "enable_grounding", False):
            grounding = None

        self.current_video_planning = planning
        self.current_video_grounding = grounding
        self.current_video_meta = meta

        logger.info(
            "Loaded GUIDE knowledge for %s/%s: planning=%s grounding=%s source=%s match_id=%s",
            domain,
            example_id,
            bool(planning),
            bool(grounding),
            meta.get("source"),
            meta.get("id"),
        )

    def reset(
        self,
        *args: Any,
        video_planning: Optional[str] = None,
        video_grounding: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        if video_planning is None:
            video_planning = self.current_video_planning
        if video_grounding is None:
            video_grounding = self.current_video_grounding
        self.agent.reset(video_planning=video_planning, video_grounding=video_grounding)

    def predict(self, instruction: str, obs: Dict[str, Any]):
        info, actions = self.agent.predict(instruction, obs)
        if actions is None:
            actions = []
        if isinstance(actions, str):
            actions = [actions]

        info = info or {}
        response = info.get("plan") or info.get("plan_code") or info.get("exec_code") or ""
        logs = {
            "user_question": instruction,
            "plan_result": response,
            "video_source": self.current_video_meta.get("source", ""),
            "video_match_id": self.current_video_meta.get("id", ""),
            "video_count": self.current_video_meta.get("video_count", 0),
            "converted_video_count": self.current_video_meta.get("converted_video_count", 0),
            "agent_s3_plan_code": info.get("plan_code", ""),
            "agent_s3_exec_code": info.get("exec_code", ""),
            "agent_s3_reflection": info.get("reflection", ""),
        }
        return response, list(actions), logs, None
