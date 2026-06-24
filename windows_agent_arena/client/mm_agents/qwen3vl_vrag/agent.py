import logging
import re
from typing import Any, Dict, List, Optional

from video_knowledge import get_task_video_knowledge

from .base_agent import Qwen3VLAgent


logger = logging.getLogger("desktopenv.qwen3vl_vrag")


_TYPEWRITE_RE = re.compile(r"(pyautogui\.(?:typewrite|write)\()\s*(['\"])(.*?)(\2)(\s*\).*)", re.DOTALL)


class Qwen3VLVragAgent(Qwen3VLAgent):
    """WindowsAgentArena adapter for the OSWorld Qwen3VL + V-RAG agent."""

    def __init__(self, *args: Any, video_json: Optional[str] = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.video_json = video_json
        self.current_video_planning: Optional[str] = None
        self.current_video_grounding: Optional[str] = None
        self.current_video_meta: Dict[str, Any] = {}
        self.current_domain: str = ""
        self.current_example_id: str = ""

    def set_task_context(self, domain: str, example_id: str, example: Dict[str, Any], args: Any) -> None:
        self.current_video_planning = None
        self.current_video_grounding = None
        self.current_video_meta = {}
        self.current_domain = domain
        self.current_example_id = example_id

        video_json = getattr(args, "video_json", None) or self.video_json
        if not video_json:
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

        if planning or grounding:
            logger.info(
                "Loaded video knowledge for %s/%s: planning=%s grounding=%s source=%s",
                domain,
                example_id,
                bool(planning),
                bool(grounding),
                meta.get("source"),
            )
        else:
            logger.info("No enabled video knowledge for %s/%s", domain, example_id)

    def reset(self, _logger=None, video_planning=None, video_grounding=None):
        if video_planning is None:
            video_planning = self.current_video_planning
        if video_grounding is None:
            video_grounding = self.current_video_grounding
        super().reset(_logger or logger, video_planning=video_planning, video_grounding=video_grounding)

    def predict(self, instruction: str, obs: Dict[str, Any]):
        response, actions = super().predict(instruction, obs)

        if actions is None:
            actions = []
        if isinstance(actions, str):
            actions = [actions]

        actions = self._normalize_actions(response, list(actions))
        logs = {
            "user_question": instruction,
            "plan_result": response or "",
            "video_source": self.current_video_meta.get("source", ""),
            "video_match_id": self.current_video_meta.get("id", ""),
            "video_count": self.current_video_meta.get("video_count", 0),
            "converted_video_count": self.current_video_meta.get("converted_video_count", 0),
        }
        return response, actions, logs, None

    @staticmethod
    def _normalize_actions(response: str, actions: List[str]) -> List[str]:
        if not actions:
            logger.warning("Qwen3VL response produced no executable action; using WAIT.")
            return ["WAIT"]

        response_text = response or ""
        failed_terminate = (
            re.search(r'"action"\s*:\s*"terminate"', response_text)
            and re.search(r'"status"\s*:\s*"failure"', response_text)
        )
        if failed_terminate:
            return ["FAIL" if action == "DONE" else action for action in actions]

        return [Qwen3VLVragAgent._escape_multiline_typewrite(action) for action in actions]

    @staticmethod
    def _escape_multiline_typewrite(action: str) -> str:
        if "\n" not in action:
            return action

        match = _TYPEWRITE_RE.fullmatch(action)
        if not match:
            return action

        prefix, _, text, _, suffix = match.groups()
        return f"{prefix}{text!r}{suffix}"
