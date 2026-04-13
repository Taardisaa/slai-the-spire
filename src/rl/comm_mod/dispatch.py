"""Screen-type dispatch: route a CommMod state message to the right handler."""

import logging

from src.rl.comm_mod.handlers import rules
from src.rl.comm_mod.handlers.rl_agent import RLHandler
from src.rl.comm_mod.tracker import RunTracker


log = logging.getLogger(__name__)


class Dispatcher:
    """Routes incoming messages to either the RL handler (combat / map /
    card-reward) or a rule-based handler (everything else)."""

    def __init__(
        self,
        rl: RLHandler,
        character: str = "SILENT",
        ascension: int = 0,
        seed: str | None = None,
    ) -> None:
        self.rl = rl
        self.character = character
        self.ascension = ascension
        self.seed = seed
        self._started = False
        self.tracker = RunTracker(agent_name=f"rl_{character.lower()}")

    def on_message(self, msg: dict) -> str | None:
        # Error messages can arrive without game_state; resync via STATE.
        if "error" in msg:
            log.warning("mod error: %s", msg.get("error"))
            return "state"

        self.tracker.update(msg)

        if not msg.get("ready_for_command", False):
            return None

        available = set(msg.get("available_commands", []) or [])

        # Not in a run → start one (and reset tracker between runs).
        if not msg.get("in_game", False):
            if self._started:
                self.tracker.reset()
                self._started = False
            if "start" in available:
                self._started = True
                seed_part = f" {self.seed}" if self.seed else ""
                return f"start {self.character} {self.ascension}{seed_part}"
            # Main menu navigation fallback.
            return rules.handle_default(msg)

        game = msg.get("game_state", {}) or {}
        screen_type = (game.get("screen_type") or "NONE").upper()
        room_phase = (game.get("room_phase") or "").upper()

        # RL-driven screens
        if room_phase == "COMBAT" and screen_type in ("NONE", ""):
            return self.rl.decide(msg)
        if screen_type == "MAP":
            return self.rl.decide(msg)
        if screen_type in ("CARD_REWARD",):
            return self.rl.decide(msg)

        # Rule-based
        handler = rules.HANDLERS.get(screen_type, rules.handle_default)
        try:
            return handler(msg)
        except Exception:
            log.exception("rule handler failed for %s; using default", screen_type)
            return rules.handle_default(msg)
