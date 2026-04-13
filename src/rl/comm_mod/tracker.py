"""Per-run telemetry: accumulate combat/elite/boss/relic info, emit summary at game-over."""

import logging


log = logging.getLogger(__name__)


class RunTracker:
    def __init__(self, agent_name: str = "rl_agent") -> None:
        self.agent = agent_name
        self.reset()

    def reset(self) -> None:
        self.seed: str | None = None
        self.floor: int = 0
        self.score: int = 0
        self.victory: bool = False
        self._last_monsters: list[str] = []  # most recent combat's monster names
        self._elites: list[str] = []
        self._bosses: list[str] = []
        self._summary_emitted: bool = False

    def update(self, msg: dict) -> None:
        game = msg.get("game_state") or {}
        if not game:
            return

        if self.seed is None:
            s = game.get("seed_played") or game.get("seed")
            if s is not None:
                self.seed = str(s)

        floor = game.get("floor")
        if isinstance(floor, int):
            self.floor = floor

        room_type = (game.get("room_type") or "").upper()
        combat = game.get("combat_state") or {}
        if combat:
            monsters = [
                (m.get("name") or m.get("id") or "?")
                for m in combat.get("monsters", [])
                if not m.get("is_gone")
            ]
            if monsters:
                self._last_monsters = monsters
                if "ELITE" in room_type:
                    for n in monsters:
                        if n not in self._elites:
                            self._elites.append(n)
                elif "BOSS" in room_type or room_type == "MONSTER_ROOM_BOSS":
                    for n in monsters:
                        if n not in self._bosses:
                            self._bosses.append(n)

        screen_type = (game.get("screen_type") or "").upper()
        if screen_type == "GAME_OVER" and not self._summary_emitted:
            screen_state = game.get("screen_state") or {}
            self.victory = bool(screen_state.get("victory", False))
            self.score = int(screen_state.get("score", 0) or 0)
            self._emit_summary(game)

    def _emit_summary(self, game: dict) -> None:
        relics = [r.get("name") or r.get("id") or "?" for r in game.get("relics", []) or []]
        died_to = ",".join(self._last_monsters) if not self.victory else ""
        parts = [
            f"Seed:{self.seed or '?'}",
            f"Floor:{self.floor}",
            f"Score:{self.score}",
            f"Agent: {self.agent}",
            f"DiedTo: {died_to}",
            f"Bosses: {','.join(self._bosses)}",
            f"Elites: {','.join(self._elites)}",
            f"Relics: {','.join(relics)}",
        ]
        line = ", ".join(parts)
        log.info("RUN_SUMMARY %s", line)
        self._summary_emitted = True
