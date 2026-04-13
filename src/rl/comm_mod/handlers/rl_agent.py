"""RL-driven handler: combat, map-node pick, card-reward pick."""

import logging

import torch

from src.rl.action_space.masks import get_masks
from src.rl.comm_mod.adapter import to_view_game_state
from src.rl.comm_mod.command import action_to_command
from src.rl.encoding.state import encode_batch_view_game_state
from src.rl.models import ActorCritic


log = logging.getLogger(__name__)


class RLHandler:
    def __init__(self, model: ActorCritic, device: torch.device, greedy: bool = False) -> None:
        self.model = model
        self.device = device
        self.greedy = greedy

    def decide(self, msg: dict) -> str:
        view = to_view_game_state(msg)
        x = encode_batch_view_game_state([view], self.device)
        primary_mask, secondary_masks = get_masks(view, self.device)

        # Intersect with available_commands: if `end` isn't available,
        # mask out COMBAT_TURN_END; if no monsters, mask MONSTER_SELECT.
        available = set(msg.get("available_commands", []) or [])
        if "end" not in available:
            from src.rl.action_space.types import ActionChoice

            primary_mask[0, ActionChoice.COMBAT_TURN_END] = False
        if "play" not in available:
            from src.rl.action_space.types import ActionChoice

            primary_mask[0, ActionChoice.CARD_PLAY] = False

        # If every primary is masked, fall back to a plain "end" / "proceed".
        if not primary_mask.any():
            log.warning("no valid primary actions; emitting fallback")
            return "end" if "end" in available else "proceed"

        with torch.no_grad():
            out = self.model.forward_single(
                x, primary_mask, secondary_masks, sample=not self.greedy
            )
        action = out.to_action()
        return action_to_command(action, msg)
