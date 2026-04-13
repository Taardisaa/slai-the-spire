"""Convert a sampled RL Action into a CommunicationMod command string.

Key translations:
- PLAY uses 1-indexed hand positions.
- Monster targets are 0-indexed.
- CHOOSE falls back to numeric index when name is unknown.
"""

from src.game.action import Action
from src.game.action import ActionType


def action_to_command(
    action: Action,
    msg: dict,
) -> str:
    game = msg.get("game_state", {}) or {}
    combat = game.get("combat_state") or {}

    match action.type:
        case ActionType.COMBAT_TURN_END:
            return "end"

        case ActionType.COMBAT_CARD_IN_HAND_SELECT:
            # action.index is 0-based hand position; CommMod PLAY is 1-based.
            hand = combat.get("hand", []) or []
            idx = action.index or 0
            idx = max(0, min(idx, len(hand) - 1))
            card = hand[idx] if hand else {}
            pos = idx + 1
            if card.get("has_target"):
                target = _pick_default_target(combat)
                return f"play {pos} {target}"
            return f"play {pos}"

        case ActionType.COMBAT_MONSTER_SELECT:
            # Used when a previous card is waiting for a target (rare — our PLAY
            # usually includes the target). Emit a CHOOSE on the monster.
            idx = action.index or 0
            return f"choose {idx}"

        case ActionType.CARD_REWARD_SELECT:
            idx = action.index or 0
            return f"choose {idx}"

        case ActionType.CARD_REWARD_SKIP:
            # In CommMod, skipping a card reward uses "skip" or "proceed"
            # depending on screen state; "skip" is generally available.
            available = msg.get("available_commands", []) or []
            if "skip" in available:
                return "skip"
            return "proceed"

        case ActionType.MAP_NODE_SELECT:
            # action.index is the target x column; CHOOSE by x index among
            # available map choices.
            idx = action.index or 0
            return f"choose {idx}"

        case ActionType.REST_SITE_REST:
            return "choose rest"

        case ActionType.REST_SITE_UPGRADE:
            return "choose smith"

    raise ValueError(f"unhandled action type: {action.type}")


def _pick_default_target(combat: dict) -> int:
    """Return the 0-indexed monster to target when none was chosen."""
    monsters = combat.get("monsters", []) or []
    for i, m in enumerate(monsters):
        if not m.get("is_gone") and not m.get("half_dead"):
            return i
    return 0
