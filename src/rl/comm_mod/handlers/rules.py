"""Rule-based handlers for screens the RL agent wasn't trained on.

These are intentionally simple heuristics modelled on bottled_ai's smart_agent
defaults. The goal is to keep the run progressing without making novel
decisions in spaces the agent doesn't cover.
"""

import logging
from typing import Callable


log = logging.getLogger(__name__)


# Rough Watcher card priority for shop/event/card-reward-outside-combat.
# Higher index = more desired.
_CARD_PRIORITY = [
    # Rares
    "Blasphemy",
    "Devotion",
    "Judgment",
    "MasterReality",
    "Alpha",
    # Strong uncommons
    "Meditate",
    "FollowUp",
    "Eruption",
    "Wallop",
    "Worship",
    # Generic decent
    "Protect",
    "Vigilance",
]

# Boss relic rough priority (higher = more desired).
_BOSS_RELIC_PRIORITY = [
    "Cloak Clasp",
    "Runic Pyramid",
    "Sozu",
    "Orrery",
    "Astrolabe",
]


def handle_neow(msg: dict) -> str:
    opts = _screen_choices(msg)
    # Prefer max-HP / remove-card; else first option.
    for i, name in enumerate(opts):
        if "max hp" in name.lower():
            return f"choose {i}"
    for i, name in enumerate(opts):
        if "remove" in name.lower():
            return f"choose {i}"
    return "choose 0"


def handle_event(msg: dict) -> str:
    opts = _screen_choices(msg)
    # Prefer "leave" / "skip" when available to avoid risky events.
    for i, name in enumerate(opts):
        low = name.lower()
        if low.startswith("leave") or low.startswith("skip") or low.startswith("ignore"):
            return f"choose {i}"
    # Else take first option.
    return "choose 0"


def handle_chest(msg: dict) -> str:
    # Always open the chest.
    available = set(msg.get("available_commands", []) or [])
    if "choose" in available:
        return "choose 0"
    return "proceed"


def handle_combat_reward(msg: dict) -> str:
    """Post-combat reward screen: take gold/potion, skip cards handled elsewhere."""
    screen_state = (msg.get("game_state", {}) or {}).get("screen_state", {}) or {}
    rewards = screen_state.get("rewards", []) or []
    for i, r in enumerate(rewards):
        kind = (r.get("reward_type") or "").lower()
        if kind in ("gold", "stolen_gold", "relic", "potion", "emerald_key", "sapphire_key"):
            return f"choose {i}"
    # Cards reward: skip — card-reward screen is handled by RL on the next state.
    return "proceed"


def handle_campfire(msg: dict) -> str:
    player = (msg.get("game_state", {}) or {})
    hp = player.get("current_hp", 1)
    mx = player.get("max_hp", 1)
    ratio = hp / max(mx, 1)
    opts = _screen_choices(msg)
    # Smith when healthy, rest otherwise.
    preferred = "smith" if ratio > 0.70 else "rest"
    fallback = "rest" if preferred == "smith" else "smith"
    for opt in (preferred, fallback):
        for i, name in enumerate(opts):
            if opt in name.lower():
                return f"choose {i}"
    return "choose 0"


def handle_shop_entrance(msg: dict) -> str:
    available = set(msg.get("available_commands", []) or [])
    if "choose" in available:
        return "choose 0"
    return "proceed"


def handle_shop(msg: dict) -> str:
    # Leave — buying decisions are outside the scope of this eval.
    available = set(msg.get("available_commands", []) or [])
    if "leave" in available:
        return "leave"
    if "cancel" in available:
        return "cancel"
    if "proceed" in available:
        return "proceed"
    return "choose 0"


def handle_boss_reward(msg: dict) -> str:
    opts = _screen_choices(msg)
    # Prefer by priority list, else skip.
    best_i = None
    best_rank = -1
    for i, name in enumerate(opts):
        for rank, preferred in enumerate(reversed(_BOSS_RELIC_PRIORITY)):
            if preferred.lower() in name.lower() and rank > best_rank:
                best_i, best_rank = i, rank
                break
    if best_i is not None:
        return f"choose {best_i}"
    available = set(msg.get("available_commands", []) or [])
    if "skip" in available:
        return "skip"
    return "choose 0"


def handle_grid_select(msg: dict) -> str:
    """Purge / transform / upgrade / armaments grid.

    For single-pick grids (remove card, upgrade), pick the highest-index
    "bad" card (strike, defend, curse) or the first card as fallback.
    """
    screen_state = (msg.get("game_state", {}) or {}).get("screen_state", {}) or {}
    cards = screen_state.get("cards", []) or []
    # Prefer curses / basic cards.
    bad = []
    for i, c in enumerate(cards):
        cid = (c.get("id") or "").lower()
        if "curse" in cid or cid.startswith("strike") or cid.startswith("defend"):
            bad.append(i)
    if bad:
        return f"choose {bad[0]}"
    if cards:
        return "choose 0"
    return "proceed"


def handle_hand_select(msg: dict) -> str:
    """Mass-discard style screen (e.g. Headbutt in combat).

    Pick the minimum required number of cards from the end of the hand.
    """
    screen_state = (msg.get("game_state", {}) or {}).get("screen_state", {}) or {}
    hand = screen_state.get("hand", []) or []
    num_cards = screen_state.get("num_cards", 1) or 1
    if not hand:
        return "confirm"
    # CHOOSE by index; StS will auto-confirm when the required count is reached.
    idx = max(0, len(hand) - 1)
    return f"choose {idx}"


def handle_default(msg: dict) -> str:
    """Fallback: PROCEED / CONFIRM / continue."""
    available = set(msg.get("available_commands", []) or [])
    for cmd in ("proceed", "confirm", "continue", "skip", "cancel", "return"):
        if cmd in available:
            return cmd
    if "choose" in available:
        return "choose 0"
    return "wait 10"


def _screen_choices(msg: dict) -> list[str]:
    screen_state = (msg.get("game_state", {}) or {}).get("screen_state", {}) or {}
    choices = screen_state.get("choice_list") or screen_state.get("options") or []
    return [str(c).lower() for c in choices]


HANDLERS: dict[str, Callable[[dict], str]] = {
    "EVENT": handle_event,
    "CHEST": handle_chest,
    "COMBAT_REWARD": handle_combat_reward,
    "REST": handle_campfire,
    "SHOP_ROOM": handle_shop_entrance,
    "SHOP_SCREEN": handle_shop,
    "BOSS_REWARD": handle_boss_reward,
    "GRID": handle_grid_select,
    "HAND_SELECT": handle_hand_select,
    "MAP": handle_default,  # RL handler takes precedence in dispatch
    "NONE": handle_default,
}
