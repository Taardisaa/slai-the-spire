"""Adapter from CommunicationMod JSON to ViewGameState.

Only fills what the RL path reads. Fields the encoder doesn't consume are left
at defaults. See `src/rl/encoding/*.py` for what each sub-encoder touches.
"""

import logging

from src.game.core.fsm import FSM
from src.game.entity.actor import ModifierType
from src.game.entity.card import CardColor
from src.game.entity.card import CardRarity
from src.game.entity.card import CardType
from src.game.entity.map_node import RoomType
from src.game.entity.monster import Intent
import src.game.factory  # noqa: F401 — populates FACTORY_LIB_* via registration
from src.game.factory.lib import FACTORY_LIB_CARD as _CARD_LIB
from src.game.view.card import ViewCard
from src.game.view.character import ViewCharacter
from src.game.view.energy import ViewEnergy
from src.game.view.map_ import ViewMap
from src.game.view.map_ import ViewMapNode
from src.game.view.monster import ViewMonster
from src.game.view.state import ViewGameState
from src.rl.comm_mod.names import sim_card_name
from src.rl.comm_mod.names import sim_monster_name


log = logging.getLogger(__name__)


# CommMod power id -> sim ModifierType. Missing entries are dropped silently.
_POWER_TO_MODIFIER: dict[str, ModifierType] = {
    "Vulnerable": ModifierType.VULNERABLE,
    "Weakened": ModifierType.WEAK,
    "Weak": ModifierType.WEAK,
    "Strength": ModifierType.STRENGTH,
    "Dexterity": ModifierType.DEXTERITY,
    "Ritual": ModifierType.RITUAL,
    "AfterImage": ModifierType.AFTER_IMAGE,
    "After Image": ModifierType.AFTER_IMAGE,
    "Blur": ModifierType.BLUR,
    "Burst": ModifierType.BURST,
    "DoubleDamage": ModifierType.DOUBLE_DAMAGE,
    "Double Damage": ModifierType.DOUBLE_DAMAGE,
    "InfiniteBlades": ModifierType.INFINITE_BLADES,
    "Infinite Blades": ModifierType.INFINITE_BLADES,
    "ModeShift": ModifierType.MODE_SHIFT,
    "Mode Shift": ModifierType.MODE_SHIFT,
    "NextTurnBlock": ModifierType.NEXT_TURN_BLOCK,
    "NextTurnEnergy": ModifierType.NEXT_TURN_ENERGY,
    "Phantasmal": ModifierType.PHANTASMAL,
    "SharpHide": ModifierType.SHARP_HIDE,
    "Sharp Hide": ModifierType.SHARP_HIDE,
    "SporeCloud": ModifierType.SPORE_CLOUD,
    "Spore Cloud": ModifierType.SPORE_CLOUD,
    "ThousandCuts": ModifierType.THOUSAND_CUTS,
    "Thousand Cuts": ModifierType.THOUSAND_CUTS,
    "Accuracy": ModifierType.ACCURACY,
}

_SYMBOL_TO_ROOM_TYPE: dict[str, RoomType] = {
    "M": RoomType.COMBAT_MONSTER,
    "E": RoomType.COMBAT_MONSTER,  # elites — sim has no elite room type
    "B": RoomType.COMBAT_BOSS,
    "R": RoomType.REST_SITE,
    # The sim has no Shop/Event/Treasure room types — collapse to COMBAT_MONSTER.
    "$": RoomType.COMBAT_MONSTER,
    "?": RoomType.COMBAT_MONSTER,
    "T": RoomType.COMBAT_MONSTER,
}


def _convert_powers(powers: list[dict]) -> dict[ModifierType, int | None]:
    out: dict[ModifierType, int | None] = {}
    for p in powers or []:
        name = p.get("name") or p.get("id") or ""
        mod = _POWER_TO_MODIFIER.get(name)
        if mod is None:
            continue
        amt = p.get("amount")
        out[mod] = amt
    return out


def _template_card(sim_name: str) -> ViewCard:
    """Build a ViewCard from a factory-generated EntityCard template.

    Encoder reads: name, cost, effects, exhaust, innate, requires_target,
    requires_discard. Other fields are preserved from the template.
    """
    base = sim_name.rstrip("+")
    upgraded = sim_name.endswith("+")
    factory = _CARD_LIB.get(base)
    if factory is None:
        # Last-resort placeholder — should not happen because `sim_card_name`
        # already falls back to 'Strike'.
        factory = _CARD_LIB["Strike"]
        base = "Strike"

    entity = factory(upgraded)
    from src.game.utils import does_card_require_discard
    from src.game.utils import does_card_require_target

    return ViewCard(
        name=entity.name,
        color=entity.color,
        type=entity.type,
        rarity=entity.rarity,
        cost=entity.cost,
        effects=entity.effects,
        exhaust=entity.exhaust,
        innate=entity.innate,
        is_active=False,
        requires_target=does_card_require_target(entity),
        requires_discard=does_card_require_discard(entity),
    )


def _convert_card(raw: dict) -> ViewCard:
    card_id = raw.get("id") or raw.get("name") or ""
    upgrades = int(raw.get("upgrades", 0))
    sim_name = sim_card_name(card_id, upgrades)
    view = _template_card(sim_name)
    # Real cost may differ (X-cost, cost modifiers, free this turn).
    cost = raw.get("cost")
    if isinstance(cost, int) and cost >= 0:
        from dataclasses import replace

        view = replace(view, cost=cost)
    return view


def _convert_monster(raw: dict) -> ViewMonster | None:
    if raw.get("is_gone") or raw.get("half_dead"):
        return None
    monster_id = raw.get("id") or raw.get("name") or ""
    sim_name = sim_monster_name(monster_id)
    # Use a factory to get a plausible max_health if `max_hp` missing.
    hp_max = int(raw.get("max_hp", raw.get("current_hp", 1)) or 1)
    hp_cur = int(raw.get("current_hp", hp_max) or 0)
    block = int(raw.get("block", 0) or 0)
    powers = _convert_powers(raw.get("powers", []))
    intent = Intent(
        damage=raw.get("move_adjusted_damage") if raw.get("move_adjusted_damage", -1) >= 0 else None,
        instances=raw.get("move_hits"),
        block=bool(raw.get("intent", "").lower().startswith("defend")),
        buff=bool("buff" in str(raw.get("intent", "")).lower()),
        debuff_powerful=bool("debuff" in str(raw.get("intent", "")).lower()),
    )
    return ViewMonster(
        name=sim_name,
        health_current=hp_cur,
        health_max=hp_max,
        block_current=block,
        modifiers=powers,
        intent=intent,
    )


def _convert_character(player: dict, msg_game: dict) -> ViewCharacter:
    hp_max = int(msg_game.get("max_hp", player.get("max_hp", 1)) or 1)
    hp_cur = int(msg_game.get("current_hp", player.get("current_hp", hp_max)) or 0)
    block = int(player.get("block", 0) or 0)
    powers = _convert_powers(player.get("powers", []))
    # sim Silent character name
    return ViewCharacter(
        name="silent",
        health_current=hp_cur,
        health_max=hp_max,
        block_current=block,
        modifiers=powers,
        card_reward_roll_offset=0,
    )


def _convert_map(msg_game: dict) -> ViewMap:
    """Build ViewMap.nodes (list of rows) from CommMod map[]."""
    nodes_raw = msg_game.get("map", []) or []
    if not nodes_raw:
        return ViewMap(nodes=[], y_current=None, x_current=None)

    max_y = max(n["y"] for n in nodes_raw)
    # ViewMap.nodes is a list of MAP_HEIGHT rows, width MAP_WIDTH (=7).
    rows: list[list[ViewMapNode | None]] = [
        [None] * 7 for _ in range(max_y + 1)
    ]
    for n in nodes_raw:
        y = n["y"]
        x = n["x"]
        if not (0 <= x < 7):
            continue
        symbol = n.get("symbol", "M")
        room_type = _SYMBOL_TO_ROOM_TYPE.get(symbol, RoomType.COMBAT_MONSTER)
        x_next = {c["x"] for c in n.get("children", []) if 0 <= c.get("x", -1) < 7}
        rows[y][x] = ViewMapNode(room_type=room_type, x_next=x_next)

    y_cur = msg_game.get("floor")
    # `floor` is 1-indexed in CommMod; map y is 0-indexed.
    y_current = (y_cur - 1) if isinstance(y_cur, int) and y_cur > 0 else None
    # x_current: CommMod doesn't expose it directly; attempt from screen_state
    # for map screen or leave None (indicates "pre-first-floor").
    x_current = None
    screen = msg_game.get("screen_state", {}) or {}
    current_node = screen.get("current_node")
    if isinstance(current_node, dict) and "x" in current_node:
        x_current = current_node["x"]

    return ViewMap(nodes=rows, y_current=y_current, x_current=x_current)


def _derive_fsm(msg_game: dict) -> FSM:
    screen = (msg_game.get("screen_type") or "").upper()
    room_phase = (msg_game.get("room_phase") or "").upper()
    if screen == "COMBAT_REWARD" or screen == "CARD_REWARD":
        return FSM.CARD_REWARD
    if screen == "MAP":
        return FSM.MAP
    if screen == "REST":
        return FSM.REST_SITE
    if room_phase == "COMBAT":
        combat = msg_game.get("combat_state") or {}
        # Heuristics for targeting state; not perfect but the RL agent emits
        # single-shot PLAY commands so we rarely end up here.
        if combat.get("awaiting_target"):
            return FSM.COMBAT_AWAIT_TARGET_CARD
        return FSM.COMBAT_DEFAULT
    return FSM.COMBAT_DEFAULT


def to_view_game_state(msg: dict) -> ViewGameState:
    """Translate a full CommunicationMod state message to a ViewGameState."""
    game = msg.get("game_state", {}) or {}
    combat = game.get("combat_state") or {}

    player = combat.get("player", {}) or {}
    energy = ViewEnergy(
        current=int(player.get("energy", 0) or 0),
        max=3,
    )
    character = _convert_character(player, game)
    monsters = [m for m in (_convert_monster(m) for m in combat.get("monsters", [])) if m]

    hand = [_convert_card(c) for c in combat.get("hand", [])]
    pile_draw = [_convert_card(c) for c in combat.get("draw_pile", [])]
    pile_disc = [_convert_card(c) for c in combat.get("discard_pile", [])]
    pile_exhaust = [_convert_card(c) for c in combat.get("exhaust_pile", [])]
    deck = [_convert_card(c) for c in game.get("deck", [])]

    reward_combat: list[ViewCard] = []
    screen_state = game.get("screen_state", {}) or {}
    if game.get("screen_type") in ("CARD_REWARD", "COMBAT_REWARD"):
        for r in screen_state.get("cards", []) or []:
            reward_combat.append(_convert_card(r))

    return ViewGameState(
        character=character,
        monsters=monsters,
        deck=deck,
        hand=hand,
        pile_draw=pile_draw,
        pile_disc=pile_disc,
        pile_exhaust=pile_exhaust,
        reward_combat=reward_combat,
        energy=energy,
        map=_convert_map(game),
        fsm=_derive_fsm(game),
    )
