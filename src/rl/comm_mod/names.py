"""Name mappings from CommunicationMod `id` strings to the in-sim names.

The in-house sim currently ships only Silent cards and a small Act 1 monster
pool. Real StS play will surface many unknown `id` strings. This module maps
real game ids to the closest sim equivalent so the encoder doesn't crash on
KeyError. When a real id has no reasonable sim analogue we map to a placeholder
('Strike' for cards, 'Cultist' for monsters) and log a warning.

Drift between the mapped tensor and the real game state is expected and is
exactly what the eval harness is designed to measure — do not try to fix it
here.
"""

import logging

from src.game.factory.lib import FACTORY_LIB_CARD
from src.game.factory.lib import FACTORY_LIB_MONSTER


log = logging.getLogger(__name__)


# CommunicationMod card ids use PascalCase with no spaces (e.g. "Strike_R",
# "Defend_G", "EruptionWatcher", "Halt"). Sim card names are lowercase with
# underscores. Build a best-effort alias table — anything missing falls back to
# "strike".
_SIM_CARDS = set(FACTORY_LIB_CARD.keys())
_SIM_MONSTERS = set(FACTORY_LIB_MONSTER.keys())

_CARD_ALIASES: dict[str, str] = {
    "Strike_R": "Strike",
    "Strike_G": "Strike",
    "Strike_B": "Strike",
    "Strike_P": "Strike",
    "Defend_R": "Defend",
    "Defend_G": "Defend",
    "Defend_B": "Defend",
    "Defend_P": "Defend",
    "Neutralize": "Neutralize",
    "Survivor": "Survivor",
    "Shiv": "Shiv",
    # Watcher common / uncommon / rare — approximate mappings to sim analogues
    "Eruption": "Strike",
    "Vigilance": "Defend",
    "FlurryOfBlows": "Strike",
    "FollowUp": "Strike",
    "CrushJoints": "Strike",
    "CutThroughFate": "Strike",
    "EmptyBody": "Defend",
    "EmptyMind": "Survivor",
    "Halt": "Defend",
    "Prostrate": "Defend",
    "Protect": "Defend",
    "SashWhip": "Strike",
    "TheBomb": "Strike",
    "Evaluate": "Defend",
    "ThirdEye": "Defend",
    "Tranquility": "Defend",
    "Consecrate": "Strike",
    "Crescendo": "Defend",
    "ClearTheMind": "Defend",
    "Pray": "Defend",
    "Scrawl": "Survivor",
    "WreathOfFlame": "Strike",
    "AscendersBane": "Shiv",
}

_MONSTER_ALIASES: dict[str, str] = {
    "Cultist": "Cultist",
    "FungiBeast": "Fungi Beast",
    "JawWorm": "Jaw Worm",
    "GreenLouse": "Cultist",
    "LouseGreen": "Cultist",
    "RedLouse": "Cultist",
    "LouseRed": "Cultist",
    "AcidSlime_L": "Cultist",
    "AcidSlime_M": "Cultist",
    "AcidSlime_S": "Cultist",
    "SpikeSlime_L": "Cultist",
    "SpikeSlime_M": "Cultist",
    "SpikeSlime_S": "Cultist",
    "Looter": "Cultist",
    "Mugger": "Cultist",
    "GremlinFat": "Cultist",
    "GremlinThief": "Cultist",
    "GremlinTsundere": "Cultist",
    "GremlinWarrior": "Cultist",
    "GremlinWizard": "Cultist",
    "GremlinNob": "Jaw Worm",
    "Lagavulin": "Jaw Worm",
    "Sentry": "Cultist",
    "Hexaghost": "The Guardian",
    "SlimeBoss": "The Guardian",
    "TheGuardian": "The Guardian",
}


def sim_card_name(card_id: str, upgrades: int) -> str:
    """Map a CommMod card id + upgrade count to a sim card name."""
    base = _CARD_ALIASES.get(card_id)
    if base is None:
        if card_id in _SIM_CARDS:
            base = card_id
        else:
            log.warning("unknown card id %r, falling back to 'Strike'", card_id)
            base = "Strike"

    if upgrades > 0:
        return base + "+"
    return base


def sim_monster_name(monster_id: str) -> str:
    """Map a CommMod monster id to a sim monster name."""
    name = _MONSTER_ALIASES.get(monster_id)
    if name is None:
        if monster_id in _SIM_MONSTERS:
            name = monster_id
        else:
            log.warning("unknown monster id %r, falling back to 'Cultist'", monster_id)
            name = "Cultist"
    return name
