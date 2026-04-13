#!/usr/bin/env bash
# Launcher for CommunicationMod to spawn.
#
# Point CommunicationMod's config.properties at this script:
#     command=/mnt/aigo/hao/slai-the-spire/scripts/comm_mod_launch.sh
#
# CommMod launches this with the StS process's cwd, so we must cd into the repo.
# All stderr / logs go to comm_mod.log; stdout is reserved for the mod protocol.

set -euo pipefail

REPO="/mnt/aigo/hao/slai-the-spire"
EXP_PATH="${EXP_PATH:-experiments/ppo/test_new_grp}"
CHARACTER="${CHARACTER:-SILENT}"
ASCENSION="${ASCENSION:-0}"
DEVICE="${DEVICE:-cpu}"

cd "$REPO"

exec poetry run python -m src.rl.comm_mod.main \
    --exp-path "$EXP_PATH" \
    --character "$CHARACTER" \
    --ascension "$ASCENSION" \
    --device "$DEVICE" \
    ${SEED:+--seed "$SEED"} \
    ${GREEDY:+--greedy}
