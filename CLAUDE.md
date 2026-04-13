# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

A comprehensive agent guide already exists at `AGENTS.md` — read it for setup, style, typing, naming, and testing conventions. This file highlights the essentials and the architectural big picture.

## Common Commands

- Install: `poetry install`
- Format: `poetry run isort . && poetry run black .`
- Lint: `poetry run flake8 .`
- Tests: `poetry run pytest` (single test: `poetry run pytest path::test_name`, filter: `-k "mask or actor_critic"`)
- Run the game loop: `poetry run python -m src.game.main`
- PPO training (multi-worker master): `poetry run python -m src.rl.algorithms.actor_critic.master`
- Single-threaded A2C debug training: `poetry run python -m src.rl.algorithms.actor_critic.train`
- Render a trained agent: `poetry run python -m src.rl.test_agent --exp-path experiments/ppo_hierarchical_v1`

Line length is 99 across black/isort/flake8; isort uses `force_single_line = true`. Imports are absolute from `src.*`.

There is currently no `tests/` directory — pytest may collect zero tests. Use the smallest relevant `python -m ...` entrypoint as a smoke check. Do not run long RL training as routine validation.

## Architecture

Two top-level packages under `src/`:

### `src/game/` — Slay-the-Spire simulator

- `state.py`, `action.py`, `types_.py`, `const.py` — core state, action types, and enums. Note trailing-underscore names (`map_`, `types_`) avoid reserved-word clashes; preserve them.
- `core/` — FSM and effect primitives driving turn flow.
- `engine/` — effect queue and `process_effect/` handlers that mutate state.
- `entity/`, `factory/`, `create.py` — entities (cards, monsters, relics) and their construction.
- `level/`, `map_.py` — map/room generation and traversal.
- `view/`, `draw.py` — terminal rendering.
- `main.py` — interactive game loop entrypoint.

### `src/rl/` — RL stack over the simulator

The forward path is traceable end-to-end; follow it rather than searching for abstractions.

- `encoding/` — converts a `GameState` into tensor containers (per-file: `card`, `monster`, `character`, `map_`, `health_block`, etc.) aggregated by `state.py` into an `XGameState`. Encodings are typically frozen dataclasses.
- `action_space/` — `types.py` defines the hierarchical action space; `masks.py` produces validity masks per state.
- `models/` — PyTorch modules:
  - `entity_projector.py` + `entity_transformer.py` encode variable-sized entity sets.
  - `map_encoder.py` encodes the map.
  - `core.py` fuses these into a `CoreOutput`.
  - `heads.py` contains policy/value heads (hierarchical over action space).
  - `actor_critic.py` wires core + heads into the full actor-critic model.
- `policies.py` — sampling strategies (e.g. grouped greedy) over masked action logits.
- `reward.py` — reward shaping.
- `algorithms/actor_critic/` — PPO training:
  - `master.py` is the multi-worker PPO driver; `worker.py` runs rollouts; `train.py` is a single-threaded A2C debug variant.
  - `config.yml` holds hyperparameters; experiments are written to `experiments/<exp_name>/`. Do not overwrite those unless asked.
- `test_agent.py` — loads a checkpoint and renders play; useful for qualitative checks.
- `legacy/` — older DQN harness (`legacy/dqn_algorithm/config.yml`, `legacy/test`). Prefer the actor-critic path unless a task explicitly targets legacy code.

### Key invariants when editing model/RL code

- Tensor shape comments in model code are load-bearing — preserve them.
- Extend existing dataclasses/enums rather than introducing dict-based structures.
- Keep the `GameState → encoding → core → heads → action` path easy to trace; avoid framework-like indirection.
- Masks must stay aligned with the hierarchical action-space layout in `action_space/types.py`.
