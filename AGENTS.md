# AGENTS.md

Guide for coding agents working in `slai-the-spire`.

## Repo Snapshot

- Language: Python 3.10+
- Package manager: Poetry
- Source root: `src/`
- Main areas: `src/game/` for simulation and `src/rl/` for RL/training
- No existing `AGENTS.md` was present in this repo root
- No Cursor rules were found in `.cursor/rules/` or `.cursorrules`
- No Copilot instructions were found in `.github/copilot-instructions.md`

## Setup

- Install dependencies with `poetry install`
- Run commands through Poetry: `poetry run ...`
- Imports are rooted at `src`, for example `from src.game.action import Action`

## Build, Format, Lint, Test

### Install

- `poetry install`

### Format

- `poetry run isort .`
- `poetry run black .`
- Preferred full pass: `poetry run isort . && poetry run black .`

### Lint

- `poetry run flake8 .`

### Tests

- All tests: `poetry run pytest`
- Verbose: `poetry run pytest -vv`
- Single file: `poetry run pytest path/to/test_file.py`
- Single test: `poetry run pytest path/to/test_file.py::test_name`
- Single class test: `poetry run pytest path/to/test_file.py::TestClass::test_name`
- Name filter: `poetry run pytest -k "mask or actor_critic"`

### Current Test Reality

- There is no dedicated `tests/` directory in the current repository state
- There is no repo-level pytest config file
- `pytest` exists as a dev dependency, but the repo may currently collect zero tests
- If you add tests, prefer standard pytest discovery under `tests/`

## Useful Smoke Commands

- Game loop: `poetry run python -m src.game.main`
- PPO training: `poetry run python -m src.rl.algorithms.actor_critic.master`
- Single-threaded A2C debug training: `poetry run python -m src.rl.algorithms.actor_critic.train`
- Trained-agent renderer: `poetry run python -m src.rl.test_agent --exp-path experiments/ppo_hierarchical_v1`
- Legacy harness: `poetry run python -m src.rl.legacy.test`

## Validation Guidance

- Start with the smallest relevant check
- Prefer a single pytest node id over a full run when iterating
- If no pytest tests exist for the touched area, use the smallest relevant module entrypoint as a smoke test
- Do not use long RL training runs as routine validation
- Do not start million-episode training as a check for small code changes

## Important Config Paths

- PPO config: `src/rl/algorithms/actor_critic/config.yml`
- Legacy DQN config: `src/rl/legacy/dqn_algorithm/config.yml`
- Training scripts write artifacts under `experiments/<exp_name>/`
- Be careful not to overwrite experiment outputs unless the task calls for it

## Formatting Rules

- Black line length is `99`
- isort line length is `99`
- Flake8 line length is `99`
- Flake8 ignores `E203` and `W503`
- isort uses `force_single_line = true`
- `typing`, `abc`, and `dataclasses` are excluded from single-line splitting

## Import Style

- Use absolute imports from `src.*`
- Do not switch files to relative imports
- Keep imports grouped as standard library, third-party, then local `src.*`
- Prefer one imported symbol per line for local imports, matching repo style
- Example:

```python
from dataclasses import dataclass

import torch

from src.game.action import Action
from src.game.action import ActionType
```

## Typing And Data Structures

- Use modern Python 3.10 type syntax: `list[int]`, `dict[str, Any]`, `Foo | None`
- Add parameter and return annotations for new or changed functions
- Use `@dataclass` for structured records
- Use `@dataclass(frozen=True)` for immutable view or encoding snapshots when appropriate
- Use `NamedTuple` only when tuple semantics are actually helpful
- Keep tensor container types explicit, as in `XGameState` and `CoreOutput`
- Preserve useful tensor shape comments in model code

## Naming Conventions

- Modules: `snake_case`
- Classes: `PascalCase`
- Functions and variables: `snake_case`
- Constants: `UPPER_SNAKE_CASE`
- Enum members: `UPPER_SNAKE_CASE`
- Internal helpers: prefix with `_`
- Preserve existing trailing-underscore names for reserved words, such as `map_` and `types_`

## Code Organization

- Prefer small direct helpers over extra abstraction
- Follow nearby file patterns before introducing a new pattern
- Extend existing dataclasses and enums instead of introducing ad hoc dict-based structures
- In RL/model code, keep the forward path easy to trace end-to-end
- Avoid adding compatibility layers or framework-like indirection without a concrete need

## Error Handling

- Fail fast on invalid states and broken invariants
- Use specific exceptions such as `ValueError`, `RuntimeError`, `NotImplementedError`, or local custom exceptions
- Catch exceptions only when you can recover or improve the boundary behavior
- Narrow environmental exception handling is fine, for example `OSError` around terminal sizing
- Avoid silent fallback behavior unless the code already requires it
- If you touch nearby code with placeholder messages like `TODO: add message`, prefer improving them

## Comments And Docstrings

- Add docstrings when they clarify intent, invariants, or shape contracts
- Keep comments focused on why, non-obvious behavior, or ordering constraints
- Avoid comments that merely restate the code
- Shape comments in RL/model code are meaningful and should usually be preserved

## Testing Expectations For New Work

- Add or update pytest coverage when feasible
- Prefer deterministic unit tests over long-running integration or training tests
- For game logic, test state transitions and action validity
- For RL/model code, test tensor shapes, masks, routing, and invariants
- Re-run the narrowest relevant test command first, then broaden if needed

## Practical Agent Notes

- Read adjacent modules before editing
- Keep changes surgical
- Prefer the active actor-critic path over legacy code unless the task explicitly targets legacy code
- Use configured tools for import sorting and formatting instead of manual cleanup
- Expect some existing TODOs and placeholder messages in the repo; do not spread those patterns into new code
