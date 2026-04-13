# Running the RL agent against real StS via CommunicationMod

The `src/rl/comm_mod/` package lets [CommunicationMod](https://github.com/ForgottenArbiter/CommunicationMod)
spawn our trained agent as a subprocess and drive a real Slay the Spire run.

## Setup

1. Install CommunicationMod + BaseMod + ModTheSpire.
2. Edit `config/ModTheSpire/CommunicationMod/config.properties` and set:

   ```
   command=poetry run python -m src.rl.comm_mod.main --exp-path experiments/<name> --character WATCHER --ascension 0
   ```

3. Launch StS with CommunicationMod active. The mod starts the subprocess; the
   bridge emits `ready`, waits for state messages, and sends commands back.

## Architecture (short)

```
StS + CommMod  --stdio JSON-->  main.py
                                 │
                          Dispatcher (dispatch.py)
                           │                 │
                       RL screens         Rule screens
                  (combat / map /      (neow, event, shop,
                   card_reward)         campfire, chest, ...)
                           │                 │
                      RLHandler ──► adapter ──► ViewGameState
                      ActorCritic           ─► command string
```

- `adapter.py` translates CommMod JSON → `ViewGameState` (the dataclass
  consumed by the encoder). Unknown card/monster ids fall back to sim-known
  names via `names.py`.
- `command.py` translates the hierarchical `Action` (from the model) back to a
  CommMod command string (`play 2 1`, `end`, `choose 0`, ...).
- `handlers/rl_agent.py` is the RL seam: encode → mask → model → command.
- `handlers/rules.py` handles non-combat screens with simple heuristics.
- `client.py` is the stdio loop; stdout is reserved for command output only.

## Flags

| Flag | Default | Notes |
|---|---|---|
| `--exp-path` | `experiments/ppo_hierarchical_v1` | Checkpoint directory |
| `--checkpoint` | `model.pth` | Checkpoint filename inside `--exp-path` |
| `--character` | `WATCHER` | `IRONCLAD`, `SILENT`, `DEFECT`, `WATCHER` |
| `--ascension` | `0` | 0–20 |
| `--seed` | *(random)* | Optional seed string |
| `--greedy` | off | Use argmax instead of sampling |
| `--device` | `cpu` | `cpu` or `cuda` |
| `--log-level` | `INFO` | Logs go to `<exp-path>/comm_mod.log` and stderr |

## Expected drift

The sim is Silent-only with a small card/monster pool. Watcher runs will hit
many unknown ids; the adapter maps them to the nearest sim analogue (default:
`strike` / `cultist`) and logs a warning. This drift is intentional — the
whole point of real-game eval is to measure how far trained policies
generalise. Do not add compensation logic here.

## Smoke test (no mod required)

```
echo '{"ready_for_command":true,"in_game":false,"available_commands":["start"]}' \
  | poetry run python -m src.rl.comm_mod.main --exp-path experiments/<name>
```

The process should emit `ready`, read the line, and reply with
`start WATCHER 0`. Use this to validate the stdio handshake before plugging
into the mod.
