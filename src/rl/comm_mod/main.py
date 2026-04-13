"""Entrypoint for CommunicationMod to spawn.

Example `config.properties` line for the mod:
    command=poetry run python -m src.rl.comm_mod.main --exp-path experiments/<name>

The mod pipes newline-JSON state over stdin and expects newline-terminated
commands on stdout. Everything else (logs, tracebacks) goes to stderr or a
log file — anything on stdout that isn't a command will desync the mod.
"""

import logging
import os
import sys

import click
import torch

from src.rl.comm_mod.client import CommModClient
from src.rl.comm_mod.client import run_loop
from src.rl.comm_mod.dispatch import Dispatcher
from src.rl.comm_mod.handlers.rl_agent import RLHandler
from src.rl.models import ActorCritic
from src.rl.utils import load_config


def _setup_logging(exp_path: str, level: str) -> None:
    log_path = os.path.join(exp_path, "comm_mod.log")
    os.makedirs(exp_path, exist_ok=True)
    handlers = [logging.FileHandler(log_path), logging.StreamHandler(sys.stderr)]
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=handlers,
    )


def _load_model(exp_path: str, checkpoint: str | None, device: torch.device) -> ActorCritic:
    config = load_config(f"{exp_path}/config.yml")
    model = ActorCritic(**config["model"])
    ckpt_name = checkpoint or "model.pth"
    state = torch.load(f"{exp_path}/{ckpt_name}", map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    model.to(device)
    return model


@click.command()
@click.option("--exp-path", default="experiments/ppo_hierarchical_v1")
@click.option("--checkpoint", default=None)
@click.option("--device", default="cpu")
@click.option("--character", default="SILENT", help="STS class: IRONCLAD, SILENT, DEFECT, WATCHER")
@click.option("--ascension", default=0, type=int)
@click.option("--seed", default=None, type=str)
@click.option("--greedy", is_flag=True, help="Deterministic argmax instead of sampling")
@click.option("--log-level", default="INFO")
def main(
    exp_path: str,
    checkpoint: str | None,
    device: str,
    character: str,
    ascension: int,
    seed: str | None,
    greedy: bool,
    log_level: str,
) -> None:
    _setup_logging(exp_path, log_level)
    log = logging.getLogger(__name__)
    log.info("starting CommMod bridge: exp=%s character=%s asc=%d", exp_path, character, ascension)

    dev = torch.device(device)
    model = _load_model(exp_path, checkpoint, dev)
    rl = RLHandler(model, dev, greedy=greedy)
    dispatcher = Dispatcher(rl, character=character, ascension=ascension, seed=seed)

    client = CommModClient()
    try:
        run_loop(client, dispatcher.on_message)
    except KeyboardInterrupt:
        log.info("interrupted")
    except Exception:
        log.exception("fatal error in run loop")
        raise


if __name__ == "__main__":
    main()
