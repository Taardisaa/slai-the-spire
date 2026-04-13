"""Stdio client for CommunicationMod.

CommunicationMod spawns this process and talks over stdin/stdout with newline-
terminated JSON (inbound) and newline-terminated command strings (outbound).
Anything written to stdout that isn't a command will desync the mod, so all
logging goes to stderr / a file.
"""

import json
import logging
import sys
from typing import Callable
from typing import Iterator


log = logging.getLogger(__name__)


class CommModClient:
    """Line-oriented bridge to CommunicationMod.

    Usage:
        client = CommModClient()
        client.ready()
        for msg in client.messages():
            cmd = decide(msg)
            client.send(cmd)
    """

    def __init__(
        self,
        stdin=None,
        stdout=None,
    ) -> None:
        self._stdin = stdin if stdin is not None else sys.stdin
        self._stdout = stdout if stdout is not None else sys.stdout

    def ready(self) -> None:
        self._write("ready")

    def send(self, command: str) -> None:
        if not command:
            raise ValueError("empty command")
        log.info("-> %s", command)
        self._write(command)

    def messages(self) -> Iterator[dict]:
        for line in self._stdin:
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                log.warning("skipping non-JSON line: %r", line[:200])
                continue
            yield msg

    def _write(self, text: str) -> None:
        self._stdout.write(text + "\n")
        self._stdout.flush()


def run_loop(
    client: CommModClient,
    on_message: Callable[[dict], str | None],
) -> None:
    """Drive the client with a message -> command callback.

    If `on_message` returns None, no command is sent (useful for `ready_for_command: false`
    states where we just wait for the next frame).
    """
    client.ready()
    for msg in client.messages():
        try:
            cmd = on_message(msg)
        except Exception:
            log.exception("on_message failed; sending STATE to resync")
            client.send("state")
            continue

        if cmd is None:
            continue

        client.send(cmd)
