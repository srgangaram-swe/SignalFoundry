"""Private worker entry; only fixed operations can use paper/data credentials."""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pydantic import ValidationError

from signal_foundry.boundary import FoundryError, decode, encode
from signal_foundry.trading.engine import Engine
from signal_foundry.trading.service import Action, PaperResult, configuration
from signal_foundry.trading.store import Journal
from signal_foundry.worker import limits


def main() -> int:
    journal: Journal | None = None
    action: Action | None = None
    try:
        limits()
        payload = decode(sys.stdin.buffer.read(16_385))
        if not isinstance(payload, dict) or set(payload) != {
            "state",
            "config",
            "action",
        }:
            raise FoundryError("paper_envelope", "Invalid paper worker envelope.")
        action = Action.model_validate(payload["action"])
        journal = Journal(Path(payload["state"]))
        engine = Engine(
            Path(__file__).resolve().parents[2],
            journal,
            configuration(Path(payload["config"])),
        )
        artifact = None
        account_digest = None
        with journal.exclusive():
            if action.operation == "initialize":
                engine.initialize()
            else:
                engine.bound()
                if action.operation == "probe":
                    account = engine.broker.account()
                    account_digest = account.digest
                    journal.append(
                        "probe",
                        {
                            "account_digest": account.digest,
                            "clock": engine.broker.clock().at.isoformat(),
                        },
                    )
                elif action.operation == "acquire":
                    engine.acquire(action.symbol or "")
                    artifact = journal.get(f"dataset/{action.symbol}")
                elif action.operation == "research":
                    engine.research()
                    artifact = journal.get("research")
                elif action.operation == "qualify":
                    journal.append("qualification", engine.qualification())
                elif action.operation == "start":
                    engine.start()
                elif action.operation == "cycle":
                    engine.cycle(action.symbol or "")
                elif action.operation == "reconcile":
                    engine.reconcile()
                elif action.operation == "record-session":
                    engine.record_session()
                elif action.operation == "campaign":
                    artifact = engine.campaign()
                elif action.operation == "cancel":
                    engine.cancel()
                elif action.operation == "stop":
                    journal.stop()
            journal.set("last_action", action.operation, kind="operation_complete")
            journal.set("last_error", None, kind="operation_success")
            result = {
                "result": PaperResult(
                    status=engine.status(),
                    artifact=artifact,
                    account_digest=account_digest,
                ).wire()
            }
    except FoundryError as exc:
        if journal is not None:
            journal.set("last_error", exc.code, kind="operation_failed")
        result = {
            "error": {"code": exc.code, "detail": exc.detail, "status": exc.status}
        }
    except (
        ValidationError,
        ValueError,
        TypeError,
        KeyError,
        IndexError,
        ArithmeticError,
        OSError,
        sqlite3.Error,
    ) as exc:
        # An owning process boundary: retain the cause class, never vendor/secret
        # payloads. A persisted dispatch remains unresolved until reconciliation.
        result = {
            "error": {
                "code": "paper_contract",
                "detail": (
                    f"Paper operation refused ({type(exc).__name__}); "
                    "inspect the runbook and reconcile persisted intent."
                ),
                "status": 422,
            }
        }
    finally:
        if journal is not None:
            journal.close()
    sys.stdout.buffer.write(encode(result, 65_536))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
