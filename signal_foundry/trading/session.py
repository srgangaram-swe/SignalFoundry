"""Explicit bounded minute-cadence operator loop; no automatic startup or retry."""

from __future__ import annotations

import time

from signal_foundry.boundary import FoundryError
from signal_foundry.trading.service import Action, PaperResult, PaperService


def run_session(service: PaperService, cycles: int) -> PaperResult:
    """Run at most 390 minute decisions per symbol, then return to the operator.

    Every iteration rechecks source qualification and account/data admission.
    Any error or interruption halts the loop. This never retries an uncertain
    order, resets a stop, starts a daemon, or silently liquidates positions.
    The explicit cancel operation remains available after stopping.
    """
    if type(cycles) is not int or not 1 <= cycles <= 390:
        raise FoundryError("session_budget", "Choose 1–390 explicit minute cycles.")
    deadline = time.monotonic()
    try:
        for index in range(cycles):
            for symbol in service.config.plan.symbols:
                service.action(Action(operation="cycle", symbol=symbol))
            deadline += 60
            if index + 1 < cycles:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise FoundryError(
                        "session_overrun",
                        "A decision cycle exceeded its cadence budget.",
                    )
                time.sleep(remaining)
    except BaseException:
        # Even an interrupt leaves a durable stop. An in-flight request still
        # needs explicit reconciliation/cancellation; no success is fabricated.
        service.action(Action(operation="stop"))
        raise
    return PaperResult(status=service.status())
