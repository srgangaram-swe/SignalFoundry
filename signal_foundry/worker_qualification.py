"""Recompute the preserved AlphaForge rubric in its independently locked runtime."""

from __future__ import annotations

from typing import Any

from signal_foundry.boundary import FoundryError


def verify(payload: dict[str, Any]) -> dict[str, Any]:
    from alphaforge.broker.config import BrokerSessionConfig, authorize_paper_session
    from alphaforge.research.qualification import (
        EvidenceLink,
        QualificationError,
        qualify,
        standard_paper_rubric,
    )

    try:
        rubric = standard_paper_rubric()
        decision = qualify(
            candidate_id=payload["candidate_id"],
            rubric=rubric,
            expected_rubric_identity=payload["rubric_identity"],
            observations=payload["observations"],
            evidence={
                key: tuple(EvidenceLink(**link) for link in links)
                for key, links in payload["evidence"].items()
            },
            plan_hash=payload["plan_hash"],
            decided_at=payload["decided_at"],
        )
        authorize_paper_session(
            BrokerSessionConfig(
                enabled=True,
                endpoint="https://paper-api.alpaca.markets",
                keychain_service="com.signal-foundry.alpaca-paper",
            ),
            qualification=decision,
            environment={},
        )
        result = decision.to_dict()
        if not isinstance(result, dict):
            raise FoundryError(
                "paper_qualification", "Invalid source qualification result."
            )
        return result
    except (KeyError, TypeError, ValueError, QualificationError) as exc:
        raise FoundryError(
            "paper_qualification",
            "The unchanged source paper-qualification gate refused this dossier.",
            409,
        ) from exc
