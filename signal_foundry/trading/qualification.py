"""Content-bound evidence verification and unchanged source-rubric dispatch."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from signal_foundry.boundary import FoundryError, code_identity, decode, read_file
from signal_foundry.runner import Runner
from signal_foundry.trading.models import PaperConfig, timestamp
from signal_foundry.trading.store import Journal


def qualification(
    root: Path, journal: Journal, config: PaperConfig, now: datetime
) -> dict[str, Any]:
    """Verify content pointers and re-score the unchanged source rubric.

    Evidence truth and independent review remain the research owner's
    responsibility. Checksums prove identity, not that a supplied metric is
    scientifically valid. No caller-supplied verdict/pass flag is trusted.
    """
    dossier = decode(read_file(journal.root / "qualification.json", 16_384))
    if not isinstance(dossier, dict) or set(dossier) != {
        "config_identity",
        "code_identity",
        "valid_until",
        "decision",
    }:
        raise FoundryError("paper_qualification", "Invalid qualification envelope.")
    expires = timestamp(dossier["valid_until"])
    decision = dossier["decision"]
    if dossier["config_identity"] != config.identity or dossier[
        "code_identity"
    ] != code_identity(root):
        raise FoundryError(
            "paper_qualification",
            "Qualification does not bind this configuration/code.",
        )
    decided = timestamp(decision["decided_at"])
    if not decided <= now < expires or expires - decided > timedelta(days=30):
        raise FoundryError(
            "paper_qualification",
            "Qualification is future-dated, expired or valid too long.",
        )
    if (
        decision["plan_hash"] != config.plan.identity
        or decision["candidate_id"] != config.candidate
    ):
        raise FoundryError(
            "paper_qualification",
            "Qualification does not bind this frozen plan/candidate.",
        )
    observations: dict[str, float] = {}
    evidence: dict[str, list[dict[str, str]]] = {}
    criteria = decision["criteria"]
    if not isinstance(criteria, list) or len(criteria) != 8:
        raise FoundryError(
            "paper_qualification", "Expected every source qualification criterion."
        )
    for criterion in criteria:
        name = criterion["name"]
        if name in observations:
            raise FoundryError("paper_qualification", "Duplicate criterion.")
        observations[name] = criterion["observed"]
        links = criterion["evidence"]
        if not isinstance(links, list) or not 1 <= len(links) <= 8:
            raise FoundryError("paper_qualification", "Invalid evidence inventory.")
        evidence[name] = []
        for link in links:
            value = journal.read_artifact(link["content_hash"])
            if (
                not isinstance(value, dict)
                or value.get("plan_identity") != config.plan.identity
                or value.get("config_identity") != config.identity
                or value.get("evidence_kind") != link["kind"]
                or value.get("evidence_class") != "measured"
                or value.get("fixture_only") is True
            ):
                raise FoundryError(
                    "paper_qualification",
                    "Evidence content does not bind this plan and configuration.",
                )
            evidence[name].append(
                {key: link[key] for key in ("kind", "identifier", "content_hash")}
            )
    result = Runner(root, journal.root / "qualification-worker").call(
        "alphaforge",
        "paper-qualification",
        {
            "request": {
                "candidate_id": decision["candidate_id"],
                "rubric_identity": decision["rubric_identity"],
                "plan_hash": decision["plan_hash"],
                "decided_at": decision["decided_at"],
                "observations": observations,
                "evidence": evidence,
            }
        },
        timeout=20,
    )
    if not isinstance(result, dict) or result.get("verdict") != "QUALIFIED_FOR_PAPER":
        raise FoundryError("paper_qualification", "Source qualification failed.", 409)
    identity = journal.artifact({"submitted": dossier, "verified": result})
    if journal.get("qualification_identity") != identity:
        journal.set("qualification_identity", identity, kind="qualification_verified")
    return result
