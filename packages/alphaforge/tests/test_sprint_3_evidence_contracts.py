"""Semantic-governance invariants for the Sprint 3 decision record (SF-S3-MR11).

The synthesis record's whole purpose is to stop a favourable conclusion being
asserted without substantiation. That protection is a chain of invariants, and
these tests pin each link:

* a boolean evidence gate is true **exactly** when an independent semantic
  review is attached — so a gate cannot be flipped on by editing one field;
* a review must carry the specific fact *roles* its gate requires, so
  "out of sample" cannot be claimed without a temporal contract and a result;
* every reviewed fact must point at a source the family actually declared;
* an unevaluated family must be deferred, and only an evaluated family with a
  complete gate set may advance.

Together these make "advance" unreachable by omission: the record either holds
the substantiating facts or it fails to construct.
"""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from typing import Any

import pytest

from alphaforge.research.sprint_3_decision import (
    EVIDENCE_GATES,
    EvidenceFact,
    EvidenceGateSupport,
    FamilyEvidence,
    SourceArtifact,
    Sprint3DecisionError,
)

DIGEST = "0" * 64
SOURCE_PATH = "docs/evidence/study.json"

#: Roles required by the most demanding gates, so one fact set satisfies any.
ALL_ROLES = (
    "temporal_contract",
    "uncertainty_method",
    "cost_policy",
    "result",
    "selection_control",
    "experimental_control",
    "resource_boundary",
    "limitation",
)


def _fact(**overrides: object) -> EvidenceFact:
    fields: dict[str, object] = {
        "roles": ALL_ROLES,
        "source_path": SOURCE_PATH,
        "locator_kind": "json_pointer",
        "locator": "/results/oos",
        "observed": "IC 0.021",
    }
    fields.update(overrides)
    return EvidenceFact(**fields)  # type: ignore[arg-type]


def _support(gate: str = "out_of_sample", **overrides: Any) -> EvidenceGateSupport:
    fields: dict[str, object] = {
        "gate": gate,
        "verdict": "supported",
        "review_method": "independent_manual_source_semantics",
        "assertion": f"{gate} is substantiated by the linked source facts.",
        "facts": (_fact(),),
        "residual_limitations": ("synthetic panel only",),
    }
    fields.update(overrides)
    return EvidenceGateSupport(**fields)  # type: ignore[arg-type]


def _family(**overrides: object) -> FamilyEvidence:
    fields: dict[str, object] = {
        "family": "spectral_descriptors",
        "context": "synthetic_engineering",
        "evaluated": True,
        "disposition": "defer",
        "reason": "Deferred pending frozen baselines.",
        "sources": (SourceArtifact(path=SOURCE_PATH, sha256=DIGEST),),
        "gate_support": (),
        **{gate: False for gate in EVIDENCE_GATES},
    }
    fields.update(overrides)
    return FamilyEvidence(**fields)  # type: ignore[arg-type]


def test_a_coherent_family_record_is_accepted() -> None:
    family = _family()
    assert family.disposition == "defer"
    assert not any(getattr(family, gate) for gate in EVIDENCE_GATES)


# ---------------------------------------------------------------------------
# EvidenceFact
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"roles": ()}, "non-empty and bounded"),
        ({"roles": ("result", "result")}, "unique"),
        ({"roles": ("not_a_role",)}, "unsupported semantic role"),
        ({"locator_kind": "xpath"}, "unsupported evidence locator kind"),
        ({"locator": ""}, "at most 512 characters"),
        ({"locator": "x" * 513}, "at most 512 characters"),
        ({"observed": "   "}, "observed-value summary"),
        ({"observed": "x" * 1001}, "observed-value summary"),
        ({"locator": "results/oos"}, "absolute JSON pointer"),
        (
            {"locator_kind": "markdown_heading", "locator": "Results"},
            "exact heading",
        ),
    ],
)
def test_evidence_fact_validation(overrides: dict[str, object], message: str) -> None:
    with pytest.raises(Sprint3DecisionError, match=message):
        _fact(**overrides)


@pytest.mark.parametrize(
    ("kind", "locator"),
    [
        ("json_pointer", "/results/oos"),
        ("csv_column", "information_coefficient"),
        ("markdown_heading", "## Results"),
    ],
)
def test_every_locator_kind_is_constructible(kind: str, locator: str) -> None:
    assert _fact(locator_kind=kind, locator=locator).locator_kind == kind


# ---------------------------------------------------------------------------
# EvidenceGateSupport
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"gate": "looks_good"}, "unsupported evidence gate"),
        ({"verdict": "rejected"}, "supported review verdict"),
        ({"review_method": "automated_keyword_scan"}, "independent semantic review"),
        ({"assertion": "  "}, "bounded explicit assertion"),
        ({"assertion": "x" * 2001}, "bounded explicit assertion"),
        ({"facts": ()}, "fact count"),
        ({"facts": (_fact(), _fact())}, "facts must be unique"),
        ({"residual_limitations": ()}, "residual limitations"),
        ({"residual_limitations": ("  ",)}, "non-empty and bounded"),
        ({"residual_limitations": ("x" * 1001,)}, "non-empty and bounded"),
    ],
)
def test_gate_support_validation(overrides: dict[str, Any], message: str) -> None:
    with pytest.raises(Sprint3DecisionError, match=message):
        _support(**overrides)


@pytest.mark.parametrize("gate", EVIDENCE_GATES)
def test_every_gate_demands_its_required_fact_roles(gate: str) -> None:
    """A gate cannot be substantiated by facts that do not address it."""
    with pytest.raises(Sprint3DecisionError, match="lacks required roles"):
        _support(gate, facts=(_fact(roles=("limitation",)),))


@pytest.mark.parametrize("gate", EVIDENCE_GATES)
def test_every_gate_is_satisfiable_with_complete_roles(gate: str) -> None:
    assert _support(gate).gate == gate


# ---------------------------------------------------------------------------
# FamilyEvidence — the invariant chain
# ---------------------------------------------------------------------------


def test_a_gate_is_true_exactly_when_a_review_substantiates_it() -> None:
    """The core protection: a boolean cannot be flipped on by itself."""
    with pytest.raises(Sprint3DecisionError, match="true exactly when"):
        _family(out_of_sample=True)
    with pytest.raises(Sprint3DecisionError, match="true exactly when"):
        _family(gate_support=(_support("out_of_sample"),), out_of_sample=False)
    # Declared together, the record is coherent.
    family = _family(gate_support=(_support("out_of_sample"),), out_of_sample=True)
    assert family.out_of_sample is True


def test_reviewed_facts_must_reference_declared_family_sources() -> None:
    stray = _support("out_of_sample", facts=(_fact(source_path="docs/other.json"),))
    with pytest.raises(Sprint3DecisionError, match="declared family sources"):
        _family(gate_support=(stray,), out_of_sample=True)


def test_duplicate_gate_reviews_are_refused() -> None:
    with pytest.raises(Sprint3DecisionError, match="gate support records must be unique"):
        _family(
            gate_support=(_support("out_of_sample"), _support("out_of_sample")),
            out_of_sample=True,
        )


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"context": "vibes"}, "unsupported evidence context"),
        ({"disposition": "maybe"}, "unsupported disposition"),
        ({"reason": "  "}, "family reason"),
        ({"reason": "x" * 2001}, "family reason"),
        ({"sources": ()}, "at least one aggregate source"),
        ({"context": "unsupported", "evaluated": True}, "cannot be marked evaluated"),
        ({"evaluated": False, "disposition": "reject"}, "must be deferred"),
    ],
)
def test_family_validation(overrides: dict[str, object], message: str) -> None:
    with pytest.raises(Sprint3DecisionError, match=message):
        _family(**overrides)


def test_advancing_requires_every_gate_substantiated() -> None:
    """ "Advance" is unreachable while any governed gate is missing."""
    with pytest.raises(Sprint3DecisionError, match="every governed evidence gate"):
        _family(
            disposition="advance", gate_support=(_support("out_of_sample"),), out_of_sample=True
        )

    complete = _family(
        disposition="advance",
        gate_support=tuple(_support(gate) for gate in EVIDENCE_GATES),
        **{gate: True for gate in EVIDENCE_GATES},
    )
    assert complete.disposition == "advance"
    assert not complete.missing_gates


def test_an_unevaluated_family_cannot_advance() -> None:
    with pytest.raises(Sprint3DecisionError, match="deferred|may advance"):
        _family(evaluated=False, disposition="advance")


@pytest.mark.parametrize("count", [-1, True, 100_001])
def test_model_counts_are_bounded_non_negative_ints(count: object) -> None:
    with pytest.raises(Sprint3DecisionError):
        _family(model_count=count)


def test_missing_gates_lists_every_unsubstantiated_gate() -> None:
    family = _family(gate_support=(_support("out_of_sample"),), out_of_sample=True)
    assert set(family.missing_gates) == set(EVIDENCE_GATES) - {"out_of_sample"}


# ---------------------------------------------------------------------------
# EvidenceFact.verify — locator resolution against real sources
# ---------------------------------------------------------------------------


def _artifact(tmp_path: Path, name: str, payload: bytes) -> SourceArtifact:
    target = tmp_path / name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(payload)
    return SourceArtifact(path=name, sha256=sha256(payload).hexdigest())


def test_json_pointer_resolves_objects_and_arrays(tmp_path: Path) -> None:
    artifact = _artifact(tmp_path, "study.json", b'{"results": [{"ic": 0.02}]}')
    fact = _fact(source_path="study.json", locator="/results/0/ic")
    fact.verify(tmp_path, (artifact,))


def test_json_pointer_escapes_are_honoured(tmp_path: Path) -> None:
    """RFC 6901: ~1 is '/', ~0 is '~'. Getting this wrong silently mislocates."""
    artifact = _artifact(tmp_path, "study.json", b'{"a/b": {"c~d": 1}}')
    _fact(source_path="study.json", locator="/a~1b/c~0d").verify(tmp_path, (artifact,))


@pytest.mark.parametrize("locator", ["/missing", "/results/9", "/results/01", "/results/x"])
def test_unresolvable_json_pointers_are_refused(tmp_path: Path, locator: str) -> None:
    artifact = _artifact(tmp_path, "study.json", b'{"results": [{"ic": 0.02}]}')
    with pytest.raises(Sprint3DecisionError, match="does not resolve"):
        _fact(source_path="study.json", locator=locator).verify(tmp_path, (artifact,))


@pytest.mark.parametrize("locator", ["/a~", "/a~2b"])
def test_invalid_json_pointer_escapes_are_refused(tmp_path: Path, locator: str) -> None:
    artifact = _artifact(tmp_path, "study.json", b'{"a": 1}')
    with pytest.raises(Sprint3DecisionError, match="invalid escape"):
        _fact(source_path="study.json", locator=locator).verify(tmp_path, (artifact,))


def test_non_json_source_for_a_json_pointer_is_refused(tmp_path: Path) -> None:
    artifact = _artifact(tmp_path, "study.json", b"{not json")
    with pytest.raises(Sprint3DecisionError, match="not strict bounded JSON"):
        _fact(source_path="study.json").verify(tmp_path, (artifact,))


def test_csv_column_resolution(tmp_path: Path) -> None:
    artifact = _artifact(tmp_path, "metrics.csv", b"model,ic\nridge,0.02\n")
    fact = _fact(source_path="metrics.csv", locator_kind="csv_column", locator="ic")
    fact.verify(tmp_path, (artifact,))

    missing = _fact(source_path="metrics.csv", locator_kind="csv_column", locator="sharpe")
    with pytest.raises(Sprint3DecisionError, match="CSV column does not exist"):
        missing.verify(tmp_path, (artifact,))


def test_duplicate_csv_columns_are_refused(tmp_path: Path) -> None:
    """A duplicated header makes 'the ic column' ambiguous."""
    artifact = _artifact(tmp_path, "metrics.csv", b"ic,ic\n1,2\n")
    fact = _fact(source_path="metrics.csv", locator_kind="csv_column", locator="ic")
    with pytest.raises(Sprint3DecisionError, match="duplicate columns"):
        fact.verify(tmp_path, (artifact,))


def test_markdown_heading_must_resolve_exactly_once(tmp_path: Path) -> None:
    artifact = _artifact(tmp_path, "report.md", b"# Title\n\n## Results\n\ntext\n")
    fact = _fact(source_path="report.md", locator_kind="markdown_heading", locator="## Results")
    fact.verify(tmp_path, (artifact,))

    duplicated = _artifact(tmp_path, "dup.md", b"## Results\n\n## Results\n")
    ambiguous = _fact(source_path="dup.md", locator_kind="markdown_heading", locator="## Results")
    with pytest.raises(Sprint3DecisionError, match="exactly once"):
        ambiguous.verify(tmp_path, (duplicated,))

    absent = _fact(source_path="report.md", locator_kind="markdown_heading", locator="## Absent")
    with pytest.raises(Sprint3DecisionError, match="exactly once"):
        absent.verify(tmp_path, (artifact,))


def test_fact_must_reference_a_declared_source(tmp_path: Path) -> None:
    artifact = _artifact(tmp_path, "study.json", b'{"a": 1}')
    with pytest.raises(Sprint3DecisionError, match="undeclared source"):
        _fact(source_path="other.json", locator="/a").verify(tmp_path, (artifact,))


def test_unverified_source_cache_is_refused(tmp_path: Path) -> None:
    """A fact may only be read from a snapshot that already passed hashing."""
    artifact = _artifact(tmp_path, "study.json", b'{"a": 1}')
    fact = _fact(source_path="study.json", locator="/a")
    with pytest.raises(Sprint3DecisionError, match="content verification"):
        fact.verify(tmp_path, (artifact,), verified_sources={})


def test_source_artifact_rejects_a_content_mismatch(tmp_path: Path) -> None:
    (tmp_path / "study.json").write_bytes(b'{"a": 1}')
    tampered = SourceArtifact(path="study.json", sha256="0" * 64)
    with pytest.raises(Sprint3DecisionError):
        tampered.read_verified(tmp_path)
