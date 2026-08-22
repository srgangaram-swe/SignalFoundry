"""The publication gate.

SF-S5-SL-MR7. Every function here refuses. There is deliberately no function
that publishes: this module decides whether publication is *permitted*, and the
act of publishing is a separately authorized operation elsewhere.

The refusals exist because each corresponds to a way a release goes wrong in a
manner that is hard to detect afterwards:

* publishing from ``dev`` or ``prod`` produces a tag whose ancestry nobody can
  reconstruct;
* publishing from a dirty tree produces artifacts that no commit reproduces;
* reusing or moving a tag makes an earlier signature apply to different bytes,
  which silently invalidates every verification anyone already performed;
* publishing a dry-run build lets an unreviewed artifact acquire release
  identity.

None of these is recoverable by a later fix, which is why they are all
fail-closed at the gate rather than warnings.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Final

from quant_platform.release.descriptor import ReleaseDescriptor
from quant_platform.release.identity import parse_version

#: The only branch a release may be published from.
PUBLICATION_BRANCH: Final = "main"

#: Tag shape. A release tag is derived from the version, never chosen freely.
_TAG_PATTERN: Final = re.compile(r"^v\d+\.\d+\.\d+(?:(?:a|b|rc)\d+)?$")


class PublicationRefused(RuntimeError):
    """Raised when publication is not permitted.

    A distinct type from validation errors so generic retry handling cannot loop
    past it: a refused publication is refused until a human changes something.
    """


@dataclass(frozen=True, slots=True)
class RepositoryState:
    """The observed state of the repository at the moment of the request."""

    branch: str
    head_commit: str
    remote_main_commit: str
    is_clean: bool
    existing_tags: frozenset[str]
    #: Commits reachable from main; used to prove the promotion ancestry.
    main_ancestry: frozenset[str]


def tag_for_version(version: str) -> str:
    """Return the immutable tag name for a release version."""
    return f"v{parse_version(version)}"


def assert_tag_shape(tag: str) -> str:
    """Refuse a tag that is not derived from a version.

    Raises:
        PublicationRefused: On a free-form tag.
    """
    if _TAG_PATTERN.match(tag) is None:
        raise PublicationRefused(
            f"tag {tag!r} is not of the form v<major>.<minor>.<patch>; a release tag is "
            "derived from the version so the two can never disagree"
        )
    return tag


def assert_publication_permitted(
    descriptor: ReleaseDescriptor,
    state: RepositoryState,
    *,
    requested_commit: str,
    promotion_ancestry: tuple[str, ...] = (),
) -> str:
    """Return the tag to create, or refuse with the reason.

    Args:
        descriptor: The release being published.
        state: Observed repository state.
        requested_commit: The commit the operator asked to publish.
        promotion_ancestry: Commits that must be reachable from main, proving
            the dev -> prod -> main path actually happened.

    Returns:
        The immutable tag name to create.

    Raises:
        PublicationRefused: Naming the single condition that failed.
    """
    if descriptor.build_kind != "publication":
        raise PublicationRefused(
            "this descriptor describes a dry-run build; a dry-run artifact must never "
            "acquire release identity"
        )
    if state.branch != PUBLICATION_BRANCH:
        raise PublicationRefused(
            f"publication is only permitted from {PUBLICATION_BRANCH}, not {state.branch!r}; "
            "a tag cut elsewhere has an ancestry nobody can reconstruct"
        )
    if not state.is_clean:
        raise PublicationRefused(
            "the working tree is dirty; artifacts built from uncommitted changes are not "
            "reproducible from any commit"
        )
    if requested_commit != state.head_commit:
        raise PublicationRefused(
            f"requested commit {requested_commit[:12]} is not HEAD ({state.head_commit[:12]})"
        )
    if requested_commit != state.remote_main_commit:
        raise PublicationRefused(
            f"requested commit {requested_commit[:12]} is not the current origin/main "
            f"({state.remote_main_commit[:12]}); publishing a stale commit would tag bytes "
            "that main no longer describes"
        )
    if descriptor.source_commit != requested_commit:
        raise PublicationRefused(
            f"the descriptor was built from {descriptor.source_commit[:12]} but publication "
            f"targets {requested_commit[:12]}"
        )

    missing = [commit for commit in promotion_ancestry if commit not in state.main_ancestry]
    if missing:
        raise PublicationRefused(
            "the recorded dev -> prod -> main promotion is not reachable from main: "
            f"{', '.join(item[:12] for item in missing)}"
        )

    tag = assert_tag_shape(tag_for_version(descriptor.release_version))
    if tag in state.existing_tags:
        raise PublicationRefused(
            f"tag {tag} already exists; published tags are immutable, and a defective "
            "release is corrected with a new version rather than a moved tag"
        )
    return tag


def assert_dry_run_publishes_nothing(
    *, created_tags: tuple[str, ...], created_releases: tuple[str, ...]
) -> None:
    """Refuse if a dry run produced any public artifact.

    Raises:
        PublicationRefused: If a tag or release exists after a dry run.
    """
    if created_tags or created_releases:
        raise PublicationRefused(
            "a dry run created public state: "
            f"tags={list(created_tags)} releases={list(created_releases)}. "
            "The dry run exists to prove the process without publishing anything."
        )


__all__ = [
    "PUBLICATION_BRANCH",
    "PublicationRefused",
    "RepositoryState",
    "assert_dry_run_publishes_nothing",
    "assert_publication_permitted",
    "assert_tag_shape",
    "tag_for_version",
]
