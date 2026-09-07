"""Opt-in GitHub metadata freeze; offline inventory never calls this adapter."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from scripts.preservation.git import PreservationError
from scripts.preservation.inventory import canonical, sha256


def capture(repository: str, request: Callable[[str], Any]) -> dict[str, Any]:
    """Allowlist traceability metadata across bounded REST pages, without bodies.

    The authenticated transport is injected. Each response must itself have a
    bounded byte count/time. Head/tag refs are checked twice to reject races;
    discussions are a dated metadata snapshot, not an atomic GitHub transaction.
    URLs remain source identities, never assumed to move with imported code.
    """
    if repository not in {"srgangaram-swe/AlphaForge", "srgangaram-swe/Signalattice"}:
        raise PreservationError("unknown-github-repository")
    base = f"repos/{repository}"

    def pages(endpoint: str) -> list[dict[str, Any]]:
        result = []
        separator = "&" if "?" in endpoint else "?"
        for page in range(1, 21):
            records = request(f"{base}/{endpoint}{separator}per_page=100&page={page}")
            if not isinstance(records, list) or len(records) > 100:
                raise PreservationError("invalid-github-page")
            if any(not isinstance(row, dict) for row in records):
                raise PreservationError("invalid-github-record")
            result.extend(records)
            if len(records) < 100:
                return result
        raise PreservationError("github-page-limit")

    def refs() -> list[dict[str, str]]:
        return sorted(
            [
                {"ref": f"refs/heads/{row['name']}", "oid": row["commit"]["sha"]}
                for row in pages("branches")
            ]
            + [
                {"ref": f"refs/tags/{row['name']}", "oid": row["commit"]["sha"]}
                for row in pages("tags")
            ],
            key=lambda row: row["ref"],
        )

    try:
        initial_refs = refs()
        repo = request(base)
        discussions = []
        for row in pages("issues?state=all"):
            kind = "pull" if "pull_request" in row else "issue"
            discussions.append(
                {
                    "key": f"{repository}:{kind}:{row['number']}",
                    "source_url": row["html_url"],
                    "kind": kind,
                    "number": row["number"],
                    "state": row["state"],
                    "target": "source-link-retained; target tracking issue mapped at cutover",
                }
            )
        milestones = [
            {
                key: row[key]
                for key in (
                    "number",
                    "title",
                    "state",
                    "html_url",
                    "due_on",
                    "open_issues",
                    "closed_issues",
                )
            }
            for row in pages("milestones?state=all")
        ]
        releases = [
            {
                key: row[key]
                for key in ("id", "tag_name", "target_commitish", "html_url", "draft", "prerelease")
            }
            for row in pages("releases")
        ]
        protections = {}
        for branch in ("dev", "prod", "main"):
            protection = request(f"{base}/branches/{branch}/protection")
            protections[branch] = {
                key: protection.get(key)
                for key in (
                    "required_status_checks",
                    "required_pull_request_reviews",
                    "enforce_admins",
                    "allow_force_pushes",
                    "allow_deletions",
                    "required_linear_history",
                )
            }
        if refs() != initial_refs:
            raise PreservationError("github-ref-race")
        result = {
            "schema_version": 1,
            "repository": repository,
            "default_branch": repo["default_branch"],
            "license_spdx": repo["license"]["spdx_id"],
            "source_url": repo["html_url"],
            "refs": initial_refs,
            "discussions": sorted(discussions, key=lambda row: row["key"]),
            "milestones": sorted(milestones, key=lambda row: row["number"]),
            "releases": sorted(releases, key=lambda row: row["id"]),
            "protections": protections,
        }
        # Only JSON-safe values accepted; no arbitrary API object serialization.
        json.loads(canonical(result))
        return {**result, "snapshot_sha256": sha256(canonical(result))}
    except (KeyError, TypeError, ValueError) as exc:
        raise PreservationError("invalid-github-metadata") from exc
