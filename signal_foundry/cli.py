"""Local-only entry point; no credentials, broker actions or automatic downloads."""

from __future__ import annotations

import argparse
import sys
import uuid
from pathlib import Path

from pydantic import ValidationError

from signal_foundry.api import create_app
from signal_foundry.boundary import FoundryError, decode, encode, read_file
from signal_foundry.contracts import JobState, Problem, ResearchRequest
from signal_foundry.manager import Manager
from signal_foundry.nexus import load_bundle
from signal_foundry.runner import Runner
from signal_foundry.store import Store


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--state", type=Path, default=Path("var/research"))
    parser.add_argument(
        "--bundles",
        type=Path,
        help="Approved local bundle parent; never a provider URL",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    serve = commands.add_parser("serve", help="Serve only http://127.0.0.1:8765")
    serve.add_argument("--port", type=int, default=8765)
    serve.add_argument(
        "--nexus", action="store_true", help="Mount the verified local Nexus build"
    )
    commands.add_parser("catalog")
    commands.add_parser("example", help="Print the complete default synthetic request")
    for name in ("validate", "run"):
        command = commands.add_parser(name)
        command.add_argument("request", type=Path)
        if name == "run":
            command.add_argument("--key", default=None)
    args = parser.parse_args(argv)

    def factory() -> Manager:
        runner = Runner(args.root, args.state, args.bundles)
        return Manager(Store(args.state), runner)

    try:
        if args.command == "serve":
            import uvicorn

            bundle = (
                load_bundle(
                    args.root / "apps/nexus/dist",
                    contract=args.root / "contracts/openapi-v1.json",
                )
                if args.nexus
                else None
            )
            uvicorn.run(
                create_app(factory, port=args.port, nexus=bundle),
                host="127.0.0.1",
                port=args.port,
                workers=1,
                proxy_headers=False,
                access_log=False,
                limit_concurrency=32,
                timeout_keep_alive=5,
                h11_max_incomplete_event_size=16_384,
            )
            return 0
        if args.command == "example":
            print(ResearchRequest().canonical().decode())
            return 0
        runner = Runner(args.root, args.state, args.bundles)
        if args.command == "catalog":
            print(runner.catalog().canonical().decode())
            return 0
        payload = read_file(args.request, 16_384)
        decode(payload)
        request = ResearchRequest.model_validate_json(payload)
        if args.command == "validate":
            print(runner.validate(request).canonical().decode())
            return 0
        manager = Manager(Store(args.state), runner)
        try:
            job = manager.submit(request, args.key or uuid.uuid4().hex)
            completed = manager.wait(job.job_id)
            print(encode({"job": completed.model_dump(mode="json")}).decode())
            return 0 if completed.state == JobState.SUCCEEDED else 1
        finally:
            manager.close()
    except (FoundryError, ValidationError) as exc:
        problem = (
            Problem(code=exc.code, detail=exc.detail)
            if isinstance(exc, FoundryError)
            else Problem(
                code="invalid_request", detail="Request violates the research schema."
            )
        )
        print(problem.canonical().decode(), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
