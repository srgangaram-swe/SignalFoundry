"""Local research and explicit paper operations; no live-order capability."""

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
from signal_foundry.trading.service import Action, PaperService
from signal_foundry.trading.session import run_session
from signal_foundry.trading.store import Journal


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--state", type=Path, default=Path("var/research"))
    parser.add_argument(
        "--paper-config",
        type=Path,
        help="Explicit local paper configuration; never credentials",
    )
    parser.add_argument("--paper-state", type=Path, default=Path("var/paper"))
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
    paper = commands.add_parser(
        "paper", help="Explicit bounded Alpaca paper/data operations"
    )
    paper.add_argument(
        "operation",
        choices=[
            "status",
            "initialize",
            "probe",
            "acquire",
            "research",
            "qualify",
            "start",
            "cycle",
            "reconcile",
            "record-session",
            "campaign",
            "stop",
            "cancel",
            "show-artifact",
            "audit",
            "import-evidence",
            "run",
        ],
    )
    paper.add_argument("--symbol")
    paper.add_argument("--artifact")
    paper.add_argument("--file", type=Path)
    paper.add_argument("--cycles", type=int, default=1)
    paper.add_argument("--after", type=int, default=0)
    paper.add_argument("--limit", type=int, default=100)
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

        def paper_factory() -> PaperService:
            if args.paper_config is None:
                raise FoundryError(
                    "paper_configuration",
                    "Supply --paper-config for explicit paper operations.",
                )
            return PaperService(args.root, args.paper_state, args.paper_config)

        if args.command == "paper":
            service = paper_factory()
            if args.operation == "status":
                print(encode(service.status().wire()).decode())
            elif args.operation == "run":
                print(encode(run_session(service, args.cycles).wire()).decode())
            elif args.operation in {"show-artifact", "import-evidence", "audit"}:
                journal = Journal(args.paper_state)
                try:
                    if args.operation == "audit":
                        print(encode(journal.audit(args.after, args.limit)).decode())
                    elif args.operation == "show-artifact":
                        print(
                            encode(journal.read_artifact(args.artifact or "")).decode()
                        )
                    else:
                        if args.file is None:
                            raise FoundryError(
                                "paper_evidence",
                                "Supply --file to import private evidence.",
                            )
                        value = decode(read_file(args.file), 4 << 20)
                        if (
                            not isinstance(value, dict)
                            or value.get("config_identity") != service.config.identity
                            or value.get("plan_identity")
                            != service.config.plan.identity
                        ):
                            raise FoundryError(
                                "paper_evidence",
                                "Evidence must bind the frozen plan and configuration.",
                            )
                        with journal.exclusive():
                            identity = journal.artifact(value)
                            journal.append("evidence_import", {"identity": identity})
                        print(
                            encode(
                                {"artifact": identity, "qualification_implied": False}
                            ).decode()
                        )
                finally:
                    journal.close()
            else:
                print(
                    encode(
                        service.action(
                            Action(operation=args.operation, symbol=args.symbol)
                        ).wire()
                    ).decode()
                )
            return 0
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
                create_app(
                    factory,
                    port=args.port,
                    nexus=bundle,
                    paper_factory=paper_factory if args.paper_config else None,
                ),
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
