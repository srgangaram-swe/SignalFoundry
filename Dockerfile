# syntax=docker/dockerfile:1.20.0@sha256:26147acbda4f14c5add9946e2fd2ed543fc402884fd75146bd342a7f6271dc1d
# check=error=true

# Immutable references were resolved from the official Docker Hub and GHCR
# repositories on 2026-08-09. Tags remain only as human-readable provenance;
# each build is identity-bound by its sha256 digest.
# https://hub.docker.com/r/docker/dockerfile
# https://hub.docker.com/_/python
# https://github.com/astral-sh/uv/pkgs/container/uv
ARG SOURCE_DATE_EPOCH=0
ARG VCS_REF=unknown

FROM ghcr.io/astral-sh/uv:0.11.32@sha256:df4cae8f3a96d175e2e5f992e597550000edbe78fdc2594d5cd8de1a217f504c AS uv-tool

FROM python:3.13-slim-bookworm@sha256:67a1e1f215ccda113cfc024e8639049257e88f273898f595b61476d128d387e8 AS builder-base

ARG SOURCE_DATE_EPOCH
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_COMPILE_BYTECODE=0 \
    UV_LINK_MODE=copy \
    UV_NO_CACHE=1 \
    UV_NO_PROGRESS=1 \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /build

COPY --from=uv-tool --chown=0:0 --chmod=0555 /uv /usr/local/bin/uv


FROM builder-base AS cli-builder

COPY pyproject.toml uv.lock README.md LICENSE ./
COPY src ./src

# uv verifies every selected wheel against uv.lock. The slim builder contains
# no compiler; a dependency without a compatible locked wheel therefore fails
# closed instead of compiling an unreviewed native artifact.
RUN uv sync --locked --no-dev --no-editable


FROM cli-builder AS test-runner

# The CI-only runner resolves the committed development extra from the same
# universal lock. It is never copied into either runtime image.
RUN uv sync --locked --no-dev --no-editable --extra dev

ENTRYPOINT ["/opt/venv/bin/python"]


FROM builder-base AS service-builder

COPY pyproject.toml uv.lock ./

# Install only the narrow, lock-resolved HTTP/runtime dependency group. The
# project itself is copied later as an exact source allowlist, so this builder
# cannot smuggle the research dependency graph or generated distribution data
# into the service image.
RUN uv sync --locked --only-group service-runtime --no-install-project


FROM python:3.13-slim-bookworm@sha256:67a1e1f215ccda113cfc024e8639049257e88f273898f595b61476d128d387e8 AS runtime-base

ARG VCS_REF
ENV HOME=/nonexistent \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PATH=/opt/venv/bin:/usr/local/bin:/usr/bin:/bin \
    PYTHONHASHSEED=0 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

LABEL org.opencontainers.image.title="Signalattice" \
      org.opencontainers.image.description="Local research-evidence service and CLI" \
      org.opencontainers.image.licenses="MIT" \
      org.opencontainers.image.revision="${VCS_REF}" \
      org.opencontainers.image.source="https://github.com/srgangaram-swe/Signalattice"

# The runtime identity has neither a login shell nor a created home. Runtime
# state is supplied through read-only mounts and bounded tmpfs volumes.
# groupadd and useradd are invoked by absolute path: the ENV PATH above
# deliberately excludes /usr/sbin so the runtime identity cannot reach sbin
# tooling, and widening it to make this layer build would give that back.
RUN --network=none /usr/sbin/groupadd --gid 10001 signalattice \
    && /usr/sbin/useradd --uid 10001 --gid 10001 --no-create-home \
        --home-dir /nonexistent --shell /usr/sbin/nologin signalattice \
    && install -d -o 10001 -g 10001 -m 0700 /run/signalattice \
    && install -d -o 0 -g 10001 -m 0550 \
        /var/lib/signalattice/registry /var/lib/signalattice/cas \
    && rm -rf /usr/local/lib/python3.13/site-packages \
        /usr/local/lib/python3.13/ensurepip \
        /usr/local/bin/pip /usr/local/bin/pip3 /usr/local/bin/pip3.13 \
    && find / -xdev -type f -perm /6000 -exec chmod a-s {} + \
    && rm -rf /root/.cache /root/.local /tmp/* /var/tmp/*

WORKDIR /app


FROM runtime-base AS service

ENV SIGNALATTICE_REGISTRY_DB=/var/lib/signalattice/registry/registry.sqlite3 \
    SIGNALATTICE_CAS_ROOT=/var/lib/signalattice/cas \
    SIGNALATTICE_API_SOCKET=/run/signalattice/api.sock \
    PYTHONPATH=/app/src

COPY --from=service-builder --chown=0:0 /opt/venv /opt/venv
COPY --chown=0:0 --chmod=0444 \
    src/quant_platform/__init__.py \
    src/quant_platform/py.typed \
    /app/src/quant_platform/
COPY --chown=0:0 --chmod=0444 \
    src/quant_platform/service/__init__.py \
    src/quant_platform/service/__main__.py \
    src/quant_platform/service/admission.py \
    src/quant_platform/service/api.py \
    src/quant_platform/service/contracts.py \
    src/quant_platform/service/entrypoint.py \
    src/quant_platform/service/exporter.py \
    src/quant_platform/service/http_protocol.py \
    src/quant_platform/service/manifests.py \
    src/quant_platform/service/metrics.py \
    src/quant_platform/service/middleware.py \
    src/quant_platform/service/models.py \
    src/quant_platform/service/problems.py \
    src/quant_platform/service/server.py \
    src/quant_platform/service/telemetry.py \
    src/quant_platform/service/telemetry_contracts.py \
    /app/src/quant_platform/service/
COPY --chown=0:0 --chmod=0444 \
    src/quant_platform/tracking/__init__.py \
    src/quant_platform/tracking/cas.py \
    src/quant_platform/tracking/contracts.py \
    src/quant_platform/tracking/migrations.py \
    src/quant_platform/tracking/read_ports.py \
    src/quant_platform/tracking/registry.py \
    src/quant_platform/tracking/retention.py \
    /app/src/quant_platform/tracking/

# The venv is copied wholesale from the builder, so it carries whatever modes
# uv produced there. The application sources are pinned to 0444 at COPY time,
# but a directory tree cannot be, so strip group and other write bits here.
# Execute bits are preserved, which the interpreter and console scripts need.
# The container verifier rejects any file under /opt/venv or /app/src whose
# mode intersects 0o022; a writable path inside a read-only runtime is a
# tampering surface, not a convenience.
# PEP 770 vendored SBOMs under third-party *.dist-info/sboms/ record the
# upstream maintainer's own build machine -- ruff's ships /Users/runner/ and
# pydantic-core's ships /home/runner/work/. They have no runtime purpose, and
# the container verifier's property is that no build-host path appears in the
# image. Removing them makes that property true; adding an exception to the
# check would leave the paths in place and merely stop looking for them.
#
# Tradeoff: syft catalogs distributions from METADATA and RECORD, which remain,
# but any vendored component detail these files carried is not in our SBOM.
# The runtime-base stage strips setuid/setgid across the base filesystem, but
# that runs before this stage copies /opt/venv from the builder, so the venv
# never passed through it. Strip again here, after every COPY.
RUN --network=none find /opt/venv -type d -path "*.dist-info/sboms" -exec rm -rf {} + \
    && find /opt/venv /app/src -type f -perm /6000 -exec chmod a-s {} + \
    && chmod -R go-w /opt/venv /app/src

USER 10001:10001

# This probe crosses the real Unix socket and readiness route without adding a
# shell, network client, or TCP listener to the runtime contract.
HEALTHCHECK --interval=10s --timeout=2s --start-period=5s --retries=3 \
    CMD ["python", "-c", "import socket; s=socket.socket(socket.AF_UNIX); s.settimeout(1); s.connect('/run/signalattice/api.sock'); s.sendall(b'GET /health/ready HTTP/1.1\\r\\nHost: localhost\\r\\nConnection: close\\r\\n\\r\\n'); f=s.makefile('rb'); line=f.readline(4097); f.close(); s.close(); parts=line.split(b' ',2); raise SystemExit(0 if len(line)<=4096 and line.endswith(b'\\r\\n') and len(parts)>=2 and parts[0] in (b'HTTP/1.0',b'HTTP/1.1') and parts[1]==b'200' else 1)"]

ENTRYPOINT ["python", "-m", "quant_platform.service"]
CMD ["--registry-db", "/var/lib/signalattice/registry/registry.sqlite3", "--cas-root", "/var/lib/signalattice/cas", "--socket-path", "/run/signalattice/api.sock", "--digest-key-file", "/run/secrets/registry-digest-key"]


# Keep the historic default image target as the full research CLI. The
# dedicated service target above deliberately excludes configs, scripts, data,
# reports, models, tests, documentation, and VCS metadata.
FROM runtime-base AS cli

ENV SIGNALATTICE_DATA_DIR=/app/data \
    MPLCONFIGDIR=/tmp/matplotlib \
    XDG_CACHE_HOME=/tmp/.cache

COPY --from=cli-builder --chown=0:0 /opt/venv /opt/venv
COPY --chown=10001:10001 configs ./configs
COPY --chown=10001:10001 scripts ./scripts
COPY --chown=10001:10001 data ./data

RUN --network=none install -d -o 10001 -g 10001 -m 0750 \
        /app/reports/figures /app/experiments /app/models

USER 10001:10001

ENTRYPOINT ["signalattice"]
CMD ["--help"]
