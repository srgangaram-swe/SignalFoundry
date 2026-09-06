"""Adversarial tests for the release-archive publication boundary."""

from __future__ import annotations

import base64
import csv
import gzip
import hashlib
import io
import stat
import struct
import tarfile
import warnings
import zipfile
import zlib
from pathlib import Path

import pytest
from scripts import verify_distributions as verifier

_DIST_INFO = "signalattice-0.2.1.dist-info"


def _record_hash(payload: bytes) -> str:
    encoded = base64.urlsafe_b64encode(hashlib.sha256(payload).digest()).rstrip(b"=")
    return "sha256=" + encoded.decode("ascii")


def _gzip_member(
    payload: bytes,
    *,
    filename: str,
    comment: bytes | None = None,
    extra: bytes | None = None,
) -> bytes:
    """Build one deterministic RFC 1952 member for gzip-header adversaries."""

    filename_bytes = filename.encode("ascii")
    flags = 0x08
    optional = bytearray(filename_bytes + b"\0")
    if comment is not None:
        flags |= 0x10
        optional.extend(comment + b"\0")
    if extra is not None:
        flags |= 0x04
        optional = bytearray(struct.pack("<H", len(extra)) + extra) + optional
    compressor = zlib.compressobj(level=9, wbits=-zlib.MAX_WBITS)
    compressed = compressor.compress(payload) + compressor.flush()
    header = b"\x1f\x8b\x08" + bytes((flags,)) + struct.pack("<I", 0) + b"\x02\xff"
    trailer = struct.pack("<II", zlib.crc32(payload), len(payload) & 0xFFFFFFFF)
    return header + bytes(optional) + compressed + trailer


def _service_metadata(
    *,
    provides_service: bool = True,
    service_dependencies: tuple[str, ...] = ("fastapi", "starlette", "uvicorn"),
    unconditional_dependencies: tuple[str, ...] = (),
) -> bytes:
    lines = [
        "Metadata-Version: 2.4",
        "Name: signalattice",
        "Version: 0.2.1",
        "Provides-Extra: dev",
    ]
    if provides_service:
        lines.append("Provides-Extra: service")
    lines.extend(f'Requires-Dist: {name}>=1; extra == "service"' for name in service_dependencies)
    lines.extend(f"Requires-Dist: {name}>=1" for name in unconditional_dependencies)
    return ("\n".join(lines) + "\n").encode("ascii")


def _wheel_files(*, metadata: bytes | None = None) -> dict[str, bytes]:
    files = {
        name: (b"" if name.endswith("py.typed") else b"# bounded package fixture\n")
        for name in verifier._REQUIRED_WHEEL_FILES
    }
    files[f"{_DIST_INFO}/METADATA"] = _service_metadata() if metadata is None else metadata
    files[f"{_DIST_INFO}/WHEEL"] = b"Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n"
    files[f"{_DIST_INFO}/entry_points.txt"] = (
        b"[console_scripts]\nsignalattice = quant_platform.cli:main\n"
    )
    files[f"{_DIST_INFO}/licenses/LICENSE"] = b"bounded license fixture\n"
    files[f"{_DIST_INFO}/licenses/DISCLAIMER.md"] = b"bounded disclaimer fixture\n"
    files[f"{_DIST_INFO}/top_level.txt"] = b"quant_platform\n"
    return files


def _write_wheel(
    path: Path,
    *,
    metadata: bytes | None = None,
    record_defect: str | None = None,
    recorded_extra_entries: tuple[tuple[str, bytes], ...] = (),
    extra_entries: tuple[tuple[str, bytes], ...] = (),
    explicit_directories: tuple[str, ...] = (),
) -> None:
    files = _wheel_files(metadata=metadata)
    for name, payload in recorded_extra_entries:
        if name in files:
            raise AssertionError("recorded wheel fixture entries must be unique")
        files[name] = payload
    rows = [
        [name, _record_hash(payload), str(len(payload))] for name, payload in sorted(files.items())
    ]
    if record_defect == "hash":
        rows[0][1] = _record_hash(b"different bytes")
    elif record_defect == "size":
        rows[0][2] = str(int(rows[0][2]) + 1)
    elif record_defect == "missing":
        rows.pop(0)
    elif record_defect == "huge_size":
        rows[0][2] = "9" * 10_000
    elif record_defect not in {None, "hash", "size", "missing", "huge_size"}:
        raise AssertionError("unsupported test defect")
    record_name = f"{_DIST_INFO}/RECORD"
    rows.append([record_name, "", ""])
    buffer = io.StringIO(newline="")
    csv.writer(buffer, lineterminator="\n").writerows(rows)
    files[record_name] = buffer.getvalue().encode("utf-8")

    with zipfile.ZipFile(path, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, payload in sorted(files.items()):
            archive.writestr(name, payload)
        for name, payload in extra_entries:
            archive.writestr(name, payload)
        for name in explicit_directories:
            archive.writestr(name, b"")


def _insert_zip_bytes(path: Path, *, offset: int, payload: bytes) -> None:
    """Insert bytes before the central directory while preserving central ZIP coordinates."""

    if not payload:
        raise AssertionError("ZIP mutation payload must be non-empty")
    original = bytearray(path.read_bytes())
    eocd_offset = original.rfind(b"PK\x05\x06")
    if eocd_offset < 0:
        raise AssertionError("wheel fixture has no EOCD")
    member_count = struct.unpack_from("<H", original, eocd_offset + 10)[0]
    central_size = struct.unpack_from("<L", original, eocd_offset + 12)[0]
    central_offset = struct.unpack_from("<L", original, eocd_offset + 16)[0]
    if not 0 <= offset <= central_offset:
        raise AssertionError("ZIP mutation must precede the central directory")

    original[offset:offset] = payload
    displacement = len(payload)
    shifted_central_offset = central_offset + displacement
    cursor = shifted_central_offset
    for _ in range(member_count):
        if original[cursor : cursor + 4] != b"PK\x01\x02":
            raise AssertionError("wheel fixture central directory is malformed")
        name_size, extra_size, comment_size = struct.unpack_from("<3H", original, cursor + 28)
        local_offset = struct.unpack_from("<L", original, cursor + 42)[0]
        if local_offset >= offset:
            struct.pack_into("<L", original, cursor + 42, local_offset + displacement)
        cursor += 46 + name_size + extra_size + comment_size
    if cursor != shifted_central_offset + central_size:
        raise AssertionError("wheel fixture central directory size is inconsistent")
    shifted_eocd_offset = eocd_offset + displacement
    struct.pack_into("<L", original, shifted_eocd_offset + 16, shifted_central_offset)
    path.write_bytes(original)


def _set_tar_device_major(path: Path, *, member_name: str, value: int) -> None:
    """Inject an otherwise ignored device field into one regular tar header."""

    expanded = bytearray(gzip.decompress(path.read_bytes()))
    header_offset = next(
        (
            offset
            for offset in range(0, len(expanded), tarfile.BLOCKSIZE)
            if expanded[offset : offset + 100].rstrip(b"\0").decode("ascii") == member_name
        ),
        -1,
    )
    if header_offset < 0:
        raise AssertionError("source-distribution fixture member was not found")
    expanded[header_offset + 329 : header_offset + 337] = f"{value:07o}\0".encode("ascii")
    expanded[header_offset + 148 : header_offset + 156] = b"        "
    checksum = sum(expanded[header_offset : header_offset + tarfile.BLOCKSIZE])
    expanded[header_offset + 148 : header_offset + 156] = f"{checksum:06o}\0 ".encode("ascii")
    path.write_bytes(
        _gzip_member(
            bytes(expanded),
            filename=path.name.removesuffix(".gz"),
        )
    )


def _write_sdist(
    path: Path,
    *,
    alias_readme: bool = False,
    collide_docs: bool = False,
    extra_files: tuple[tuple[str, bytes], ...] = (),
    extra_directories: tuple[str, ...] = (),
    member_pax_headers: dict[str, str] | None = None,
    global_pax_headers: dict[str, str] | None = None,
    readme_linkname: str = "",
    readme_mode: int | None = None,
    readme_devmajor: int = 0,
) -> None:
    root = "signalattice-0.2.1"
    files = {
        f"{root}/{name}": b"# bounded source fixture\n" for name in verifier._REQUIRED_SDIST_FILES
    }
    files[f"{root}/PKG-INFO"] = b"Metadata-Version: 2.4\nName: signalattice\n"
    files[f"{root}/setup.cfg"] = b"[metadata]\nname = signalattice\n"
    for relative_name in verifier._SDIST_EGG_INFO_FILES:
        files[f"{root}/{relative_name}"] = b"# bounded build-metadata fixture\n"
    if alias_readme:
        files[f"{root}/./README.md"] = files.pop(f"{root}/README.md")
    if collide_docs:
        files[f"{root}/docs"] = b"not a directory"
    for name, payload in extra_files:
        if name in files:
            raise AssertionError("source-distribution fixture entries must be unique")
        files[name] = payload

    with tarfile.open(
        path,
        mode="w:gz",
        format=tarfile.PAX_FORMAT,
        pax_headers=global_pax_headers,
    ) as archive:
        for name, payload in sorted(files.items()):
            member = tarfile.TarInfo(name)
            member.size = len(payload)
            if name == f"{root}/README.md":
                member.linkname = readme_linkname
                member.devmajor = readme_devmajor
                if readme_mode is not None:
                    member.mode = readme_mode
                if member_pax_headers is not None:
                    member.pax_headers = dict(member_pax_headers)
            archive.addfile(member, io.BytesIO(payload))
        for name in extra_directories:
            member = tarfile.TarInfo(name)
            member.type = tarfile.DIRTYPE
            archive.addfile(member)
    if readme_devmajor:
        _set_tar_device_major(
            path,
            member_name=f"{root}/README.md",
            value=readme_devmajor,
        )


def test_valid_minimal_wheel_and_sdist_satisfy_the_contract(tmp_path: Path) -> None:
    wheel = tmp_path / "signalattice-0.2.1-py3-none-any.whl"
    sdist = tmp_path / "signalattice-0.2.1.tar.gz"
    _write_wheel(wheel)
    _write_sdist(sdist)

    verifier.verify_wheel(wheel)
    verifier.verify_sdist(sdist)


@pytest.mark.parametrize(
    ("metadata", "message"),
    [
        (b"\xff", "core metadata is malformed"),
        (_service_metadata(provides_service=False), "exactly one service extra"),
        (
            _service_metadata(unconditional_dependencies=("fastapi",)),
            "not isolated behind one optional extra",
        ),
        (
            _service_metadata(service_dependencies=("fastapi", "starlette")),
            "omits a required framework dependency",
        ),
        (
            _service_metadata(service_dependencies=("fastapi", "starlette", "uvicorn", "httpx")),
            "outside its framework allowlist",
        ),
        (
            _service_metadata(service_dependencies=("fastapi", "starlette", "uvicorn", "fastapi")),
            "duplicate dependency",
        ),
    ],
)
def test_wheel_service_frameworks_are_confined_to_the_exact_optional_extra(
    tmp_path: Path,
    metadata: bytes,
    message: str,
) -> None:
    wheel = tmp_path / "service-boundary.whl"
    _write_wheel(wheel, metadata=metadata)

    with pytest.raises(verifier.DistributionContractError, match=message):
        verifier.verify_wheel(wheel)


@pytest.mark.parametrize(
    "alias",
    [
        "./quant_platform/tracking/cas.py",
        "quant_platform/./tracking/cas.py",
        "quant_platform/tracking/../tracking/cas.py",
    ],
)
def test_wheel_rejects_normalized_alias_overwrite_paths(tmp_path: Path, alias: str) -> None:
    wheel = tmp_path / "alias.whl"
    _write_wheel(wheel, extra_entries=((alias, b"sentinel overwrite"),))

    with pytest.raises(verifier.DistributionContractError, match="canonical|unsafe"):
        verifier.verify_wheel(wheel)


def test_wheel_rejects_duplicate_canonical_names_and_prefix_collisions(tmp_path: Path) -> None:
    duplicate = tmp_path / "duplicate.whl"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        _write_wheel(
            duplicate,
            extra_entries=(("quant_platform/tracking/cas.py", b"sentinel overwrite"),),
        )
    with pytest.raises(verifier.DistributionContractError, match="duplicate canonical"):
        verifier.verify_wheel(duplicate)

    collision = tmp_path / "collision.whl"
    _write_wheel(collision, extra_entries=(("quant_platform", b"not a directory"),))
    with pytest.raises(verifier.DistributionContractError, match="file/directory"):
        verifier.verify_wheel(collision)


@pytest.mark.parametrize(
    "member_name",
    [
        "quant_platform/tracking/CAS.py",
        "quant_platform/tracking/trailing.",
        "quant_platform/tracking/trailing ",
        "quant_platform/tracking/payload:stream.py",
        "quant_platform/tracking/CON.py",
    ],
)
def test_wheel_rejects_case_aliases_and_windows_hazardous_components(
    tmp_path: Path,
    member_name: str,
) -> None:
    wheel = tmp_path / "portable-paths.whl"
    _write_wheel(
        wheel,
        recorded_extra_entries=((member_name, b"# portable path adversary\n"),),
    )

    with pytest.raises(
        verifier.DistributionContractError,
        match="portable|Windows|reserved",
    ):
        verifier.verify_wheel(wheel)


def test_wheel_rejects_case_insensitive_prefix_collisions(tmp_path: Path) -> None:
    wheel = tmp_path / "portable-prefix.whl"
    _write_wheel(
        wheel,
        recorded_extra_entries=(("QUANT_PLATFORM", b"portable prefix adversary"),),
    )

    with pytest.raises(verifier.DistributionContractError, match="file/directory"):
        verifier.verify_wheel(wheel)


def test_wheel_required_resources_must_be_files_and_zip_types_must_agree(tmp_path: Path) -> None:
    source = tmp_path / "source.whl"
    spoofed = tmp_path / "directory-spoof.whl"
    _write_wheel(source)
    required = "quant_platform/tracking/cas.py"
    with zipfile.ZipFile(source) as original, zipfile.ZipFile(spoofed, mode="w") as rewritten:
        for member in original.infolist():
            if member.filename != required:
                rewritten.writestr(member, original.read(member))
        rewritten.writestr(required + "/", b"")
    with pytest.raises(
        verifier.DistributionContractError,
        match="explicit directory|omits a required",
    ):
        verifier.verify_wheel(spoofed)

    type_mismatch = tmp_path / "type-mismatch.whl"
    with zipfile.ZipFile(source) as original, zipfile.ZipFile(type_mismatch, mode="w") as rewritten:
        for member in original.infolist():
            rewritten.writestr(member, original.read(member))
        hostile = zipfile.ZipInfo("quant_platform/type-mismatch.py")
        hostile.external_attr = (stat.S_IFDIR | 0o755) << 16
        rewritten.writestr(hostile, b"")
    with pytest.raises(verifier.DistributionContractError, match="type disagrees"):
        verifier.verify_wheel(type_mismatch)


@pytest.mark.parametrize("defect", ["hash", "size", "missing", "huge_size"])
def test_wheel_record_must_bind_every_actual_member(
    tmp_path: Path,
    defect: str,
) -> None:
    wheel = tmp_path / f"record-{defect}.whl"
    _write_wheel(wheel, record_defect=defect)

    with pytest.raises(verifier.DistributionContractError, match="RECORD"):
        verifier.verify_wheel(wheel)


def test_wheel_record_rejects_unrecorded_allowlisted_file(tmp_path: Path) -> None:
    wheel = tmp_path / "unrecorded.whl"
    _write_wheel(wheel, extra_entries=(("quant_platform/unrecorded.py", b"pass\n"),))

    with pytest.raises(verifier.DistributionContractError, match="exact file inventory"):
        verifier.verify_wheel(wheel)


def test_wheel_rejects_undeclared_dist_info_payloads(tmp_path: Path) -> None:
    wheel = tmp_path / "private-metadata.whl"
    _write_wheel(
        wheel,
        recorded_extra_entries=(
            (f"{_DIST_INFO}/operator-private.txt", b"undisclosed metadata payload"),
        ),
    )

    with pytest.raises(verifier.DistributionContractError, match="exact publication inventory"):
        verifier.verify_wheel(wheel)


def test_wheel_rejects_undeclared_directories(tmp_path: Path) -> None:
    wheel = tmp_path / "private-directory.whl"
    _write_wheel(
        wheel,
        explicit_directories=("quant_platform/operator-private/",),
    )

    with pytest.raises(verifier.DistributionContractError, match="explicit directory"):
        verifier.verify_wheel(wheel)


@pytest.mark.parametrize("metadata_kind", ["archive_comment", "member_comment", "member_extra"])
def test_wheel_rejects_metadata_outside_record(
    tmp_path: Path,
    metadata_kind: str,
) -> None:
    source = tmp_path / "source.whl"
    hostile = tmp_path / f"{metadata_kind}.whl"
    _write_wheel(source)
    with zipfile.ZipFile(source) as original, zipfile.ZipFile(hostile, mode="w") as rewritten:
        if metadata_kind == "archive_comment":
            rewritten.comment = b"undeclared archive metadata"
        for member in original.infolist():
            payload = original.read(member)
            if member.filename.endswith("/METADATA"):
                if metadata_kind == "member_comment":
                    member.comment = b"undeclared member metadata"
                elif metadata_kind == "member_extra":
                    member.extra = b"\xfe\xca\x00\x00"
            rewritten.writestr(member, payload)

    with pytest.raises(verifier.DistributionContractError, match="comment|ZIP metadata"):
        verifier.verify_wheel(hostile)


def test_wheel_rejects_metadata_present_only_in_a_local_header(tmp_path: Path) -> None:
    wheel = tmp_path / "local-extra.whl"
    _write_wheel(wheel)
    with zipfile.ZipFile(wheel) as archive:
        target = archive.infolist()[0]
    raw = wheel.read_bytes()
    name_size, extra_size = struct.unpack_from("<2H", raw, target.header_offset + 26)
    assert extra_size == 0
    insertion = target.header_offset + verifier._ZIP_LOCAL_HEADER_BYTES + name_size
    local_extra = b"\xfe\xca\x00\x00"
    _insert_zip_bytes(wheel, offset=insertion, payload=local_extra)
    mutated = bytearray(wheel.read_bytes())
    struct.pack_into("<H", mutated, target.header_offset + 28, len(local_extra))
    wheel.write_bytes(mutated)

    with pytest.raises(verifier.DistributionContractError, match="local header.*ZIP metadata"):
        verifier.verify_wheel(wheel)


@pytest.mark.parametrize("layout_kind", ["prefix", "gap"])
def test_wheel_rejects_bytes_outside_exact_local_member_extents(
    tmp_path: Path,
    layout_kind: str,
) -> None:
    wheel = tmp_path / f"{layout_kind}.whl"
    _write_wheel(wheel)
    if layout_kind == "prefix":
        insertion = 0
    else:
        with zipfile.ZipFile(wheel) as archive:
            insertion = sorted(member.header_offset for member in archive.infolist())[1]
    _insert_zip_bytes(wheel, offset=insertion, payload=b"undeclared-layout-bytes")

    with pytest.raises(verifier.DistributionContractError, match="prefix, gap, or overlapping"):
        verifier.verify_wheel(wheel)


def test_sdist_rejects_normalized_aliases_and_prefix_collisions(tmp_path: Path) -> None:
    alias = tmp_path / "alias.tar.gz"
    _write_sdist(alias, alias_readme=True)
    with pytest.raises(verifier.DistributionContractError, match="canonical"):
        verifier.verify_sdist(alias)

    collision = tmp_path / "collision.tar.gz"
    _write_sdist(collision, collide_docs=True)
    with pytest.raises(verifier.DistributionContractError, match="file/directory"):
        verifier.verify_sdist(collision)


@pytest.mark.parametrize(
    "relative_name",
    [
        "README.MD",
        "tests/trailing.",
        "tests/trailing ",
        "tests/payload:stream.py",
        "tests/NUL.py",
    ],
)
def test_sdist_rejects_case_aliases_and_windows_hazardous_components(
    tmp_path: Path,
    relative_name: str,
) -> None:
    sdist = tmp_path / "portable-paths.tar.gz"
    _write_sdist(
        sdist,
        extra_files=((f"signalattice-0.2.1/{relative_name}", b"# portable path adversary\n"),),
    )

    with pytest.raises(
        verifier.DistributionContractError,
        match="portable|Windows|reserved",
    ):
        verifier.verify_sdist(sdist)


def test_sdist_rejects_case_insensitive_prefix_collisions(tmp_path: Path) -> None:
    sdist = tmp_path / "portable-prefix.tar.gz"
    _write_sdist(
        sdist,
        extra_files=(("signalattice-0.2.1/DOCS", b"portable prefix adversary"),),
    )

    with pytest.raises(verifier.DistributionContractError, match="file/directory"):
        verifier.verify_sdist(sdist)


def test_sdist_rejects_undeclared_egg_info_payloads(tmp_path: Path) -> None:
    sdist = tmp_path / "private-metadata.tar.gz"
    _write_sdist(
        sdist,
        extra_files=(
            (
                "signalattice-0.2.1/src/signalattice.egg-info/operator-private.txt",
                b"undisclosed metadata payload",
            ),
        ),
    )

    with pytest.raises(verifier.DistributionContractError, match="outside its allowlist"):
        verifier.verify_sdist(sdist)


def test_sdist_rejects_undeclared_directories(tmp_path: Path) -> None:
    sdist = tmp_path / "private-directory.tar.gz"
    _write_sdist(
        sdist,
        extra_directories=("signalattice-0.2.1/operator-private",),
    )

    with pytest.raises(verifier.DistributionContractError, match="explicit directory"):
        verifier.verify_sdist(sdist)


@pytest.mark.parametrize(
    "headers",
    [
        {"comment": "undeclared member metadata"},
        {"SCHILY.xattr.user.private": "undeclared extended attribute"},
        {"comment": "x" * (128 * 1024)},
    ],
)
def test_sdist_rejects_undeclared_member_pax_metadata(
    tmp_path: Path,
    headers: dict[str, str],
) -> None:
    sdist = tmp_path / "member-pax.tar.gz"
    _write_sdist(sdist, member_pax_headers=headers)

    with pytest.raises(verifier.DistributionContractError, match="member PAX metadata"):
        verifier.verify_sdist(sdist)


def test_sdist_rejects_global_pax_metadata(tmp_path: Path) -> None:
    sdist = tmp_path / "global-pax.tar.gz"
    _write_sdist(
        sdist,
        global_pax_headers={"comment": "undeclared global metadata"},
    )

    with pytest.raises(verifier.DistributionContractError, match="global PAX metadata"):
        verifier.verify_sdist(sdist)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"readme_linkname": "undeclared-target"}, "link target"),
        ({"readme_mode": 0o4644}, "permissions"),
        ({"readme_mode": 0o666}, "permissions"),
        ({"readme_devmajor": 7}, "device identifiers"),
    ],
)
def test_sdist_rejects_undeclared_regular_file_header_metadata(
    tmp_path: Path,
    kwargs: dict[str, object],
    message: str,
) -> None:
    sdist = tmp_path / "tar-header-metadata.tar.gz"
    _write_sdist(sdist, **kwargs)  # type: ignore[arg-type]

    with pytest.raises(verifier.DistributionContractError, match=message):
        verifier.verify_sdist(sdist)


def test_sdist_rejects_longname_metadata_beyond_portable_component_bounds(
    tmp_path: Path,
) -> None:
    sdist = tmp_path / "longname.tar.gz"
    component = "x" * (verifier._MAX_ARCHIVE_COMPONENT_BYTES + 1)
    _write_sdist(
        sdist,
        extra_files=((f"signalattice-0.2.1/tests/{component}.py", b"pass\n"),),
    )

    with pytest.raises(verifier.DistributionContractError, match="component|member PAX metadata"):
        verifier.verify_sdist(sdist)


def test_archive_member_file_and_container_ceilings_precede_semantic_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wheel = tmp_path / "bounded.whl"
    _write_wheel(wheel)

    monkeypatch.setattr(verifier, "_MAX_ARCHIVE_MEMBERS", 1)
    with pytest.raises(verifier.DistributionContractError, match="member-count"):
        verifier.verify_wheel(wheel)

    monkeypatch.setattr(verifier, "_MAX_ARCHIVE_MEMBERS", 4_096)
    monkeypatch.setattr(verifier, "_MAX_ARCHIVE_FILES", 1)
    with pytest.raises(verifier.DistributionContractError, match="file-count"):
        verifier.verify_wheel(wheel)

    monkeypatch.setattr(verifier, "_MAX_ARCHIVE_FILES", 2_048)
    monkeypatch.setattr(verifier, "_MAX_ARCHIVE_CONTAINER_BYTES", wheel.stat().st_size - 1)
    with pytest.raises(verifier.DistributionContractError, match="container byte"):
        verifier.verify_wheel(wheel)


def test_sdist_stream_stops_at_the_member_ceiling(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sdist = tmp_path / "bounded.tar.gz"
    _write_sdist(sdist)
    monkeypatch.setattr(verifier, "_MAX_ARCHIVE_MEMBERS", 1)

    with pytest.raises(verifier.DistributionContractError, match="member-count"):
        verifier.verify_sdist(sdist)


def test_sdist_gzip_expansion_is_bounded_before_tar_metadata_parsing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sdist = tmp_path / "bounded.tar.gz"
    _write_sdist(sdist)
    monkeypatch.setattr(verifier, "_MAX_EXPANDED_TAR_BYTES", 1)

    def reject_tar_parser_entry(*_args: object, **_kwargs: object) -> tarfile.TarFile:
        raise AssertionError("tar metadata parsing preceded the expanded-byte ceiling")

    monkeypatch.setattr(verifier.tarfile, "open", reject_tar_parser_entry)

    with pytest.raises(verifier.DistributionContractError, match="expanded-tar byte ceiling"):
        verifier.verify_sdist(sdist)


def test_sdist_rejects_nonzero_bytes_after_the_first_tar_end_marker(tmp_path: Path) -> None:
    sdist = tmp_path / "hidden-trailer.tar.gz"
    _write_sdist(sdist)
    expanded = gzip.decompress(sdist.read_bytes())
    sdist.write_bytes(
        _gzip_member(
            expanded + (b"X" * tarfile.BLOCKSIZE),
            filename="hidden-trailer.tar",
        )
    )

    with pytest.raises(verifier.DistributionContractError, match="hidden bytes after tar EOF"):
        verifier.verify_sdist(sdist)


def test_sdist_rejects_concatenated_gzip_payloads(tmp_path: Path) -> None:
    sdist = tmp_path / "concatenated.tar.gz"
    _write_sdist(sdist)
    second_stream = _gzip_member(
        b"X" * tarfile.BLOCKSIZE,
        filename="second.tar",
    )
    sdist.write_bytes(sdist.read_bytes() + second_stream)

    with pytest.raises(verifier.DistributionContractError, match="multiple gzip members"):
        verifier.verify_sdist(sdist)


def test_sdist_rejects_a_concatenated_empty_gzip_member(tmp_path: Path) -> None:
    sdist = tmp_path / "concatenated-empty.tar.gz"
    _write_sdist(sdist)
    sdist.write_bytes(
        sdist.read_bytes()
        + _gzip_member(
            b"",
            filename="hidden-empty-member.tar",
        )
    )

    with pytest.raises(verifier.DistributionContractError, match="multiple gzip members"):
        verifier.verify_sdist(sdist)


@pytest.mark.parametrize("metadata_kind", ["filename", "comment", "extra"])
def test_sdist_rejects_undeclared_gzip_header_metadata(
    tmp_path: Path,
    metadata_kind: str,
) -> None:
    sdist = tmp_path / "gzip-metadata.tar.gz"
    _write_sdist(sdist)
    expanded = gzip.decompress(sdist.read_bytes())
    filename = "undeclared-name.tar" if metadata_kind == "filename" else "gzip-metadata.tar"
    sdist.write_bytes(
        _gzip_member(
            expanded,
            filename=filename,
            comment=(b"undeclared comment" if metadata_kind == "comment" else None),
            extra=(b"undeclared extra" if metadata_kind == "extra" else None),
        )
    )

    with pytest.raises(
        verifier.DistributionContractError,
        match="gzip header|gzip filename",
    ):
        verifier.verify_sdist(sdist)


@pytest.mark.parametrize("extra_kind", ["file", "directory", "symlink"])
def test_distribution_directory_contains_only_the_verified_archive_pair(
    tmp_path: Path,
    extra_kind: str,
) -> None:
    directory = tmp_path / "dist"
    directory.mkdir()
    wheel = directory / "signalattice-0.2.1-py3-none-any.whl"
    sdist = directory / "signalattice-0.2.1.tar.gz"
    _write_wheel(wheel)
    _write_sdist(sdist)
    assert verifier._distribution_pair(directory) == (wheel, sdist)

    extra = directory / "operator-private"
    if extra_kind == "file":
        extra.write_text("undeclared publication payload", encoding="utf-8")
    elif extra_kind == "directory":
        extra.mkdir()
    else:
        extra.symlink_to(wheel)

    with pytest.raises(verifier.DistributionContractError, match="exactly the wheel"):
        verifier._distribution_pair(directory)


def _canonical_padded_stream(logical_end: int) -> io.BytesIO:
    """Return a stream padded exactly as tarfile pads a closed archive.

    Two zero blocks for the terminator, then zero fill to the next RECORDSIZE
    boundary. Content before ``logical_end`` is irrelevant to the padding check.
    """
    total = logical_end + 2 * tarfile.BLOCKSIZE
    fill = (-total) % tarfile.RECORDSIZE
    return io.BytesIO(b"\x00" * (total + fill))


@pytest.mark.parametrize(
    "logical_end",
    [
        0,
        tarfile.BLOCKSIZE,
        # 9728 makes tarfile emit its largest legitimate fill, pushing total
        # padding to 10752 -- above RECORDSIZE. A bound of
        # "padding_size > RECORDSIZE" rejected this canonical archive, so the
        # sdist gate failed or passed purely on content size.
        tarfile.RECORDSIZE - tarfile.BLOCKSIZE,
        tarfile.RECORDSIZE,
        2 * tarfile.RECORDSIZE - tarfile.BLOCKSIZE,
    ],
)
def test_canonical_tar_padding_is_accepted_at_every_fill_width(logical_end: int) -> None:
    stream = _canonical_padded_stream(logical_end)
    verifier._verify_tar_end_padding(
        stream,
        logical_end=logical_end,
        expanded_size=len(stream.getvalue()),
    )


def test_an_extra_whole_record_of_zeros_is_still_rejected() -> None:
    """The bound moved to the fill; a surplus record must still fail."""
    logical_end = tarfile.RECORDSIZE - tarfile.BLOCKSIZE
    stream = _canonical_padded_stream(logical_end)
    padded = stream.getvalue() + b"\x00" * tarfile.RECORDSIZE
    with pytest.raises(verifier.DistributionContractError, match="not canonical"):
        verifier._verify_tar_end_padding(
            io.BytesIO(padded),
            logical_end=logical_end,
            expanded_size=len(padded),
        )


def test_hidden_nonzero_bytes_after_eof_are_still_rejected() -> None:
    logical_end = tarfile.BLOCKSIZE
    stream = _canonical_padded_stream(logical_end)
    payload = bytearray(stream.getvalue())
    payload[-1] = 0x01
    with pytest.raises(verifier.DistributionContractError, match="hidden bytes"):
        verifier._verify_tar_end_padding(
            io.BytesIO(bytes(payload)),
            logical_end=logical_end,
            expanded_size=len(payload),
        )
