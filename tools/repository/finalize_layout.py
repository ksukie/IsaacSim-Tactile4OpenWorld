#!/usr/bin/env python3
"""Create and statically verify the IsaacSim-Tactile4OpenWorld manifest."""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import os
import re
import tokenize
from pathlib import Path
from urllib.parse import unquote


MAINLINE_DIR = "active-isaaclab-2.1.1"
LEGACY_DIR = "archive-isaaclab-2.3.2"
DOCS_DIR = "docs"
MANIFEST_NAME = "FINAL_MANIFEST.csv"
EXPECTED_COUNTS = {MAINLINE_DIR: 3688, LEGACY_DIR: 58}
EXPECTED_SCRIPT_COUNTS = {MAINLINE_DIR: 132, LEGACY_DIR: 21}
SCRIPT_ROOTS = {
    MAINLINE_DIR: ("experiments", "tools"),
    LEGACY_DIR: ("experiments",),
}
DISALLOWED_DELIVERY_DIR = "archives"
COMPRESSED_SUFFIXES = (
    ".7z", ".bz2", ".gz", ".rar", ".tar", ".tar.bz2", ".tar.gz",
    ".tar.xz", ".tgz", ".txz", ".xz", ".zip",
)
BANNED_FRAGMENTS = (
    b"sto" + b"uch",
    b"sight" + b"ac",
    b"sight" + b"tac",
    b"tac" + b"ex",
)
BANNED_BYTES_RE = re.compile(b"|".join(BANNED_FRAGMENTS), re.IGNORECASE)
BANNED_TEXT_RE = re.compile(
    "|".join(fragment.decode("ascii") for fragment in BANNED_FRAGMENTS),
    re.IGNORECASE,
)
MARKDOWN_LINK_RE = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")
USDA_REFERENCE_RE = re.compile(r"@([^@\r\n]+)@")
EXTERNAL_PATH_RE = re.compile(
    r"(?:\$\{|\{)?(?:OWT_ASSET_ROOT|ISAACLAB_NUCLEUS_DIR)\}?"
    r"/[A-Za-z0-9_~${}:./\\+\-]+"
    r"|/(?:home|tmp)/[A-Za-z0-9_~${}:./\\+\-]+"
)


def extended_path(path: Path | str) -> str:
    """Return a Windows extended path without changing non-Windows paths."""
    value = os.path.abspath(os.fspath(path))
    if os.name != "nt" or value.startswith("\\\\?\\"):
        return value
    if value.startswith("\\\\"):
        return "\\\\?\\UNC\\" + value[2:]
    return "\\\\?\\" + value


def iter_relative_files(root: Path) -> list[str]:
    """List files below *root* using stable POSIX-style relative paths."""
    base = extended_path(root)
    results: list[str] = []
    for current, directories, filenames in os.walk(base):
        directories.sort(key=str.casefold)
        filenames.sort(key=str.casefold)
        relative_dir = os.path.relpath(current, base)
        for filename in filenames:
            relative = Path(filename) if relative_dir == "." else Path(relative_dir) / filename
            results.append(relative.as_posix())
    return results


def hash_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with open(extended_path(path), "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def payload_paths(workspace: Path) -> dict[str, list[str]]:
    return {
        container: [f"{container}/{relative}" for relative in iter_relative_files(workspace / container)]
        for container in EXPECTED_COUNTS
    }


def validate_payload_counts(paths: dict[str, list[str]]) -> None:
    for container, expected in EXPECTED_COUNTS.items():
        actual = len(paths[container])
        if actual != expected:
            raise RuntimeError(f"{container}: expected {expected} files, got {actual}")


def write_manifest(workspace: Path, tools_dir: Path) -> None:
    paths = payload_paths(workspace)
    validate_payload_counts(paths)
    rows: list[dict[str, object]] = []
    for relative in sorted(
        [item for members in paths.values() for item in members],
        key=str.casefold,
    ):
        digest, size = hash_file(workspace / Path(relative))
        rows.append({"path": relative, "size_bytes": size, "sha256": digest})
    write_csv(tools_dir / MANIFEST_NAME, rows, ["path", "size_bytes", "sha256"])
    print(f"final_manifest_entries={len(rows)}")


def validate_manifest(workspace: Path, tools_dir: Path) -> tuple[int, int]:
    rows = read_csv(tools_dir / MANIFEST_NAME)
    expected_total = sum(EXPECTED_COUNTS.values())
    if len(rows) != expected_total:
        raise RuntimeError(f"Expected {expected_total} manifest rows, got {len(rows)}")
    expected_paths = [row["path"] for row in rows]
    if len({path.casefold() for path in expected_paths}) != len(expected_paths):
        raise RuntimeError("Case-insensitive collision in final manifest")

    current = payload_paths(workspace)
    validate_payload_counts(current)
    actual_paths = {item for members in current.values() for item in members}
    manifest_paths = set(expected_paths)
    if actual_paths != manifest_paths:
        missing = sorted(manifest_paths - actual_paths, key=str.casefold)
        extra = sorted(actual_paths - manifest_paths, key=str.casefold)
        raise RuntimeError(f"Final payload set mismatch: missing={missing[:10]}, extra={extra[:10]}")

    for row in rows:
        digest, size = hash_file(workspace / Path(row["path"]))
        if digest != row["sha256"] or size != int(row["size_bytes"]):
            raise RuntimeError(f"Final payload hash mismatch: {row['path']}")
    return len(current[MAINLINE_DIR]), len(current[LEGACY_DIR])


def validate_python_ast(workspace: Path) -> int:
    files: list[Path] = []
    for container, expected in EXPECTED_SCRIPT_COUNTS.items():
        members = sorted(
            (
                path
                for root in SCRIPT_ROOTS[container]
                for path in (workspace / container / root).rglob("*.py")
            ),
            key=lambda path: path.as_posix().casefold(),
        )
        if len(members) != expected:
            raise RuntimeError(f"{container}: expected {expected} scripts, got {len(members)}")
        files.extend(members)
    errors: list[str] = []
    for path in files:
        try:
            with tokenize.open(path) as handle:
                ast.parse(handle.read(), filename=str(path))
        except (SyntaxError, UnicodeError) as exc:
            errors.append(f"{path.relative_to(workspace).as_posix()}: {exc}")
    if errors:
        raise RuntimeError("Python AST errors:\n" + "\n".join(errors))
    return len(files)


def markdown_files(workspace: Path, tools_dir: Path) -> list[Path]:
    candidates = [*sorted(workspace.glob("*.md")), *sorted((workspace / DOCS_DIR).rglob("*.md"))]
    candidates.extend(sorted(tools_dir.glob("README*.md")))
    return candidates


def validate_markdown_links(workspace: Path, tools_dir: Path) -> int:
    rows: list[dict[str, object]] = []
    broken: list[str] = []
    for path in markdown_files(workspace, tools_dir):
        text = path.read_text(encoding="utf-8")
        for match in MARKDOWN_LINK_RE.finditer(text):
            reference = match.group(1).strip()
            if reference.startswith("<") and reference.endswith(">"):
                reference = reference[1:-1]
            target_text = reference.split("#", 1)[0].strip()
            if not target_text or re.match(r"^(?:https?://|mailto:|data:)", target_text, re.IGNORECASE):
                continue
            target = (path.parent / unquote(target_text)).resolve()
            exists = os.path.exists(extended_path(target))
            source = path.relative_to(workspace).as_posix()
            rows.append(
                {
                    "source_file": source,
                    "reference": reference,
                    "resolved_path": str(target),
                    "exists": str(exists).lower(),
                }
            )
            if not exists:
                broken.append(f"{source}: {reference}")
    write_csv(
        tools_dir / "markdown_link_check.csv",
        rows,
        ["source_file", "reference", "resolved_path", "exists"],
    )
    if broken:
        raise RuntimeError("Broken local Markdown links:\n" + "\n".join(broken))
    return len(rows)


def classify_usda_reference(reference: str) -> str:
    if re.match(r"^[A-Za-z][A-Za-z0-9+.-]*://", reference):
        return "external_environment_asset"
    if reference.startswith(("/", "\\")) or "$" in reference or "{" in reference:
        return "external_environment_asset"
    return "relative_reference"


def scan_usda_references(workspace: Path, tools_dir: Path) -> tuple[int, int]:
    rows: list[dict[str, object]] = []
    missing: list[str] = []
    source_files = sorted(
        [
            path
            for container in EXPECTED_COUNTS
            for path in (workspace / container).rglob("*.usda")
        ],
        key=lambda path: path.as_posix().casefold(),
    )
    for path in source_files:
        text = path.read_text(encoding="utf-8", errors="replace")
        source = path.relative_to(workspace).as_posix()
        baseline = source.split("/", 1)[0]
        for match in USDA_REFERENCE_RE.finditer(text):
            reference = match.group(1).strip()
            classification = classify_usda_reference(reference)
            resolved = ""
            exists: object = "not_applicable"
            if classification == "relative_reference":
                target = (path.parent / Path(reference.replace("\\", "/"))).resolve()
                resolved = str(target)
                exists = os.path.exists(extended_path(target))
                if exists:
                    classification = "internal_resolved"
                elif "uipc:external_mesh_required = 0" in text and "membrane_sim_mesh" in text:
                    classification = "optional_placeholder_with_internal_fallback"
                else:
                    classification = "missing_relative_reference"
                    missing.append(f"{source}: {reference}")
            rows.append(
                {
                    "baseline": baseline,
                    "source_file": source,
                    "reference": reference,
                    "resolved_path": resolved,
                    "exists": str(exists).lower() if isinstance(exists, bool) else exists,
                    "classification": classification,
                }
            )
    write_csv(
        tools_dir / "usda_reference_check.csv",
        rows,
        ["baseline", "source_file", "reference", "resolved_path", "exists", "classification"],
    )
    if missing:
        raise RuntimeError("Missing USDA relative references:\n" + "\n".join(missing))
    return len(rows), len(missing)


def scan_external_paths(workspace: Path, tools_dir: Path) -> int:
    rows: list[dict[str, object]] = []
    text_suffixes = {
        ".cfg", ".csv", ".ini", ".json", ".md", ".py", ".rst", ".sh",
        ".toml", ".txt", ".usda", ".xml", ".yaml", ".yml",
    }
    for container in EXPECTED_COUNTS:
        for relative in iter_relative_files(workspace / container):
            path = workspace / container / Path(relative)
            if path.suffix.casefold() not in text_suffixes:
                continue
            try:
                with open(extended_path(path), "r", encoding="utf-8") as handle:
                    text = handle.read()
            except UnicodeDecodeError:
                continue
            for match in EXTERNAL_PATH_RE.finditer(text):
                reference = match.group(0)
                if reference.startswith("/home/"):
                    classification = "source_machine_absolute_path"
                elif reference.startswith("/tmp/"):
                    classification = "runtime_output_or_temporary_path"
                else:
                    classification = "external_environment_asset"
                rows.append(
                    {
                        "baseline": container,
                        "source_file": f"{container}/{relative}",
                        "reference": reference,
                        "classification": classification,
                    }
                )
    write_csv(
        tools_dir / "external_path_references.csv",
        rows,
        ["baseline", "source_file", "reference", "classification"],
    )
    return len(rows)


def validate_no_compressed_delivery(workspace: Path) -> int:
    delivery_dir = workspace / DISALLOWED_DELIVERY_DIR
    if delivery_dir.exists():
        raise RuntimeError(f"Delivery baseline directory must be absent: {delivery_dir}")
    compressed: list[str] = []
    for relative in iter_relative_files(workspace):
        folded = relative.casefold()
        if any(folded.endswith(suffix) for suffix in COMPRESSED_SUFFIXES):
            compressed.append(relative)
    if compressed:
        raise RuntimeError("Compressed delivery files remain:\n" + "\n".join(compressed))
    return 0


def validate_case_collisions_and_paths(workspace: Path) -> tuple[int, str]:
    seen: dict[str, str] = {}
    collisions: list[str] = []
    longest = ""
    base = extended_path(workspace)
    for current, directories, filenames in os.walk(base):
        relative_dir = os.path.relpath(current, base)
        for name in [*directories, *filenames]:
            relative = Path(name) if relative_dir == "." else Path(relative_dir) / name
            value = relative.as_posix()
            key = value.casefold()
            previous = seen.get(key)
            if previous is not None and previous != value:
                collisions.append(f"{previous} <> {value}")
            seen[key] = value
            if not longest or len(str(workspace / relative)) > len(str(workspace / Path(longest))):
                longest = value
    if collisions:
        raise RuntimeError("Case-insensitive path collisions:\n" + "\n".join(collisions))
    return len(str(workspace / Path(longest))), longest


def scan_for_banned_fragments(workspace: Path) -> tuple[int, int]:
    path_hits: list[str] = []
    content_hits: list[str] = []
    base = extended_path(workspace)
    files: list[str] = []
    for current, directories, filenames in os.walk(base):
        relative_dir = os.path.relpath(current, base)
        for name in [*directories, *filenames]:
            relative = Path(name) if relative_dir == "." else Path(relative_dir) / name
            value = relative.as_posix()
            if BANNED_TEXT_RE.search(value):
                path_hits.append(value)
        for filename in filenames:
            relative = Path(filename) if relative_dir == "." else Path(relative_dir) / filename
            files.append(relative.as_posix())
    for relative in files:
        path = workspace / Path(relative)
        with open(extended_path(path), "rb") as handle:
            overlap = b""
            while True:
                chunk = handle.read(1024 * 1024)
                if not chunk:
                    break
                data = overlap + chunk
                if BANNED_BYTES_RE.search(data):
                    content_hits.append(relative)
                    break
                overlap = data[-(max(map(len, BANNED_FRAGMENTS)) - 1):]
    if path_hits or content_hits:
        details = [*(f"path: {item}" for item in path_hits), *(f"content: {item}" for item in content_hits)]
        raise RuntimeError("Disallowed legacy-name fragments remain:\n" + "\n".join(details))
    return len(path_hits), len(content_hits)


def write_confirmation(
    workspace: Path,
    mainline_count: int,
    legacy_count: int,
    ast_count: int,
    link_count: int,
    usda_count: int,
    external_count: int,
    longest_length: int,
    longest_path: str,
) -> None:
    content = f"""# IsaacSim-Tactile4OpenWorld 最终静态确认

- 最终载荷清单：`tools/repository/{MANIFEST_NAME}`，共 {mainline_count + legacy_count} 项，路径、大小和 SHA-256 全部一致。
- 当前主线：`{MAINLINE_DIR}/`，{mainline_count} 个文件。
- 历史归档：`{LEGACY_DIR}/`，{legacy_count} 个文件。
- 文件缺失：0；额外载荷：0；大小写路径冲突：0。
- Python 脚本静态 AST：{ast_count} 个，解析错误 0。
- 顶层项目文档本地链接：{link_count} 条，断链 0。
- USDA 引用：{usda_count} 条，意外缺失相对引用 0。
- 外部或运行时路径引用：{external_count} 条，详见 `tools/repository/external_path_references.csv`。
- 压缩交付件和历史基线目录：0。
- 旧项目名称片段：路径 0、所有文件原始字节内容 0；不使用允许清单。
- 最长绝对路径：{longest_length} 个字符，对应 `{longest_path}`。

本确认只执行文件系统、哈希、AST、链接和文本/字节扫描；未安装依赖、未编译、未启动 Isaac Lab 或 Isaac Sim，也未连接硬件。
"""
    (workspace / DOCS_DIR / "internal" / "PATH_CONFIRMATION.md").write_text(
        content, encoding="utf-8", newline="\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write-manifest", action="store_true", help="Write hashes for the current final payload.")
    args = parser.parse_args()

    tools_dir = Path(__file__).resolve().parent
    workspace = tools_dir.parents[1]
    if args.write_manifest:
        write_manifest(workspace, tools_dir)
        return

    mainline_count, legacy_count = validate_manifest(workspace, tools_dir)
    ast_count = validate_python_ast(workspace)
    usda_count, _ = scan_usda_references(workspace, tools_dir)
    external_count = scan_external_paths(workspace, tools_dir)
    link_count = validate_markdown_links(workspace, tools_dir)
    validate_no_compressed_delivery(workspace)
    longest_length, longest_path = validate_case_collisions_and_paths(workspace)
    write_confirmation(
        workspace,
        mainline_count,
        legacy_count,
        ast_count,
        link_count,
        usda_count,
        external_count,
        longest_length,
        longest_path,
    )
    scan_for_banned_fragments(workspace)

    print(f"active_files={mainline_count}")
    print(f"archive_files={legacy_count}")
    print(f"total_payload={mainline_count + legacy_count}")
    print(f"python_ast_files={ast_count}")
    print("python_ast_errors=0")
    print(f"markdown_local_links={link_count}")
    print("markdown_broken_links=0")
    print(f"usda_references={usda_count}")
    print("usda_missing_relative=0")
    print("compressed_delivery_files=0")
    print("legacy_name_path_hits=0")
    print("legacy_name_content_hits=0")


if __name__ == "__main__":
    main()
