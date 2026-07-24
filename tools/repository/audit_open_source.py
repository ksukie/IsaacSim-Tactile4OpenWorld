#!/usr/bin/env python3
"""Run the repository's dependency-free open-source release checks."""

from __future__ import annotations

import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
REQUIRED_FILES = (
    "LICENSE",
    "NOTICE",
    "THIRD_PARTY_NOTICES.md",
    "CITATIONS.md",
    "CITATION.cff",
    "CONTRIBUTING.md",
    "SECURITY.md",
)
REQUIRED_LICENSES = (
    "archive-isaaclab-2.3.2/LICENSE",
    "active-isaaclab-2.1.1/vendor/agilex-piper/piper_ros/LICENSE",
    "active-isaaclab-2.1.1/packages/uipc/libuipc/LICENSE",
    "active-isaaclab-2.1.1/packages/uipc/libuipc/python/LICENSE",
    "active-isaaclab-2.1.1/packages/uipc/libuipc/external/tetgen/LICENSE",
    "active-isaaclab-2.1.1/packages/uipc/libuipc/external/muda/LICENSE",
    "active-isaaclab-2.1.1/packages/uipc/libuipc/external/muda/external/catch2/LICENSE_1_0.txt",
    "active-isaaclab-2.1.1/packages/uipc/libuipc/external/octree/Octree/LICENSE",
    "active-isaaclab-2.1.1/packages/uipc/libuipc/src/geometry/bvh/LICENSE-SimpleBVH",
    "active-isaaclab-2.1.1/packages/uipc/libuipc/src/geometry/bvh/LICENSE-NSEssentials",
    "active-isaaclab-2.1.1/packages/core/openworldtactile/simulation_approaches/fots/LICENSE",
    "active-isaaclab-2.1.1/packages/core/openworldtactile/simulation_approaches/fem_based/sim/LICENSE",
    "active-isaaclab-2.1.1/packages/assets/openworldtactile_assets/data/Sensors/GelSight_Mini/LICENSE",
    "active-isaaclab-2.1.1/packages/assets/openworldtactile_assets/data/Robots/Franka/GelSight_Mini/LICENSE",
    "active-isaaclab-2.1.1/packages/assets/openworldtactile_assets/data/Robots/Franka/LICENSE-APACHE-2.0",
    "active-isaaclab-2.1.1/vendor/agilex-piper/piper_ros/src/piper_moveit/moveit-1.1.11/LICENSE.txt",
)
DISALLOWED_SUFFIXES = {".ckpt", ".dll", ".dylib", ".exe", ".pt", ".pyd", ".so"}
GENERATED_DIR_NAMES = {".mypy_cache", ".pytest_cache", ".ruff_cache", "__pycache__"}
SECRET_PATTERNS = {
    "private key": re.compile(rb"-----BEGIN (?:EC |OPENSSH |RSA )?PRIVATE KEY-----"),
    "GitHub token": re.compile(rb"\bgh[pousr]_[A-Za-z0-9]{20,}\b"),
    "AWS access key": re.compile(rb"\bAKIA[0-9A-Z]{16}\b"),
    "OpenAI-style key": re.compile(rb"\bsk-[A-Za-z0-9_-]{20,}\b"),
}
TEXT_SUFFIXES = {
    "", ".cfg", ".cff", ".cmake", ".csv", ".ini", ".json", ".md",
    ".py", ".rst", ".sh", ".toml", ".txt", ".xml", ".yaml", ".yml",
}


def relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def main() -> int:
    failures: list[str] = []

    for name in (*REQUIRED_FILES, *REQUIRED_LICENSES):
        if not (ROOT / name).is_file():
            failures.append(f"missing required file: {name}")

    for path in ROOT.rglob("*"):
        if path.is_dir() and (path.name in GENERATED_DIR_NAMES or path.name.endswith(".egg-info")):
            failures.append(f"generated directory is tracked: {relative(path)}")
        elif path.is_file() and path.suffix.casefold() in DISALLOWED_SUFFIXES:
            failures.append(f"opaque/native release artifact is tracked: {relative(path)}")

    excluded_paths = (
        "archive-isaaclab-2.3.2/hardware-sdk/openworldtactile",
        "active-isaaclab-2.1.1/packages/assets/openworldtactile_assets/data/Props/tactile_test_shapes",
        "active-isaaclab-2.1.1/packages/assets/openworldtactile_assets/data/Policies",
    )
    for name in excluded_paths:
        if (ROOT / name).exists():
            failures.append(f"excluded release material was restored: {name}")

    expected_package_licenses = {
        "active-isaaclab-2.1.1/packages/core/setup.py": 'license="BSD-3-Clause"',
        "active-isaaclab-2.1.1/packages/tasks/setup.py": 'license="BSD-3-Clause"',
        "active-isaaclab-2.1.1/packages/assets/setup.py": "Multiple licenses; see THIRD_PARTY_NOTICES.md",
        "active-isaaclab-2.1.1/packages/uipc/setup.py": "Multiple licenses; see THIRD_PARTY_NOTICES.md",
    }
    for name, marker in expected_package_licenses.items():
        if marker not in (ROOT / name).read_text(encoding="utf-8"):
            failures.append(f"package license metadata mismatch: {name}")

    for path in ROOT.rglob("extension.toml"):
        text = path.read_text(encoding="utf-8")
        if "IsaacLabExtensionTemplate" in text or 'title = "Extension Template"' in text:
            failures.append(f"stale template metadata: {relative(path)}")

    scanned_files = 0
    for path in ROOT.rglob("*"):
        if not path.is_file() or path.suffix.casefold() not in TEXT_SUFFIXES or path.stat().st_size > 2_000_000:
            continue
        data = path.read_bytes()
        scanned_files += 1
        for label, pattern in SECRET_PATTERNS.items():
            if pattern.search(data):
                failures.append(f"possible {label}: {relative(path)}")

    if failures:
        print("open_source_audit=failed")
        for failure in failures:
            print(f"- {failure}")
        return 1

    print("open_source_audit=passed")
    print(f"required_policy_files={len(REQUIRED_FILES)}")
    print(f"required_license_files={len(REQUIRED_LICENSES)}")
    print(f"secret_scanned_text_files={scanned_files}")
    print("native_or_opaque_artifacts=0")
    print("generated_metadata_dirs=0")
    print("stale_template_metadata=0")
    return 0


if __name__ == "__main__":
    sys.exit(main())
