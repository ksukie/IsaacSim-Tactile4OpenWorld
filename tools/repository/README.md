<p align="right">
  <strong>English</strong> · <a href="README.zh-CN.md">简体中文</a>
</p>

# Repository validation tools

This directory contains static release inventories, reports, and validation scripts. It is maintainer tooling, not part of either Isaac Lab runtime payload.

## Generated records

| File | Purpose |
|---|---|
| `FINAL_MANIFEST.csv` | Path, size, and SHA-256 for every file in the two versioned runtime payloads |
| `PORTABILITY_ADAPTATIONS.csv` | Configurable SDK, Python, and external-asset entry points |
| `ENTRYPOINT_INVENTORY.csv` | Static Python entry-point, import, and asset-reference inventory |
| `PATH_ADAPTATION_PLAN.csv` | Classification and disposition of external/runtime path references |
| `external_path_references.csv` | External path scan results |
| `usda_reference_check.csv` | USDA relative and external reference checks |
| `markdown_link_check.csv` | Local-link checks for maintained top-level documentation |

## Commands

Run from the repository root with a standard Python 3 interpreter:

```bash
python tools/repository/audit_open_source.py
python tools/repository/build_static_navigation.py
python tools/repository/finalize_layout.py
```

These checks do not import project packages, install dependencies, compile UIPC, or launch Isaac Sim.

Only rewrite the payload manifest after confirming that every runtime-payload change is intentional:

```bash
python tools/repository/finalize_layout.py --write-manifest
```

`build_static_navigation.py` refreshes the generated inventory and [`docs/internal/ENTRYPOINT_MATRIX.md`](../../docs/internal/ENTRYPOINT_MATRIX.md). `finalize_layout.py` verifies manifest hashes plus AST syntax, documentation links, USDA references, path conflicts, compressed files, and legacy-name fragments. `audit_open_source.py` checks release policies, licenses, package metadata, credential-like content, generated directories, and prohibited native/opaque payloads.
