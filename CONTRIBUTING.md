# Contributing

Contributions are welcome when they keep the repository reproducible, attributable, and legally redistributable.

## Before opening a change

1. Discuss large behavioral or directory changes in an issue before implementation.
2. Work against the appropriate baseline: `active-isaaclab-2.1/` for the current line and `archive-isaaclab-2.3/` only for historical maintenance.
3. Do not commit credentials, personal data, generated package metadata, build products, model checkpoints, proprietary SDKs, or native binaries.
4. For copied or adapted work, retain upstream notices and add the exact source URL/revision, SPDX license identifier, modification note, and required citation. Update `THIRD_PARTY_NOTICES.md` and `CITATIONS.md` when applicable.
5. By submitting a contribution, you confirm that you have the right to provide it under the license stated for the affected files. Use `git commit -s` to add a Developer Certificate of Origin sign-off to each commit.

## Validation

Run the repository checks before submitting:

```bash
py tools/repository/audit_open_source.py
py tools/repository/build_static_navigation.py
py tools/repository/finalize_layout.py
```

The checks are static. If a change affects simulation, CUDA, UIPC, or hardware behavior, document the exact environment and the runtime command you verified. Do not report an unexecuted scenario as tested.

## Pull-request scope

Keep each pull request focused. Explain the research question or defect, list affected baselines, identify third-party material, describe validation, and call out compatibility or licensing changes explicitly.
