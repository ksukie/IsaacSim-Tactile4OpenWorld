<p align="right">
  <strong>English</strong> · <a href="README.zh-CN.md">简体中文</a>
</p>

# OpenWorldTactile documentation

This is the user documentation for IsaacSim-Tactile4OpenWorld. The English and Simplified Chinese trees have the same structure and content.

## Start here

1. [Installation](en/getting-started/installation.md) — prepare Isaac Sim, Isaac Lab, build tools, and project packages.
2. [Quick start](en/getting-started/quick-start.md) — verify imports, run lightweight tests, and generate the first tactile result.
3. Choose a workflow:
   - [Experiments](en/guides/experiments.md)
   - [Tasks and training](en/guides/tasks-and-training.md)
   - [Data and outputs](en/guides/data-and-outputs.md)
   - [Custom integration](en/guides/custom-integration.md)

## Reference

| Document | Use it when you need to… |
|---|---|
| [Architecture](en/reference/architecture.md) | understand packages, data flow, and active/archive boundaries |
| [Compatibility](en/reference/compatibility.md) | choose versions and understand what has been validated |
| [Configuration](en/reference/configuration.md) | look up environment variables, wrapper options, and common flags |
| [Experiment lineage](en/reference/experiment-lineage.md) | distinguish stable starting points from historical research stages |
| [Repository layout](en/reference/repository-layout.md) | find source, assets, tasks, tools, and internal records |

## Help and project information

- [Troubleshooting](en/help/troubleshooting.md)
- [FAQ](en/help/faq.md)
- [Legacy 2.3.2 route](en/guides/legacy.md)
- [Contributing](../CONTRIBUTING.md)
- [Security policy](../SECURITY.md)
- [Citation guide](../CITATIONS.md)
- [Third-party notices](../THIRD_PARTY_NOTICES.md)

## Documentation scope

`docs/en/` and `docs/zh-CN/` are the maintained user guides. [`docs/internal/`](internal/) contains generated inventories and release-maintenance records; they are not setup instructions. Research-stage notes beside individual experiments are retained for provenance and may describe obsolete or unverified workflows.

When documentation and source disagree, treat the source and `--help` output as authoritative, then report the mismatch.
