<p align="right">
  <strong>English</strong> · <a href="../../zh-CN/help/faq.md">简体中文</a>
</p>

# Frequently asked questions

## Is this a standalone Isaac Sim or Isaac Lab distribution?

No. The project installs extensions into an external Isaac Lab 2.1.1 / Isaac Sim 4.5 environment. Follow [Installation](../getting-started/installation.md) first.

## Which script should I run first?

Run the reduced V1 case in [Quick start](../getting-started/quick-start.md). It verifies a real UIPC membrane solve and produces a small, inspectable output without requiring the V6.2 contract chain.

## What does “open world” mean here?

It means extending tactile experiments across robots, sensor models, objects, and contact conditions. It does not claim a general-purpose world model, open-vocabulary reasoning, or zero-shot control in arbitrary scenes.

## Why are there both 2.1.1 and 2.3.2 directories?

They are separate Isaac Lab baselines. `active-isaaclab-2.1.1/` is the maintained OpenWorldTactile/UIPC mainline. `archive-isaaclab-2.3.2/` preserves an older GelSight/SDK route. They must not be mixed in one environment.

## Are V1, V5, and V6.2 software releases?

No. They are tactile-bench research stages. A larger number does not automatically mean a more stable or easier script. See [Experiment lineage](../reference/experiment-lineage.md).

## Why does V6.2 need earlier V5.7d/e/f outputs?

V6.2 uses a frozen, validated deformation contract and estimator. The prerequisite runs verify frame removal, membrane topology/area, normal response, and repeatability. The script intentionally refuses an incomplete or failed contract.

## Are the tactile forces in Newtons?

Not by default. V1 outputs `sim_constitutive_force`; the frozen estimator uses TU-valued quantities. V6.2 separately records physical UIPC reaction/applied coupling arrays in N or N·m where its metadata says so. Never infer units only from a filename; read metadata.

## Can I use the project without compiling UIPC?

Some optical/marker code and non-UIPC assets can be installed with `./run.sh --install`, but the documented V1/V6.2 workflows and UIPC task variant require `./run.sh --install all`. The tasks extension also declares an integration dependency on UIPC.

## Is Windows supported?

The upstream Isaac products and libuipc have Windows support paths, but this repository's end-to-end installation is documented only for Linux/Bash. A tested PowerShell wrapper/toolchain matrix has not been published.

## Does the repository include pretrained policies?

It includes agent configurations, not a complete, validated policy set. An opaque historical checkpoint was deliberately excluded because its training provenance and license were not documented. Train using the configured task/backend or supply a compatible, properly licensed checkpoint.

## Does it include the physical camera SDK?

No. Historical vendor binaries were removed because redistribution rights could not be verified. Obtain the SDK from an authorized source and configure `OWT_SDK_ROOT` only for affected legacy work.

## Can I connect real robot or camera hardware?

Some archived code references hardware/SDK workflows, but the project is not safety certified. Review and test in isolation, add physical limits and emergency-stop procedures outside this codebase, and never interpret static checks as hardware validation.

## Where are results written?

Each experiment has its own default, often under `/tmp`. Set `--output_dir` explicitly and keep `--workspace_dir` separate. See [Data and outputs](../guides/data-and-outputs.md).

## Why is my task ID unknown to Isaac Lab's trainer?

The external launcher must import `openworldtactile_tasks` after starting Isaac Sim. Unmodified upstream launchers import only upstream task registrations. See [Connect an Isaac Lab launcher](../guides/tasks-and-training.md#connect-an-isaac-lab-launcher).

## Which documentation is authoritative?

The maintained user guides are under `docs/en/` and `docs/zh-CN/`. Source code and a script's current `--help` are authoritative when they disagree with prose. `docs/internal/` and notes beside experiments are provenance/maintenance records, not setup guides.

## Can I use this project commercially?

The original project code is BSD-3-Clause, but the repository is multi-license. GPL/AGPL and other terms apply to specific bindings, assets, and third-party subtrees. Review [Third-party notices](../../../THIRD_PARTY_NOTICES.md) and obtain legal advice for your distribution; the documentation is not legal advice.

## How should I cite it?

Use [`CITATION.cff`](../../../CITATION.cff) for project metadata and [Research citations](../../../CITATIONS.md) for methods used in your work. Cite both the project and the original papers for the actual simulation approaches you use.

## How do I report a useful issue?

Follow the diagnostic template at the end of [Troubleshooting](troubleshooting.md#report-a-reproducible-issue). Include exact versions, command, traceback, and the smallest reproducing case; do not attach credentials or proprietary SDK files.
