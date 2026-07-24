<p align="right">
  <strong>English</strong> · <a href="SECURITY.zh-CN.md">简体中文</a>
</p>

# Security policy

## Supported code

Security fixes target the latest state of `active-isaaclab-2.1.1/`. The `archive-isaaclab-2.3.2/` line is retained for research reproducibility and does not receive routine security maintenance.

## Reporting a vulnerability

Do not open a public issue for a suspected vulnerability, exposed credential, unsafe native dependency, or hardware-control risk. After the repository is published, use the hosting platform's private vulnerability-reporting channel or contact the maintainers privately through the repository owner profile. Include affected paths and versions, reproduction steps, impact, and any suggested mitigation.

Maintainers should acknowledge a complete report within seven days, coordinate disclosure, and publish remediation notes when a fix is available. This is a best-effort research project, not a safety-certified robotics product.

## Operational safety

Simulation scripts may control GPU workloads, robots, cameras, or other hardware through external dependencies. Review commands and limits in an isolated environment before connecting physical equipment. Never treat the archived code or a static-check result as evidence of hardware safety.
