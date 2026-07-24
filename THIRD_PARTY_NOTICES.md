<p align="right">
  <strong>English</strong> · <a href="THIRD_PARTY_NOTICES.zh-CN.md">简体中文</a>
</p>

# Third-party notices

IsaacSim-Tactile4OpenWorld is a multi-license research distribution built around the OpenWorldTactile framework. The root [`LICENSE`](LICENSE) is BSD-3-Clause and covers original OpenWorldTactile contributions unless a file or subtree states otherwise. It does not replace an upstream notice, relicense third-party work, or grant trademark rights.

## Distributed components

| Component | Distributed path | Upstream | License and local notice | Notes |
|---|---|---|---|---|
| Isaac Lab and ORBIT-derived code | `active-isaaclab-2.1.1/packages/{core,assets,tasks}/`, selected experiment code, and `archive-isaaclab-2.3.2/packages/contrib/` | <https://github.com/isaac-sim/IsaacLab> | BSD-3-Clause; file SPDX headers and [`archive-isaaclab-2.3.2/LICENSE`](archive-isaaclab-2.3.2/LICENSE) | Existing Isaac Lab/ORBIT copyright lines must be retained. OpenWorldTactile modifications do not imply NVIDIA or Isaac Lab endorsement. |
| libuipc | `active-isaaclab-2.1.1/packages/uipc/libuipc/` | <https://github.com/spiriMirror/libuipc> | Apache-2.0; [`LICENSE`](active-isaaclab-2.1.1/packages/uipc/libuipc/LICENSE) | Bundled source reports version 0.9.0-alpha. Subcomponents below override the parent license for their own files. |
| libuipc Python bindings | `active-isaaclab-2.1.1/packages/uipc/libuipc/python/` | libuipc upstream | GPL-3.0-only; [`LICENSE`](active-isaaclab-2.1.1/packages/uipc/libuipc/python/LICENSE) | Do not describe this subtree as MIT or BSD. Distribution of a combined binary must satisfy the applicable GPL terms. |
| TetGen | `active-isaaclab-2.1.1/packages/uipc/libuipc/external/tetgen/` | <https://wias-berlin.de/software/tetgen/> | AGPL-3.0-or-later or a separate commercial license; [`LICENSE`](active-isaaclab-2.1.1/packages/uipc/libuipc/external/tetgen/LICENSE) | This is strong copyleft. A proprietary binary using TetGen requires separate rights from its copyright holder. |
| MuDA | `active-isaaclab-2.1.1/packages/uipc/libuipc/external/muda/` | <https://github.com/MuGdxy/muda> | Apache-2.0; [`LICENSE`](active-isaaclab-2.1.1/packages/uipc/libuipc/external/muda/LICENSE) | Catch2 within this subtree has its own license. |
| Catch2 2.13.10 | `active-isaaclab-2.1.1/packages/uipc/libuipc/external/muda/external/catch2/` | <https://github.com/catchorg/Catch2> | BSL-1.0; [`LICENSE_1_0.txt`](active-isaaclab-2.1.1/packages/uipc/libuipc/external/muda/external/catch2/LICENSE_1_0.txt) | Generated single-header distribution; retain its header notice. |
| Octree | `active-isaaclab-2.1.1/packages/uipc/libuipc/external/octree/Octree/` | <https://github.com/attcs/Octree> | MIT; [`LICENSE`](active-isaaclab-2.1.1/packages/uipc/libuipc/external/octree/Octree/LICENSE) | Copyright Attila Csikós/attcs. |
| SimpleBVH | `active-isaaclab-2.1.1/packages/uipc/libuipc/src/geometry/bvh/BVH.*` | <https://github.com/geometryprocessing/SimpleBVH> | MIT; [`LICENSE-SimpleBVH`](active-isaaclab-2.1.1/packages/uipc/libuipc/src/geometry/bvh/LICENSE-SimpleBVH) | Modified within libuipc; upstream revision checked at `69920e15db5281ce89d3618604ed63ea9e4e0bce`. |
| NSEssentials Morton code | `active-isaaclab-2.1.1/packages/uipc/libuipc/src/geometry/bvh/Morton.*` | <https://github.com/nschertler/NSEssentials> | BSD-3-Clause; [`LICENSE-NSEssentials`](active-isaaclab-2.1.1/packages/uipc/libuipc/src/geometry/bvh/LICENSE-NSEssentials) | Copyright Nico Schertler; upstream revision checked at `e5bceb01708368756528fdc95054bd3d05b7b745`. |
| AgileX Piper ROS | `active-isaaclab-2.1.1/vendor/agilex-piper/piper_ros/` | <https://github.com/agilexrobotics/piper_ros> | MIT; [`LICENSE`](active-isaaclab-2.1.1/vendor/agilex-piper/piper_ros/LICENSE) | Copyright RosenYin. |
| MoveIt 1.1.11 snapshot | `active-isaaclab-2.1.1/vendor/agilex-piper/piper_ros/src/piper_moveit/moveit-1.1.11/` | <https://github.com/moveit/moveit> | BSD-style terms; [`LICENSE.txt`](active-isaaclab-2.1.1/vendor/agilex-piper/piper_ros/src/piper_moveit/moveit-1.1.11/LICENSE.txt) | Preserve the many per-file and license-file copyright notices. |
| GPU Taxim/Taxim-derived implementation | `active-isaaclab-2.1.1/packages/core/openworldtactile/simulation_approaches/gpu_taxim/sim/` | <https://github.com/CMURoboTouch/Taxim> | MIT; [`LICENSE`](active-isaaclab-2.1.1/packages/core/openworldtactile/simulation_approaches/gpu_taxim/sim/LICENSE) | Copyright CMURoboTouch. Cite Taxim when this method is used. |
| FOTS-derived marker motion | `active-isaaclab-2.1.1/packages/core/openworldtactile/simulation_approaches/fots/sim/marker_motion.py` | <https://github.com/Rancho-zhao/FOTS> | MIT; [`LICENSE`](active-isaaclab-2.1.1/packages/core/openworldtactile/simulation_approaches/fots/LICENSE) | The file is modified for OpenWorldTactile. Upstream declares MIT in its README; the upstream `LICENSE` link was unavailable during the 2026-07-22 audit. The preserved local terms make the intended grant visible, but a release manager should recheck upstream history before a commercial redistribution. |
| ManiSkill-ViTac2025-derived tactile simulator | `active-isaaclab-2.1.1/packages/core/openworldtactile/simulation_approaches/fem_based/sim/tactile_sensor_sapienipc_modified.py` | <https://github.com/chuanyune/ManiSkill-ViTac2025> | Apache-2.0; [`LICENSE`](active-isaaclab-2.1.1/packages/core/openworldtactile/simulation_approaches/fem_based/sim/LICENSE) | The filename and header identify it as modified; retain the source permalink and modification notice. |
| GelSight Mini model and derived sensor assets | `active-isaaclab-2.1.1/packages/assets/openworldtactile_assets/data/Sensors/GelSight_Mini/` and GelSight-bearing Franka assemblies | <https://github.com/gelsightinc/gsrobotics> | GPL-3.0-only; local `LICENSE` files in both asset roots | The upstream repository supplies the GelSight Mini CAD model under GPL-3.0. Converted USD and combined asset forms are distributed conservatively under GPL-3.0-only. |
| Franka robot descriptions and meshes | GelSight-bearing Franka assemblies under `active-isaaclab-2.1.1/packages/assets/openworldtactile_assets/data/Robots/Franka/` | <https://github.com/frankarobotics/franka_ros> | Apache-2.0; [`LICENSE-APACHE-2.0`](active-isaaclab-2.1.1/packages/assets/openworldtactile_assets/data/Robots/Franka/LICENSE-APACHE-2.0) | Assemblies that also contain GelSight material must satisfy both applicable sets of terms. |

## Excluded from the public distribution

The 2026-07-22 release audit removed the following from this repository because redistribution rights or provenance were not sufficiently documented:

- the historical `hardware-sdk/openworldtactile/` bundle, including `SonixCamera.dll` and `libSonixCamera.so`;
- tactile test-shape assets sourced from `danfergo/gelsight-simulation`, whose upstream repository did not expose an explicit license during the audit;
- the opaque `Policies/ball_rolling/IK_old.pt` checkpoint, which had no author, training provenance, or model license;
- generated `*.egg-info` package metadata that incorrectly reported MIT.

These exclusions are deliberate release boundaries, not assertions that the original material infringes copyright. Restore an excluded component only with a verifiable license or written redistribution permission, its exact upstream revision, and the required notices.

## Research attribution

Licensing and scholarly citation are separate obligations. Papers associated with implemented methods are listed in [`CITATIONS.md`](CITATIONS.md); upstream projects may contain additional citation requests in their own READMEs.
