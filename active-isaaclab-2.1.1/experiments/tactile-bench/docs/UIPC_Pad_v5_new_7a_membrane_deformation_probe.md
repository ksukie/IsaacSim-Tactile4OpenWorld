# V5 new 7a membrane deformation probe

## 目标

`OpenWorldTactile_v5_new_7a_membrane_deformation_probe.py` 只验证下面这条链路：

```text
UIPC_Pad.usda
  simulation/membrane_sim_mesh
    -> UIPC deformable object
    -> UIPC solver advance
    -> membrane.data.surf_nodal_pos_w
    -> rest/current/deformation dump
```

本阶段不创建：

- cylinder
- grasp motion
- force estimator
- pressure reconstruction
- contact geometry proxy

## 启动

```bash
cd "${OWT_ROOT}"

./run.sh -p experiments/tactile-bench/OpenWorldTactile_v5_new_7a_membrane_deformation_probe.py
```

有界面查看，直到手动关闭：

```bash
./run.sh -p experiments/tactile-bench/OpenWorldTactile_v5_new_7a_membrane_deformation_probe.py \
  --render_viewport \
  --run_steps 0 \
  --render_every 1 \
  --render_sleep_sec 0.01 \
  --output_dir /tmp/openworldtactile_uipc_v5_new_7a_gui
```

headless 快速检查：

```bash
./run.sh -p experiments/tactile-bench/OpenWorldTactile_v5_new_7a_membrane_deformation_probe.py \
  --headless \
  --run_steps 100 \
  --output_dir /tmp/openworldtactile_uipc_v5_new_7a_membrane_deformation_probe
```

## 输出

```text
initial_membrane_rest_vertices.npy
initial_membrane_current_vertices.npy
initial_membrane_deformation.npy
latest_membrane_rest_vertices.npy
latest_membrane_current_vertices.npy
latest_membrane_deformation.npy
membrane_rest_vertices.npy
membrane_current_vertices.npy
membrane_deformation.npy
membrane_deformation_max_mm_per_step.npy
metadata.json
status.json
```

`initial_*` 在 UIPC 初始化后立刻写出；`latest_*` 默认每 50 步更新一次。这样即使窗口被手动关闭或进程提前退出，也能检查最近一次 deformation。

metadata 关键字段：

```json
{
  "uipc_solver_used": true,
  "uipc_initialized": true,
  "deformation_source": "simulation/membrane_sim_mesh",
  "force_source": "none",
  "pressure_source": "none",
  "attachment_used": false
}
```

## 验收

因为 7a 没有压头、没有抓取、没有 attachment，且 UIPC gravity 为 0：

```text
initial_deformation_mm ~= 0
final_deformation_max_mm ~= 0
```

如果 deformation 明显非零，先检查：

- UIPC object 是否真的绑定到 `/World/envs/env_0/Robot/link8/UIPC_Pad/simulation`
- deformation source 是否是 `simulation/membrane_sim_mesh`
- link8 是否在 UIPC 初始化后还发生运动
- UIPC solver 是否在无接触情况下引入数值漂移

## 下一步

7a 通过后再进入 Step 3：

```text
v5_new_7b_membrane_contact_probe
```

目标先验证 `simulation/membrane_sim_mesh` 是否作为不可穿透 deformable contact layer 参与 UIPC 接触，而不是直接进入 attachment、force estimator 或抓取。
