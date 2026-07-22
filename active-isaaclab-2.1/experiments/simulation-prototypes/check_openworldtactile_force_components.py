from __future__ import annotations

from pathlib import Path
import traceback

import cv2
import numpy as np

from check_taxim_sim import BallRollingEnv, BallRollingEnvCfg, args_cli, simulation_app

import carb
import omni.ui


INITIAL_FORCE_FIELD_PATH = Path(__file__).with_name("openworldtactile_initial_force_field.npy")
X_FORCE_FIELD_DIR = Path(__file__).with_name("openworldtactile_x_force_frames")
Y_FORCE_FIELD_DIR = Path(__file__).with_name("openworldtactile_y_force_frames")


class OpenWorldTactileForceComponentsWindow:
    """Single debug window for OpenWorldTactile sensor force-field components."""

    def __init__(self, width: int, height: int):
        self.window = omni.ui.Window("/OpenWorldTactile/openworldtactile_force_components", width=width, height=height)
        self.provider = omni.ui.ByteImageProvider()

    def update(self, frame_rgb: np.ndarray):
        frame_rgba = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2RGBA)
        height, width, _ = frame_rgba.shape
        with self.window.frame:
            self.provider.set_bytes_data(frame_rgba.flatten().data, [width, height])
            omni.ui.ImageWithProvider(self.provider)


def _draw_label(frame: np.ndarray, label: str) -> np.ndarray:
    frame = frame.copy()
    cv2.putText(frame, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(frame, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return frame


def _draw_axis_component_arrows(component: np.ndarray, axis: str) -> np.ndarray:
    component = np.nan_to_num(component.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    height, width = component.shape
    frame = np.zeros((height, width, 3), dtype=np.uint8)

    max_abs = float(np.percentile(np.abs(component), 99.0))
    if max_abs <= 1.0e-6:
        max_abs = float(np.max(np.abs(component)))

    step = max(8, min(height, width) // 24)
    for y in range(step // 2, height, step):
        for x in range(step // 2, width, step):
            cv2.circle(frame, (x, y), 1, (0, 220, 120), -1)

    if max_abs <= 1.0e-6:
        return frame

    arrow_scale = 0.7 * step / max_abs
    threshold = max_abs * 0.05
    for y in range(step // 2, height, step):
        for x in range(step // 2, width, step):
            if abs(component[y, x]) <= threshold:
                continue

            if axis == "x":
                end_x = int(np.clip(x + component[y, x] * arrow_scale, 0, width - 1))
                end_y = y
            elif axis == "y":
                end_x = x
                end_y = int(np.clip(y + component[y, x] * arrow_scale, 0, height - 1))
            else:
                raise ValueError(f"Unsupported axis: {axis}")

            cv2.arrowedLine(frame, (x, y), (end_x, end_y), (255, 255, 255), 1, tipLength=0.25)

    return frame


def _draw_positive_component(component: np.ndarray) -> np.ndarray:
    component = np.nan_to_num(component.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    component = np.clip(component, 0.0, None)
    max_value = float(np.percentile(component, 99.0))
    if max_value <= 1.0e-6:
        max_value = float(np.max(component))

    if max_value <= 1.0e-6:
        return np.zeros((*component.shape, 3), dtype=np.uint8)

    normalized = np.clip(component / max_value, 0.0, 1.0)
    heat = (normalized * 255.0).astype(np.uint8)
    return cv2.cvtColor(cv2.applyColorMap(heat, cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)


def _resize_to(frame: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    target_height, target_width = target_shape
    if frame.shape[:2] == target_shape:
        return frame
    return cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_LINEAR)


def _compose_openworldtactile_force_components(env: BallRollingEnv, env_id: int = 0) -> np.ndarray:
    force_field = env.gsmini._data.output["tactile_force_field"][env_id].detach().cpu().numpy()

    current_frame = env.gsmini._draw_openworldtactile_sensor_force_field(force_field)
    target_shape = current_frame.shape[:2]

    fx_frame = _resize_to(_draw_axis_component_arrows(force_field[..., 0], "x"), target_shape)
    fy_frame = _resize_to(_draw_axis_component_arrows(force_field[..., 1], "y"), target_shape)
    fz_frame = _resize_to(_draw_positive_component(force_field[..., 2]), target_shape)

    current_frame = _draw_label(current_frame, "OpenWorldTactile force field")
    fx_frame = _draw_label(fx_frame, "Fx")
    fy_frame = _draw_label(fy_frame, "Fy")
    fz_frame = _draw_label(fz_frame, "Fz / magnitude")

    height, _, _ = current_frame.shape
    column_separator = np.full((height, 2, 3), 32, dtype=np.uint8)
    top_row = np.concatenate((current_frame, column_separator, fx_frame), axis=1)
    bottom_row = np.concatenate((fy_frame, column_separator, fz_frame), axis=1)
    row_separator = np.full((2, top_row.shape[1], 3), 32, dtype=np.uint8)
    return np.concatenate((top_row, row_separator, bottom_row), axis=0)


def _save_initial_tactile_force_field(env: BallRollingEnv, output_path: Path = INITIAL_FORCE_FIELD_PATH) -> None:
    force_field = env.gsmini._data.output["tactile_force_field"][0].detach().cpu().numpy().copy()
    np.save(output_path, force_field)
    print(f"[INFO]: Saved initial tactile_force_field to {output_path} with shape {force_field.shape}")


def _save_tactile_force_xy_frame(
    env: BallRollingEnv,
    frame_index: int,
    x_output_dir: Path = X_FORCE_FIELD_DIR,
    y_output_dir: Path = Y_FORCE_FIELD_DIR,
) -> None:
    force_field = env.gsmini._data.output["tactile_force_field"][0].detach().cpu().numpy()
    x_force_field = force_field[..., 0].copy()
    y_force_field = force_field[..., 1].copy()

    x_output_dir.mkdir(parents=True, exist_ok=True)
    y_output_dir.mkdir(parents=True, exist_ok=True)
    np.save(x_output_dir / f"x_force_field_{frame_index:06d}.npy", x_force_field)
    np.save(y_output_dir / f"y_force_field_{frame_index:06d}.npy", y_force_field)

    if frame_index == 0:
        print(
            "[INFO]: Saving tactile_force_field x/y frames to "
            f"{x_output_dir} and {y_output_dir} with shape {x_force_field.shape}"
        )


def run_simulator(env: BallRollingEnv):
    window = None
    saved_initial_force_field = False
    saved_xy_frame_count = 0
    current_num_resets = 0
    reset_interval = len(env.pattern_offsets) * env.num_step_goal_change
    print("Number of steps till reset: ", reset_interval)

    while simulation_app.is_running():
        if env.step_count % reset_interval == 0:
            print(f"[INFO]: Env reset num {current_num_resets}")
            env.reset()
            current_num_resets += 1

        env._pre_physics_step(None)
        env._apply_action()
        env.scene.write_data_to_sim()
        env.sim.step(render=False)
        env.sim.render()
        env.scene.update(dt=env.physics_dt)

        env.gsmini.update(dt=env.physics_dt, force_recompute=True)
        env._compute_sdf_tactile_force_field()

        if not saved_initial_force_field:
            _save_initial_tactile_force_field(env)
            saved_initial_force_field = True

        _save_tactile_force_xy_frame(env, saved_xy_frame_count)
        saved_xy_frame_count += 1

        frame = _compose_openworldtactile_force_components(env)
        if window is None:
            window = OpenWorldTactileForceComponentsWindow(width=frame.shape[1], height=frame.shape[0])
        window.update(frame)

    if env.openworldtactile_bridge is not None:
        env.openworldtactile_bridge.close()
    env.close()


def main():
    env_cfg = BallRollingEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    env_cfg.gsmini.debug_vis = False
    env_cfg.debug_vis = False

    experiment = BallRollingEnv(env_cfg)
    print("[INFO]: Setup complete...")
    run_simulator(env=experiment)


if __name__ == "__main__":
    try:
        main()
    except Exception as err:
        carb.log_error(err)
        carb.log_error(traceback.format_exc())
        raise
