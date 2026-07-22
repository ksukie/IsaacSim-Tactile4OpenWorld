# 中文说明：以 cube 为接触物运行原始 GelSight/OpenWorldTactile 对照脚本，方便和 OpenWorldTactile RGB 输出做同场景比较。

"""Run the native OpenWorldTactile/GelSight demo with a cube contact object.

This script intentionally bypasses the OpenWorldTactile RGB renderer so the saved
``tactile_rgb_image`` frames show the original OpenWorldTactile/GelSight rendering.
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
ORIGINAL_SCRIPT = SCRIPT_DIR / "openworldtactile_gelsight_original_sensor.py"


def main() -> None:
    args = sys.argv[1:]

    if "--contact_object_type" not in args:
        args.extend(["--contact_object_type", "cube"])

    if "--save_viz_dir" not in args:
        args.extend(["--save_viz_dir", "tactile_record_gelsight_cube"])

    sys.argv = [str(ORIGINAL_SCRIPT), *args]
    runpy.run_path(str(ORIGINAL_SCRIPT), run_name="__main__")


if __name__ == "__main__":
    main()
