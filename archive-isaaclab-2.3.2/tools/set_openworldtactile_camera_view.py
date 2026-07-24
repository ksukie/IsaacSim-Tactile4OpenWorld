# 中文说明：快速修改 OpenWorldTactile 触觉 demo 里的主视角相机位置和观察目标，便于调试仿真画面。

from pathlib import Path
import argparse
import re

parser = argparse.ArgumentParser()
parser.add_argument("--script", default="experiments/sensors/openworldtactile_finger_sensor.py")
parser.add_argument("--eye", nargs=3, type=float, default=[0.25, 0.35, 0.75])
parser.add_argument("--target", nargs=3, type=float, default=[0.0, 0.06, 0.50])
args = parser.parse_args()

p = Path(args.script)
text = p.read_text()

replacement = (
    f"    sim.set_camera_view("
    f"eye={[round(x, 6) for x in args.eye]}, "
    f"target={[round(x, 6) for x in args.target]})"
)

pattern = r"    sim\.set_camera_view\(eye=\[[^\]]+\], target=\[[^\]]+\]\)"

text_new, count = re.subn(pattern, replacement, text, count=1)

if count != 1:
    raise SystemExit("Could not find exactly one sim.set_camera_view(...) line")

p.write_text(text_new)
print(f"Updated {p}")
print(replacement)
