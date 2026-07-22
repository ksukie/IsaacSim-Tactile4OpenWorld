import argparse
import dataclasses

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

from isaaclab_assets.sensors import GELSIGHT_R15_CFG

print("=== GELSIGHT_R15_CFG ===")
print(GELSIGHT_R15_CFG)

print("\n=== possible light/color fields ===")

def walk(obj, path="GELSIGHT_R15_CFG", depth=0):
    if depth > 5:
        return

    name = path.lower()
    if any(k in name for k in ["light", "color", "rgb", "illum", "diffuse", "specular"]):
        print(path, "=", repr(obj))

    if dataclasses.is_dataclass(obj):
        for f in dataclasses.fields(obj):
            try:
                walk(getattr(obj, f.name), f"{path}.{f.name}", depth + 1)
            except Exception:
                pass
    elif isinstance(obj, dict):
        for k, v in obj.items():
            walk(v, f"{path}.{k}", depth + 1)
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            walk(v, f"{path}[{i}]", depth + 1)
    elif hasattr(obj, "__dict__"):
        for k, v in vars(obj).items():
            walk(v, f"{path}.{k}", depth + 1)

walk(GELSIGHT_R15_CFG)

simulation_app.close()
