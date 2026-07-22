"""Public package exports for OpenWorldTactile UIPC.

The Isaac Sim environment classes are loaded lazily so importing ``openworldtactile_uipc``
for installation checks does not require an already-started Isaac Sim app.
"""

_EXPORT_MODULES = {
    "MeshGenerator": "openworldtactile_uipc.utils",
    "TetMeshCfg": "openworldtactile_uipc.utils",
    "TriMeshCfg": "openworldtactile_uipc.utils",
    "UipcInteractiveScene": "openworldtactile_uipc.envs",
    "UipcRLEnv": "openworldtactile_uipc.envs",
    "UipcIsaacAttachments": "openworldtactile_uipc.sim",
    "UipcIsaacAttachmentsCfg": "openworldtactile_uipc.sim",
    "UipcSim": "openworldtactile_uipc.sim",
    "UipcSimCfg": "openworldtactile_uipc.sim",
    "UipcObject": "openworldtactile_uipc.objects",
    "UipcObjectCfg": "openworldtactile_uipc.objects",
    "UipcObjectDeformableData": "openworldtactile_uipc.objects",
    "UipcObjectRigidData": "openworldtactile_uipc.objects",
}

__all__ = sorted(_EXPORT_MODULES)


def __getattr__(name: str):
    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    from importlib import import_module

    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value
