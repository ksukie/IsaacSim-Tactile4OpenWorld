"""AgileX Piper robot configuration for Isaac Lab.

This config expects a Piper USD converted from AgileX's URDF at:
    openworldtactile_assets/data/Robots/AgileX/Piper/piper.usd

The joint/link names below follow the common AgileX Piper URDF naming scheme.
Verify them with Isaac Sim's Robot Inspector after importing the USD.
"""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

from openworldtactile_assets import OWT_ASSETS_DATA_DIR


PIPER_USD_PATH = f"{OWT_ASSETS_DATA_DIR}/Robots/AgileX/Piper/piper.usd"


AGILEX_PIPER_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=PIPER_USD_PATH,
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "joint1": 0.0,
            "joint2": 0.0,
            "joint3": 0.0,
            "joint4": 0.0,
            "joint5": 0.0,
            "joint6": 0.0,
            "joint7": 0.03,
            "joint8": -0.03,
        },
    ),
    actuators={
        "piper_arm": ImplicitActuatorCfg(
            joint_names_expr=["joint[1-6]"],
            effort_limit_sim=80.0,
            velocity_limit_sim=2.5,
            stiffness=80.0,
            damping=4.0,
        ),
        "piper_gripper": ImplicitActuatorCfg(
            joint_names_expr=["joint[7-8]"],
            effort_limit_sim=50.0,
            velocity_limit_sim=0.2,
            stiffness=1000.0,
            damping=100.0,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)


AGILEX_PIPER_HIGH_PD_CFG = AGILEX_PIPER_CFG.copy()
"""AgileX Piper configuration with stiffer PD control for differential IK demos."""

AGILEX_PIPER_HIGH_PD_CFG.actuators["piper_arm"].stiffness = 400.0
AGILEX_PIPER_HIGH_PD_CFG.actuators["piper_arm"].damping = 80.0
