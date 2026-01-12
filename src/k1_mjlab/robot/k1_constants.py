"""Booster K1 constants."""

from pathlib import Path

import mujoco
from mjlab.actuator import XmlPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.os import update_assets
from mjlab.utils.spec_config import CollisionCfg

##
# MJCF and assets.
##

K1_XML: Path = Path(__file__).parent / "xmls" / "k1.xml"

assert K1_XML.exists(), f"K1 robot XML not found at {K1_XML}"


def get_assets(meshdir: str) -> dict[str, bytes]:
    assets: dict[str, bytes] = {}
    update_assets(assets, K1_XML.parent / "assets", meshdir)
    return assets


def get_spec() -> mujoco.MjSpec:
    spec = mujoco.MjSpec.from_file(str(K1_XML))
    spec.assets = get_assets(spec.meshdir)
    return spec


##
# Keyframe config.
##

INIT_STATE = EntityCfg.InitialStateCfg(
    pos=(0, 0, 0.513),
    joint_pos={
        "Left_Shoulder_Roll": -1.4,
        "Left_Elbow_Yaw": -0.4,
        "Right_Shoulder_Roll": 1.4,
        "Right_Elbow_Yaw": 0.4,
        "Left_Hip_Pitch": -0.2,
        "Left_Knee_Pitch": 0.4,
        "Left_Ankle_Pitch": -0.2,
        "Right_Hip_Pitch": -0.2,
        "Right_Knee_Pitch": 0.4,
        "Right_Ankle_Pitch": -0.2,
    },
    joint_vel={".*": 0.0},
)

##
# Collision config.
##

_foot_regex = r"^(left|right)_foot_collision$"


# This enables all collisions, including self collisions.
# Self-collisions are given condim=1 while foot collisions are given condim=3.
FULL_COLLISION = CollisionCfg(
    geom_names_expr=[".*_collision"],
    condim={".*_collision": 3},
    priority={_foot_regex: 1},
    friction={_foot_regex: (0.6,)},
)

FULL_COLLISION_WITHOUT_SELF = CollisionCfg(
    geom_names_expr=[".*_collision"],
    contype=0,
    conaffinity=0,
    condim={".*_collision": 3},
    priority={_foot_regex: 1},
    friction={_foot_regex: (0.6,)},
)

# This disables all collisions except the feet.
# Feet get condim=3, all other geoms are disabled.
FEET_ONLY_COLLISION = CollisionCfg(
    geom_names_expr=[_foot_regex],
    contype=0,
    conaffinity=0,
    condim=3,
    priority=1,
    friction=(0.6,),
)

##
# Articulation config.
##

K1_ARTICULATION = EntityArticulationInfoCfg(
    actuators=(
        XmlPositionActuatorCfg(target_names_expr=("Head_Yaw",)),
        XmlPositionActuatorCfg(target_names_expr=("Head_Pitch",)),
        XmlPositionActuatorCfg(target_names_expr=("Left_Shoulder_Pitch",)),
        XmlPositionActuatorCfg(target_names_expr=("Left_Shoulder_Roll",)),
        XmlPositionActuatorCfg(target_names_expr=("Left_Elbow_Pitch",)),
        XmlPositionActuatorCfg(target_names_expr=("Left_Elbow_Yaw",)),
        XmlPositionActuatorCfg(target_names_expr=("Right_Shoulder_Pitch",)),
        XmlPositionActuatorCfg(target_names_expr=("Right_Shoulder_Roll",)),
        XmlPositionActuatorCfg(target_names_expr=("Right_Elbow_Pitch",)),
        XmlPositionActuatorCfg(target_names_expr=("Right_Elbow_Yaw",)),
        XmlPositionActuatorCfg(target_names_expr=("Left_Hip_Pitch",)),
        XmlPositionActuatorCfg(target_names_expr=("Left_Hip_Roll",)),
        XmlPositionActuatorCfg(target_names_expr=("Left_Hip_Yaw",)),
        XmlPositionActuatorCfg(target_names_expr=("Left_Knee_Pitch",)),
        XmlPositionActuatorCfg(target_names_expr=("Left_Ankle_Pitch",)),
        XmlPositionActuatorCfg(target_names_expr=("Left_Ankle_Roll",)),
        XmlPositionActuatorCfg(target_names_expr=("Right_Hip_Pitch",)),
        XmlPositionActuatorCfg(target_names_expr=("Right_Hip_Roll",)),
        XmlPositionActuatorCfg(target_names_expr=("Right_Hip_Yaw",)),
        XmlPositionActuatorCfg(target_names_expr=("Right_Knee_Pitch",)),
        XmlPositionActuatorCfg(target_names_expr=("Right_Ankle_Pitch",)),
        XmlPositionActuatorCfg(target_names_expr=("Right_Ankle_Roll",)),
    ),
)

##
# Final config.
##


def get_k1_robot_cfg() -> EntityCfg:
    """Get a fresh K1 robot configuration instance.

    Returns a new EntityCfg instance each time to avoid mutation issues when
    the config is shared across multiple places.
    """
    return EntityCfg(
        init_state=INIT_STATE,
        collisions=(FULL_COLLISION,),
        spec_fn=get_spec,
        articulation=K1_ARTICULATION,
    )


if __name__ == "__main__":
    import mujoco.viewer as viewer
    from mjlab.entity.entity import Entity

    robot = Entity(get_k1_robot_cfg())

    viewer.launch(robot.spec.compile())
