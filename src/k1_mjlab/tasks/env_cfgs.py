"""Booster K1 velocity tracking environment configurations."""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg

from k1_mjlab.robot.k1_constants import (
    K1_ROBOT_CFG,
)


def booster_k1_rough_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Create Booster K1 rough terrain velocity configuration."""
    cfg = make_velocity_env_cfg()

    cfg.scene.entities = {"robot": K1_ROBOT_CFG}

    # K1 is a humanoid with 2 feet.
    site_names = ("left_foot", "right_foot")
    geom_names = ("left_foot_collision", "right_foot_collision")
    target_foot_height = 0.15

    feet_ground_cfg = ContactSensorCfg(
        name="feet_ground_contact",
        primary=ContactMatch(
            mode="subtree",
            pattern=r"^(left_foot_link|right_foot_link)$",
            entity="robot",
        ),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="netforce",
        num_slots=1,
        track_air_time=True,
    )
    self_collision_cfg = ContactSensorCfg(
        name="self_collision",
        primary=ContactMatch(mode="subtree", pattern="Trunk", entity="robot"),
        secondary=ContactMatch(mode="subtree", pattern="Trunk", entity="robot"),
        fields=("found",),
        reduce="none",
        num_slots=1,
    )
    cfg.scene.sensors = (feet_ground_cfg, self_collision_cfg)

    if cfg.scene.terrain is not None and cfg.scene.terrain.terrain_generator is not None:
        cfg.scene.terrain.terrain_generator.curriculum = True

    cfg.viewer.body_name = "Trunk"
    cfg.viewer.distance = 1.5
    cfg.viewer.elevation = -10.0
    cfg.commands["twist"].viz.z_offset = 1.15

    cfg.observations["critic"].terms["foot_height"].params["asset_cfg"].site_names = site_names

    cfg.events["foot_friction"].params["asset_cfg"].geom_names = geom_names

    # Tight control when stationary: maintain stable default pose.
    cfg.rewards["pose"].params["std_standing"] = {".*": 0.05}
    # Moderate leg freedom for stepping, loose arms for natural pendulum swing.
    cfg.rewards["pose"].params["std_walking"] = {
        # Head
        r".*Head.*": 0.1,
        # Lower body.
        r".*Hip_Pitch.*": 0.3,
        r".*Hip_Roll.*": 0.15,
        r".*Hip_Yaw.*": 0.15,
        r".*Knee.*": 0.35,
        r".*Ankle_Pitch.*": 0.25,
        r".*Ankle_Roll.*": 0.1,
        # Arms.
        r".*Shoulder_Pitch.*": 0.15,
        r".*Shoulder_Roll.*": 0.15,
        r".*Elbow.*": 0.15,
    }
    # Maximum freedom for dynamic motion.
    cfg.rewards["pose"].params["std_running"] = {
        # Head
        r".*Head.*": 0.1,
        # Lower body.
        r".*Hip_Pitch.*": 0.5,
        r".*Hip_Roll.*": 0.2,
        r".*Hip_Yaw.*": 0.2,
        r".*Knee.*": 0.6,
        r".*Ankle_Pitch.*": 0.35,
        r".*Ankle_Roll.*": 0.15,
        # Arms.
        r".*Shoulder_Pitch.*": 0.5,
        r".*Shoulder_Roll.*": 0.2,
        r".*Elbow.*": 0.35,
    }

    cfg.rewards["upright"].params["asset_cfg"].body_names = ("Trunk",)
    cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("Trunk",)

    for reward_name in ["foot_clearance", "foot_swing_height", "foot_slip"]:
        cfg.rewards[reward_name].params["asset_cfg"].site_names = site_names
    cfg.rewards["foot_swing_height"].params["target_height"] = target_foot_height
    cfg.rewards["foot_clearance"].params["target_height"] = target_foot_height

    # Disable illegal contact termination (handled by self-collision sensor).
    cfg.terminations.pop("illegal_contact", None)

    # Apply play mode overrides.
    if play:
        # Effectively infinite episode length.
        cfg.episode_length_s = int(1e9)

        cfg.observations["policy"].enable_corruption = False
        cfg.events.pop("push_robot", None)

        if cfg.scene.terrain is not None:
            if cfg.scene.terrain.terrain_generator is not None:
                cfg.scene.terrain.terrain_generator.curriculum = False
                cfg.scene.terrain.terrain_generator.num_cols = 5
                cfg.scene.terrain.terrain_generator.num_rows = 5
                cfg.scene.terrain.terrain_generator.border_width = 10.0

    return cfg


def booster_k1_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Create Booster K1 flat terrain velocity configuration."""
    cfg = booster_k1_rough_env_cfg(play=play)

    # Switch to flat terrain.
    assert cfg.scene.terrain is not None
    cfg.scene.terrain.terrain_type = "plane"
    cfg.scene.terrain.terrain_generator = None

    # Disable terrain curriculum.
    assert cfg.curriculum is not None
    del cfg.curriculum["terrain_levels"]

    return cfg
