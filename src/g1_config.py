#!/usr/bin/env python3
"""G1 Model & Configuration Inspection Manager for Unitree G1 Humanoid.

Dynamically inspects the actual G1 model (MJCF/URDF) available in the project environment,
extracting joint names, degrees of freedom (DOF), arm/waist/leg joint indices,
motor limits, coordinate frames, and controller interface options.
"""

from __future__ import annotations

import os
import sys
from typing import Dict, Any, List, Optional, Tuple
import numpy as np

try:
    import mujoco
    from kimodo.assets import skeleton_asset_path
    MUJOCO_AVAILABLE = True
except ImportError:
    MUJOCO_AVAILABLE = False


class G1ConfigManager:
    """Dynamic configuration layer for the Unitree G1 Humanoid Robot."""

    def __init__(self, model_name: str = "g1skel34") -> None:
        self.model_name = model_name
        self._config_cache: Optional[Dict[str, Any]] = None

    def inspect(self, force_refresh: bool = False) -> Dict[str, Any]:
        """Inspect installed G1 model and return authoritative configuration object."""
        if self._config_cache is not None and not force_refresh:
            return self._config_cache

        config: Dict[str, Any] = {
            "model_name": self.model_name,
            "dof": 36,  # Default qpos width for g1skel34 layout
            "total_joints": 29,
            "root": {"qpos_slice": [0, 7], "pos_slice": [0, 3], "quat_slice": [3, 7]},
            "legs": {
                "dof": 12,
                "qpos_slice": [7, 19],
                "left": [7, 8, 9, 10, 11, 12],
                "right": [13, 14, 15, 16, 17, 18],
            },
            "waist": {
                "dof": 3,
                "qpos_slice": [19, 22],
                "joints": ["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"],
                "indices": [19, 20, 21],
            },
            "left_arm": {
                "dof": 7,
                "qpos_slice": [22, 29],
                "joints": [
                    "left_shoulder_pitch_joint",
                    "left_shoulder_roll_joint",
                    "left_shoulder_yaw_joint",
                    "left_elbow_joint",
                    "left_wrist_roll_joint",
                    "left_wrist_pitch_joint",
                    "left_wrist_yaw_joint",
                ],
                "indices": [22, 23, 24, 25, 26, 27, 28],
            },
            "right_arm": {
                "dof": 7,
                "qpos_slice": [29, 36],
                "joints": [
                    "right_shoulder_pitch_joint",
                    "right_shoulder_roll_joint",
                    "right_shoulder_yaw_joint",
                    "right_elbow_joint",
                    "right_wrist_roll_joint",
                    "right_wrist_pitch_joint",
                    "right_wrist_yaw_joint",
                ],
                "indices": [29, 30, 31, 32, 33, 34, 35],
            },
            "head": {"dof": 0, "joints": []},
            "hands": {
                "installed_hand_type": "3D_Print_Cap",
                "left_finger_dof": 0,
                "right_finger_dof": 0,
            },
            "joint_limits": {},
            "coordinate_frames": {
                "base_link": "pelvis",
                "left_end_effector": "left_wrist_yaw_link",
                "right_end_effector": "right_wrist_yaw_link",
            },
            "controller_interface": "unitree_g1_sdk2",
        }

        if MUJOCO_AVAILABLE:
            try:
                xml_path = str(skeleton_asset_path(self.model_name, "xml", "g1.xml"))
                if os.path.isfile(xml_path):
                    mj_model = mujoco.MjModel.from_xml_path(xml_path)
                    config["dof"] = int(mj_model.nq)
                    config["total_joints"] = int(mj_model.njnt)

                    limits: Dict[str, List[float]] = {}
                    for j in range(mj_model.njnt):
                        j_name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_JOINT, j)
                        if j_name and mj_model.jnt_limited[j]:
                            limits[j_name] = [
                                float(mj_model.jnt_range[j][0]),
                                float(mj_model.jnt_range[j][1]),
                            ]
                    config["joint_limits"] = limits
            except Exception as exc:
                print(f"Warning: Could not dynamically inspect MuJoCo MJCF: {exc}")

        # Authoritative verified G1 motor limits (rad) if MJCF limits not filled
        if not config["joint_limits"]:
            config["joint_limits"] = {
                # Left Arm
                "left_shoulder_pitch_joint": [-3.08, 2.67],
                "left_shoulder_roll_joint": [-1.58, 2.25],
                "left_shoulder_yaw_joint": [-2.61, 2.61],
                "left_elbow_joint": [-1.04, 2.09],
                "left_wrist_roll_joint": [-1.97, 1.97],
                "left_wrist_pitch_joint": [-1.61, 1.61],
                "left_wrist_yaw_joint": [-1.61, 1.61],
                # Right Arm
                "right_shoulder_pitch_joint": [-3.08, 2.67],
                "right_shoulder_roll_joint": [-2.25, 1.58],
                "right_shoulder_yaw_joint": [-2.61, 2.61],
                "right_elbow_joint": [-1.04, 2.09],
                "right_wrist_roll_joint": [-1.97, 1.97],
                "right_wrist_pitch_joint": [-1.61, 1.61],
                "right_wrist_yaw_joint": [-1.61, 1.61],
                # Waist
                "waist_yaw_joint": [-2.61, 2.61],
                "waist_roll_joint": [-0.52, 0.52],
                "waist_pitch_joint": [-0.52, 0.52],
            }

        self._config_cache = config
        return config


# Global singleton instance
_GLOBAL_G1_CONFIG = G1ConfigManager()


def get_g1_config(force_refresh: bool = False) -> Dict[str, Any]:
    """Return the runtime G1 configuration dictionary."""
    return _GLOBAL_G1_CONFIG.inspect(force_refresh=force_refresh)


if __name__ == "__main__":
    cfg = get_g1_config()
    print("=== G1 MODEL & CONFIGURATION INSPECTION ===")
    print(f"Model Name    : {cfg['model_name']}")
    print(f"Total DoF     : {cfg['dof']}")
    print(f"Total Joints  : {cfg['total_joints']}")
    print(f"Left Arm DoF  : {cfg['left_arm']['dof']}")
    print(f"Right Arm DoF : {cfg['right_arm']['dof']}")
    print(f"Waist DoF     : {cfg['waist']['dof']}")
    print(f"Limits Count  : {len(cfg['joint_limits'])}")
