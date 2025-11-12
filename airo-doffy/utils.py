# utils.py
import logging
import numpy as np
from scipy.spatial.transform import Rotation as R

logging.basicConfig(
    level=logging.INFO,   # default level>INFO
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)

logger = logging.getLogger("Teleop")


def quat_cal(matrix, last_quat):
    rot = R.from_matrix(matrix)
    quat = rot.as_quat()  # [x, y, z, w]

    if last_quat is not None and np.dot(quat, last_quat) < 0:
        quat = -quat
    return quat

def is_joint_change_safe(previous_joints, new_joints, joint_threshold):
    if previous_joints is None:
        return True
    joint_diff = np.abs(np.array(new_joints) - np.array(previous_joints))
    if np.any(joint_diff > joint_threshold):
        print(f"⚠️ joint change too large: {joint_diff}，keep pose")
        return False
    return True

