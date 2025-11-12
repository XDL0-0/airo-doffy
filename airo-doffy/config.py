import numpy as np
class Config:
    ROBOT_TYPE = "ur3e"
    UR_IP = "10.42.0.162"
    PC_IP = "10.10.131.215"
    VR_IP = "10.10.130.7"
    DATASET_DIR = "./datasets/test"
    TASK_NAME = ""
    DATASET_TYPE = 'a'  #ACT(hdf5)->'a' or lerobot->'l'
    SAVE_EEF = False
    DATA_TYPE = 'qpos'
    IP_PORT = 8000
    UR_CTRL_RATE = 100
    KELO_CTRL_RATE = 10
    COLLECT_RATE = 10
    GRIPPER_STEP = 0.02
    GRIPPER_MAX = 0.085
    INITIAL_JOINT = np.array([-1.57, -1.57, -1.57, -3.14, -1.57, 4.71])
    # INITIAL_JOINT = np.array([-1.57, -1.57, -1.57, 0, 1.57, 1.57])
    TCP_TRANSFORM = np.identity(4)
    MOVE_THRESHOLD = 1
