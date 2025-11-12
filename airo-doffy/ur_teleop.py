import sys
import time
import cv2
import numpy as np

import utils
from config import Config
from airo_robots.grippers import Robotiq2F85
from airo_spatial_algebra.se3 import SE3Container
from airo_robots.manipulators.hardware.ur_rtde import URrtde

def make_robot(ur_ip, robot_type):
    if robot_type == "ur3e":
        from ur_analytic_ik import ur3e as ik
        return URrtde(ur_ip, URrtde.UR3E_CONFIG), ik
    elif robot_type == "ur5e":
        from ur_analytic_ik import ur5e as ik
        return URrtde(ur_ip, URrtde.UR5E_CONFIG), ik
    else:
        raise ValueError("Unsupported robot type")

class URTeleop:
    def __init__(
        self,
        initial_data
    ):
        cfg = Config()
        self.ur, self.ik = make_robot(cfg.UR_IP, cfg.ROBOT_TYPE)
        gripper = Robotiq2F85("10.42.0.162")
        self.ur.gripper = gripper
        self.control_rate = cfg.UR_CTRL_RATE
        self.gripper_delta_step_size = cfg.GRIPPER_STEP
        self.initial_joint = cfg.INITIAL_JOINT
        self.save_eef = cfg.SAVE_EEF
        self.tcp_transform = cfg.TCP_TRANSFORM
        self.joint_threshold = cfg.MOVE_THRESHOLD
        self.fine_mode = False
        self.last_quat = None
        self.reset_sign = False
        utils.logger.info(f"Teleop initialized with UR:{cfg.UR_IP}, VR:{cfg.VR_IP}")

        utils.logger.info(f"Moving to initial joint:{self.initial_joint}")
        self.ur.move_to_joint_configuration(self.initial_joint,0.5).wait()
        self.ur.gripper.open().wait()
        self.gripper_solution_width = self.ur.gripper.get_current_width()
        self.previous_solution = np.append(self.initial_joint, 0)

        self._set_reference(initial_data)
        self.previous_solution = np.append(self.ur.get_joint_configuration(),
                                           (initial_data[1]['Button_BY'] - initial_data[1]["Button_AX"]))
        if self.save_eef:
            self.last_quat = utils.quat_cal(matrix=self.SE3_tcp_pose_in_base_frame_std.rotation_matrix,last_quat=self.last_quat)
            self.ur_eef_capture = np.concatenate(
                [self.last_quat, self.SE3_tcp_pose_in_base_frame_std.translation], axis=0)
            self.previous_solution_eef = np.append(self.ur_eef_capture,
                                                   (initial_data[1]['Button_BY'] - initial_data[1]["Button_AX"]))
            self.state_eef = np.concatenate([self.last_quat, self.SE3_tcp_pose_in_base_frame_std.translation,self.gripper_solution_width],axis=0)
        else:
            self.state = np.concatenate([self.previous_solution[:6],self.SE3_tcp_pose_in_base_frame_std.translation,np.array([self.gripper_solution_width])],axis=0)

    def _set_reference(self,data):
        self.SE3_controller_std = self.extract_SE3_from_data(data)
        self.SE3_tcp_pose_in_base_frame_std = SE3Container.from_homogeneous_matrix(self.ur.get_tcp_pose())

    def extract_SE3_from_data(self, controller_data):
        rotation_rh = np.array([controller_data[1]['Rotation'][0], controller_data[1]['Rotation'][2],
                                -controller_data[1]['Rotation'][1], controller_data[1]['Rotation'][3]])
        rotation_rh = np.round(rotation_rh, 4)
        position_rh = np.array([-controller_data[1]['Position'][0], -controller_data[1]['Position'][2],
                                controller_data[1]['Position'][1]])
        SE3_controller = SE3Container.from_quaternion_and_translation(rotation_rh,
                                                                      position_rh)
        return SE3_controller

    def FineModeManager(self, controller_data, mode_status):
        if mode_status == "ON" and not self.fine_mode:
            utils.logger.info("Fine Control Mode: ON")
            self.fine_mode = True
            self._set_reference(controller_data)

        elif mode_status == "OFF" and self.fine_mode:
            utils.logger.info("Fine Control Mode: OFF")
            self.fine_mode = False
            self._set_reference(controller_data)

    def capture_joint_pose(self):
        return np.array(self.ur.get_joint_configuration())

    def capture_eef_pose(self, last_quat):
        eef = SE3Container.from_homogeneous_matrix(self.ur.get_tcp_pose())
        self.last_quat = utils.quat_cal(eef.rotation_matrix, last_quat)
        return np.concatenate([self.last_quat, eef.translation], axis=0)

    def capture_gripper(self):
        return np.array([self.ur.gripper.get_current_width()])

    def update_gripper(self, gripper_state):
        if self.ur.gripper.is_an_object_grasped() and gripper_state < 0:
            gripper_state = 0

        if gripper_state:
            if abs(self.gripper_solution_width - self.capture_gripper()) < self.gripper_delta_step_size:
                self.gripper_solution_width += self.gripper_delta_step_size * gripper_state

            self.gripper_solution_width = np.clip(self.gripper_solution_width, 0.0, 0.085)
            self.ur.gripper.move(self.gripper_solution_width, 0.1)

            self.previous_solution = np.concatenate([self.previous_solution[:6], [gripper_state]])
            if self.save_eef:
                self.previous_solution_eef = np.concatenate([self.previous_solution_eef[:7], [gripper_state]])

    def reset_robot_and_gripper(self):
        gripper_to_default = self.ur.gripper.move(0.085, 0.85)
        robot_to_default = self.ur.move_to_joint_configuration(self.initial_joint, 1)

        while not robot_to_default.is_action_done():
            gripper_condition = int(not gripper_to_default.is_action_done())
            gripper_capture = self.capture_gripper()

            if self.save_eef:
                eef_pose = self.capture_eef_pose(self.last_quat)
                self.previous_solution_eef = np.concatenate([eef_pose, [gripper_condition]])
                self.state_eef = np.concatenate([self.previous_solution_eef[:7], gripper_capture], axis=0)

            else:
                self.previous_solution = np.concatenate([self.capture_joint_pose(), [gripper_condition]])
                self.state = np.concatenate([self.previous_solution[:6], gripper_capture],axis=0)

            time.sleep(1 / self.control_rate)

        # reset done
        utils.logger.info("Gripper and robot in default POSE!")

        self.previous_solution = np.concatenate([self.initial_joint, [0]])
        self.gripper_solution_width = self.ur.gripper.get_current_width()
        self.SE3_tcp_pose_in_base_frame_std = SE3Container.from_homogeneous_matrix(self.ur.get_tcp_pose())

        if self.save_eef:
            self.last_quat = utils.quat_cal(self.SE3_tcp_pose_in_base_frame_std.rotation_matrix)
            self.ur_eef_capture = np.concatenate(
                [self.last_quat, self.SE3_tcp_pose_in_base_frame_std.translation], axis=0
            )
            self.previous_solution_eef = np.concatenate([self.ur_eef_capture, [0]])

        utils.logger.info("---- Reset complete ----")

    def standby_mode(self, controller_data):
        if not controller_data[1]['GripTrigger']:
            # standby
            self._set_reference(controller_data)
            utils.logger.debug("Standby mode active, updated SE3 reference.")
            return True
        return False

    def teleop_mode(self, controller_data, gripper_state):
        if self.reset_sign:
            self.reset_sign = False
            self._set_reference(controller_data)

        SE3_controller = self.extract_SE3_from_data(controller_data)
        se3_controller = SE3_controller.homogeneous_matrix

        translation_diff = se3_controller[:3, 3] - self.SE3_controller_std.translation
        rotation_diff = np.dot(self.SE3_controller_std.rotation_matrix.T, se3_controller[:3, :3])

        # fine control or not
        if self.fine_mode:
            alpha_t, alpha_r, beta = 0.2, 0.5, 0.5
        else:
            alpha_t, alpha_r, beta = 1.0, 1.0, 0.0

        rvec, _ = cv2.Rodrigues(rotation_diff)
        rvec *= alpha_r
        rotation_diff, _ = cv2.Rodrigues(rvec)
        translation_diff *= alpha_t

        target_translation = self.SE3_tcp_pose_in_base_frame_std.translation + translation_diff
        target_rotation = rotation_diff @ self.SE3_tcp_pose_in_base_frame_std.rotation_matrix
        tcp_target = SE3Container.from_rotation_matrix_and_translation(target_rotation, target_translation)

        joint_solution = self.ik.inverse_kinematics_closest_with_tcp(
            tcp_target.homogeneous_matrix, self.tcp_transform, *self.ur.get_joint_configuration()
        )

        if joint_solution and self.ur._is_joint_confjoint_solutioniguration_reachable(joint_solution[0]):
            if utils.is_joint_change_safe(self.previous_solution[:6], joint_solution[0],self.joint_threshold):
                joint_solution[0] = beta * self.state[:6] + (1 - beta) * joint_solution[0]
                self.ur.servo_to_joint_configuration(joint_solution[0], 1 / self.control_rate)
                self.previous_solution = np.concatenate([joint_solution[0], [gripper_state]])

                if self.save_eef:
                    tcp = self.ik.forward_kinematics(*joint_solution[0])
                    self.last_quat = utils.quat_cal(tcp[:3, :3])
                    solution_eef = np.concatenate([self.last_quat, tcp[:3, 3]], axis=0)
                    self.previous_solution_eef = np.concatenate([solution_eef, [gripper_state]])

                utils.logger.debug("Teleop step executed successfully.")
            else:
                utils.logger.warning("Joint change unsafe, keeping previous pose!")
        else:
            utils.logger.warning("No valid IK solution, keeping previous pose!")

    def step(self, controller_data, fine_mode_status):
        # always capture state
        self.FineModeManager(controller_data,fine_mode_status)
        ur_pose_capture = self.capture_joint_pose()
        if self.save_eef:
            self.ur_eef_capture = self.capture_eef_pose(self.last_quat)
        gripper_capture = self.capture_gripper()

        # joystick -> gripper_state
        x = -controller_data[1]['Joystick'][1]
        gripper_state = (x > 0) - (x < 0)

        # update gripper
        self.update_gripper(gripper_state)

        # reset?
        if controller_data[1]['Joystick_Press']:
            self.reset_sign = True
            self.reset_robot_and_gripper()
            return

        # standby or teleop
        if not self.standby_mode(controller_data):
            self.state = np.concatenate([ur_pose_capture, gripper_capture], axis=0)
            self.teleop_mode(controller_data, gripper_state)

    def run(self, dataset):
        #No use
        utils.logger.info("Teleop started...")
        while True:
            obs = self.step()



