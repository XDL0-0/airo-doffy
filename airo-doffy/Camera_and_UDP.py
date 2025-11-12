import cv2
import time
import utils
import threading
from config import Config
import numpy as np
import UdpComms as U
import pyrealsense2 as rs
from typing import Dict, Tuple, List
from ur_teleop import URTeleop as urtp
from airo_camera_toolkit.cameras.realsense.realsense import Realsense
from parse_vr import parse_data

class CameraUDPManager:
    def __init__(self):
        cfg = Config()
        self.data = None
        self.fine_mode = None
        self.data_collecting_state = False
        self.data_export_state = False
        self.pc_ip = cfg.PC_IP
        self.vr_ip = cfg.VR_IP
        self.camera_images = {}
        self.camera_data = {}
        self.ip_port = cfg.IP_PORT
        self.initial_port = cfg.IP_PORT
        self.camera_num, self.camera_series_num = self._detect_cameras()
        self.camera_zoom: List[float] = []
        self.socket_list, self.camera_list = self._create_UDP_and_camera_lists()

    def _detect_cameras(self) -> Tuple[int, List[str]]:
        context = rs.context()
        devices = context.query_devices()
        camera_num = len(devices)
        camera_series_num = []

        if camera_num == 0:
            utils.logger.warning("No Realsense connected.")
        else:
            for i, device in enumerate(devices):
                name = device.get_info(rs.camera_info.name)
                serial = device.get_info(rs.camera_info.serial_number)
                utils.logger.info(f"camera {i}: {name}, serial={serial}")
                camera_series_num.append(serial)

        return camera_num, camera_series_num

    def _alloc_socket(self, idx: int, enable_rx: bool, socket_list: Dict[str, U.UdpComms]) -> None:
        name = f"socket_{idx}"
        sock = U.UdpComms(
            udpIP=self.pc_ip,
            sendIP=self.vr_ip,
            portTX=self.ip_port,
            portRX=self.ip_port + 1,
            enableRX=enable_rx,
            suppressWarnings=True,
        )
        socket_list[name] = sock
        utils.logger.info(f"{name}: TX={self.ip_port}, RX={self.ip_port + 1}, enableRX={enable_rx}")
        self.ip_port += 2

    def _create_camera(self, idx: int, camera_list: Dict[str, Realsense]) -> None:
        """创建 Realsense 相机对象"""
        name = f"camera_{idx}"
        try:
            serial = self.camera_series_num[idx]
        except IndexError:
            utils.logger.warning(f"{name}: No serial number at index {idx}, skip.")
            return

        cam = Realsense(
            fps=60,
            resolution=Realsense.RESOLUTION_480,
            enable_depth=False,
            enable_hole_filling=False,
            serial_number=serial,
        )
        camera_list[name] = cam
        utils.logger.info(f"{name}: serial={serial}, fps=60, res=480, depth=OFF")

    def _create_UDP_and_camera_lists(
        self,
    ) -> Tuple[Dict[str, U.UdpComms], Dict[str, Realsense]]:
        socket_list: Dict[str, U.UdpComms] = {}
        camera_list: Dict[str, Realsense] = {}

        utils.logger.info(f"PC IP: {self.pc_ip}, VR IP: {self.vr_ip}, base_port={self.ip_port}")

        n = max(0, self.camera_num)

        for i in range(n):
            enable_rx = i < 2
            self._alloc_socket(i, enable_rx, socket_list)
            self._create_camera(i, camera_list)

        min_sockets = 3
        total_sockets_needed = max(min_sockets, n)
        for i in range(n, total_sockets_needed):
            self._alloc_socket(i, True, socket_list)

        utils.logger.info(f"{len(camera_list)} cameras, {len(socket_list)} UDP sockets created.")
        self.camera_zoom = [1.0] * n

        return socket_list, camera_list

    def send_and_receive_data(self, socket_list, camera_list):
        for i in range(self.camera_num):
            image = camera_list[f'camera_{i}'].get_rgb_image()
            data, resized_rgb = self.data_process(image,i)
            socket_list[f'socket_{i}'].SendData(data)
        self.data = parse_data(self._read_socket(socket_list, 0))


    def get_camera_thread(self, camera, n):
        print(f"rx camera Thread {n} starts!")
        while True:
            self.camera_data[f'camera_{n}'] = camera.get_rgb_image()
            time.sleep(1/30)


    def send_camera_thread(self, socket, n):
        print(f"tx camera Thread {n} starts!")
        while True:
            # start_time_ns = time.time_ns()
            # start_time = time.time()
            # print(self.data_collecting_state)
            data, frame_rgb = self.data_process(self.camera_data[f'camera_{n}'],n)
            self.camera_images[f'camera_{n}'] = frame_rgb
            socket.SendData(data)
            # print(self.data)
            # interval_ns = time.time_ns()-start_time_ns
            # interval = time.time()-start_time
            # print(f"tx camera {n} interval:{interval}")
            # time.sleep(1/30-interval_ns*0.000000001)
            time.sleep(1/30)

    def center_zoom(self, image, scale: float = 1.5, interpolation=cv2.INTER_LINEAR):
        h, w = image.shape[:2]
        new_w, new_h = int(w * scale), int(h * scale)

        if new_w <= w or new_h <= h:
            return cv2.resize(image, (w, h), interpolation=interpolation)

        resized = cv2.resize(image, (new_w, new_h), interpolation=interpolation)

        start_x = max((new_w - w) // 2, 0)
        start_y = max((new_h - h) // 2, 0)
        cropped = resized[start_y:start_y + h, start_x:start_x + w]

        return cropped

    def data_process(self, frame, cam_idx: int):
        if frame.dtype != np.uint8:
            frame = (frame * 255).astype(np.uint8)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        zoom_factor = self.camera_zoom[cam_idx] if cam_idx < len(self.camera_zoom) else 1.0
        frame_rgb = self.center_zoom(frame_rgb, zoom_factor)

        if cam_idx == 1:
            frame = self.center_zoom(frame, 1.5)

        _, encoded_frame = cv2.imencode('.jpg', frame_rgb, [cv2.IMWRITE_JPEG_QUALITY, 50])
        data = encoded_frame.tobytes()

        return data, frame_rgb

    def resolution_data_process(self, s: str):
        s = s.strip(";")
        if not s:
            return

        for item in s.split(";"):
            if not item:
                continue
            key, value = map(str.strip, item.split(",", 1))

            if key.isdigit():
                cam_idx = (int(key) % self.initial_port) // 2
                if cam_idx < self.camera_num:
                    zoom_val = float(value[1:])
                    if not np.isclose(self.camera_zoom[cam_idx], zoom_val):
                        utils.logger.info(f"camera{cam_idx}: zoom {self.camera_zoom[cam_idx]} → {zoom_val}")
                        self.camera_zoom[cam_idx] = zoom_val
                else:
                    utils.logger.warning(f"Invalid camera index {cam_idx}, total cameras={self.camera_num}")

            else:
                self.fine_mode = value

    def _read_socket(self, socket_list, idx: int):
        return socket_list[f'socket_{idx}'].ReadReceivedData()

    def is_movement_exist(self):
        return (
                (bool(self.data[1]['GripTrigger']) or bool(self.data[1]['Joystick'][1]))
        )

    def receive_data_thread(self, socket_list):
        utils.logger.info("Receive VR Thread starts! (default rate:100Hz)")
        target_dt = 0.01  # 100Hz

        while True:
            start_time = time.time()
            try:
                raw_data = self._read_socket(socket_list, 0)
                if raw_data is not None:
                    self.data = parse_data(raw_data)

                record_control = self._read_socket(socket_list, 1)
                resolution_control = self._read_socket(socket_list, 2)

                if resolution_control:
                    self.resolution_data_process(resolution_control)

                if record_control:
                    utils.logger.debug(f"Record control received: {record_control}")
                    if record_control == "Start":
                        self.data_collecting_state = True
                        self.data_export_state = False
                    elif record_control == "Stop":
                        self.data_collecting_state = False
                        self.data_export_state = True

            except Exception as e:
                utils.logger.error(f"Error in receive_data_thread: {e}")
                time.sleep(0.1)

            elapsed = time.time() - start_time
            if elapsed < target_dt:
                time.sleep(target_dt - elapsed)

    def test_connection(self):
        while True:
            self.send_and_receive_data(self.socket_list,self.camera_list)

            if self.data is not None:  # if NEW data has been received since last ReadReceivedData function call
                return self.data
            else:
                utils.logger.info("Connecting VR.....")
                continue

    def start_comms_threads(self):
        for i in range(self.camera_num):
            thread_rxcamera = threading.Thread(target=self.get_camera_thread, args=(self.camera_list[f'camera_{i}'],i), daemon=True)
            thread_rxcamera.start()
            # rxthreads.append(thread_rxcamera)
            utils.logger.info(f"{i}  rx camera thread create!")

        time.sleep(0.4)
        utils.logger.info(self.camera_data.keys())

        # txthreads = []
        for i in range(self.camera_num):
            thread_txcamera = threading.Thread(target=self.send_camera_thread, args=(self.socket_list[f'socket_{i}'],i),
                                               daemon=True)
            thread_txcamera.start()
            # txthreads.append(thread_txcamera)
            utils.logger.info(f"{i} tx camera thread create!")

        thread_rxdata=threading.Thread(target=self.receive_data_thread, args=(self.socket_list,), daemon=True)
        thread_rxdata.start()
        time.sleep(0.1)
        utils.logger.info("rx VR data thread start")



