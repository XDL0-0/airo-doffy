from ur_teleop import URTeleop
from dataset import DatasetRecorder
from Camera_and_UDP import CameraUDPManager as CU
from config import Config as cfg
import threading
import time
import utils

def control_loop(teleop, cu_manager):
    while True:
        teleop.step(cu_manager.data, cu_manager.fine_mode)
        time.sleep(0.01)

def collect_loop(teleop, cu_manager, dataset, collect_rate):
    while True:
        start_time = time.time()
        if cu_manager.data_collecting_state and (cu_manager.is_movement_exist() or teleop.reset_sign):
            dataset.data_collection(teleop.state, teleop.previous_solution, cu_manager.camera_images)
        interval = time.time() - start_time
        time.sleep(1/collect_rate - interval)


def main():
    cu_manager = CU()
    dataset = DatasetRecorder(cu_manager.camera_num)
    teleop = URTeleop(cu_manager.test_connection())
    cu_manager.start_comms_threads()

    t1 = threading.Thread(target=control_loop, args=(teleop, cu_manager), daemon=True)
    t2 = threading.Thread(target=collect_loop, args=(teleop, cu_manager, dataset, cfg.COLLECT_RATE), daemon=True)

    t1.start()
    t2.start()

    while True:
        if cu_manager.data_export_state:
            if not dataset.collect_step:
                utils.logger.error(f'no data to export')
            else:
                dataset.data_export(cu_manager.camera_images)
                dataset.data_dict_init()
                utils.logger.info(f'data: episode_{dataset.recorded_episodes-1} exported successfully:)')
                cu_manager.data_export_state = False
        time.sleep(0.5)

if __name__ == "__main__":
    main()
