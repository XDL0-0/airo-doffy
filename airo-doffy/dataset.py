from config import Config
from lerodata_collect import LeRobotDatasetRecorder
from pathlib import Path
import numpy as np
import os
import h5py
import time
import utils

class DatasetRecorder:
    def __init__(
            self,
            camera_num
    ):
        cfg = Config()
        self.save_eef = cfg.SAVE_EEF

        self.collect_step = 0
        self.camera_num = camera_num
        self.dataset_type = cfg.DATASET_TYPE
        self.data_type = cfg.DATA_TYPE
        self.dataset_dir = cfg.DATASET_DIR

        utils.logger.info(f'Dataset Dir:{self.dataset_dir}')
        utils.logger.info(f'Dataset Type: {self.dataset_type}')
        utils.logger.info(f'Data Type: {self.data_type}')

        self.data_dict_init()
        self._dataset_init()

    def data_dict_init(self):
        if self.save_eef:
            self.data_dict_eef = {
                '/observations/qpos': [],
                '/action': [],
            }
            for i in range(self.camera_num):
                self.data_dict_eef[f'/observations/images/camera_{i}'] = []
        else:
            self.data_dict = {
                '/observations/qpos': [],
                '/action': [],
            }
            for i in range(self.camera_num):
                self.data_dict[f'/observations/images/camera_{i}'] = []
        self.camera_images = {}
        self.collect_step = 0

    def _dataset_init(self):
        if self.dataset_type == 'a':
            if not os.path.exists(self.dataset_dir):
                os.makedirs(self.dataset_dir)

            existing_episodes = 0
            for file in os.listdir(self.dataset_dir):
                if file.startswith('episode_') and file.endswith('.hdf5'):
                    try:
                        episode_num = int(file.split('_')[1].split('.')[0])
                        existing_episodes = max(existing_episodes, episode_num + 1)
                    except ValueError:
                        continue
            if existing_episodes != 0:
                utils.logger.warning(f'Dataset already exists. Recording data from{existing_episodes}')
            self.recorded_episodes = existing_episodes

        elif self.dataset_type == 'l':
            example_obs = {
                "robot_pose": np.zeros((6,), dtype=np.float32),
                "gripper_state": np.zeros((1,), dtype=np.float32),
                "camera_0": np.zeros((480, 640, 3), dtype=np.float32),
                "camera_1": np.zeros((480, 640, 3), dtype=np.float32),
            }
            example_action = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7], dtype=np.float32)
            self.dataset_recorder = LeRobotDatasetRecorder(
                example_obs_dict=example_obs,
                example_action=example_action,
                root_dataset_dir=Path("datasets_lero"),
                dataset_name="test_dataset",
                fps=30,
                use_videos=True,
            )
    def data_collection(self,state,previous_solution,camera_images):
                self.collect_step += 1
                self.data_dict['/observations/qpos'].append(state)
                self.data_dict['/action'].append(previous_solution)
                for name in camera_images:
                    self.data_dict[f'/observations/images/{name}'].append(camera_images[name])


    def data_collection_EEF(self):
        while True:
            if self.data_collecting_state:
                start_time = time.time()
                self.collect_step += 1
                state = np.concatenate([self.ur_eef_capture, self.gripper_capture],axis=0)

                self.data_dict_eef['/observations/qpos'].append(state)
                self.data_dict_eef['/action'].append(self.previous_solution_eef)
                # Capture images from cameras
                for name in self.camera_images:
                    self.data_dict_eef[f'/observations/images/{name}'].append(self.camera_images[name])
                interval = time.time()-start_time
                # print(f"collecting data interval:{interval}")
                time.sleep(1/10-interval)
            else:
                time.sleep(1/10)

    def data_export(self, camera_images):
        t0 = time.time()
        if self.save_eef:
            max_timesteps = len(self.data_dict_eef['/observations/qpos'])
        else:
            max_timesteps = len(self.data_dict['/observations/qpos'])
        # print("qpos:", len(self.data_dict['/observations/qpos']), "action", len(self.data_dict['/action']))
        utils.logger.info(f"max_timesteps:{max_timesteps}")
        utils.logger.info(f'collect episodes:{self.collect_step}')
        if self.dataset_type == 'l':
            for name in camera_images:
                self.data_dict[f'/observations/images/{name}'] = np.array(self.data_dict[f'/observations/images/{name}']) / 255.0

            for i in range(max_timesteps):

                example_obs = {
                    "robot_pose": np.array(self.data_dict['/observations/qpos'][i][:6], dtype=np.float32),
                    "gripper_state": np.array([self.data_dict['/observations/qpos'][i][6]], dtype=np.float32),
                }
                for name in camera_images:
                    example_obs[f'{name}'] = np.array(self.data_dict[f'/observations/images/{name}'][i], dtype=np.float32)

                self.dataset_recorder.record_step(example_obs, np.array(self.data_dict['/action'][i], dtype=np.float32),task_name=self.task_name)
            self.dataset_recorder.save_episode()

        elif self.dataset_type == 'a':

            #
            if self.save_eef:
                dataset_path_eef = os.path.join(f'{self.dataset_dir}', f'episode_{self.recorded_episodes}')
                with h5py.File(dataset_path_eef + '.hdf5', 'w', rdcc_nbytes=1024 ** 2 * 2) as root:
                    root.attrs['sim'] = False
                    obs = root.create_group('observations')
                    image = obs.create_group('images')
                    qpos = obs.create_dataset('qpos', (max_timesteps, 8))
                    action = root.create_dataset('action', (max_timesteps, 8))
                    for name in camera_images:
                        # _ = image.create_dataset(name, (max_timesteps, 640, 480, 3), dtype='uint8',
                        #                          chunks=(1, 640, 480, 3), )
                        _ = image.create_dataset(name, (max_timesteps, 480, 640, 3), dtype='uint8',
                                                 chunks=(1, 480, 640, 3), )

                    for name, array in self.data_dict_eef.items():
                        root[name][...] = array[:max_timesteps]

                description_file_path = os.path.join(f'{self.dataset_dir}', 'episode_descriptions.txt')
                with open(description_file_path, 'a') as f:
                    f.write(f"Episode {self.recorded_episodes}: max_timesteps = {max_timesteps}\n")

            else:
                dataset_path = os.path.join(self.dataset_dir, f'episode_{self.recorded_episodes}')
                with h5py.File(dataset_path + '.hdf5', 'w', rdcc_nbytes=1024 ** 2 * 2) as root:
                    root.attrs['sim'] = False
                    obs = root.create_group('observations')
                    image = obs.create_group('images')
                    qpos = obs.create_dataset('qpos', (max_timesteps, 7))
                    action = root.create_dataset('action', (max_timesteps, 7))
                    for name in camera_images:

                        _ = image.create_dataset(name, (max_timesteps, 480, 640, 3), dtype='uint8',
                                                 chunks=(1, 480, 640, 3), )

                    for name, array in self.data_dict.items():
                        # print(name)
                        root[name][...] = array[:max_timesteps]

                description_file_path = os.path.join(self.dataset_dir, 'episode_descriptions.txt')
                with open(description_file_path, 'a') as f:
                    f.write(f"Episode {self.recorded_episodes}: max_timesteps = {max_timesteps}\n")
            self.recorded_episodes += 1

        utils.logger.info(f'\n\n\n\n\n\n\n\n\n\nSaving: {time.time() - t0:.1f} secs\n')

