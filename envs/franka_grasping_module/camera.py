from pathlib import Path

from isaacgym import gymapi
from isaacgym import gymtorch
import math
import numpy as np

from isaacgym.torch_utils import *


class Camera:
    def create(self, gym, sim, width, height, image_type):
        self.gym = gym
        self.sim = sim
        self.cam_props = gymapi.CameraProperties()
        self.cam_props.width = width
        self.cam_props.height = height
        self.cam_props.enable_tensors = True
        self.cam_props.horizontal_fov = 60

        self.camera_pos = gymapi.Vec3(0.4, 0.2, 0.8)
        self.camera_target = gymapi.Vec3(0, 0.4, 0.5)

        self.image_type =image_type
        
    def add(self, env):
        # add camera
        cam_handle = self.gym.create_camera_sensor(env, self.cam_props)
        self.gym.set_camera_location(cam_handle, env, self.camera_pos, self.camera_target)

        # obtain camera tensor
        cam_tensor = self.gym.get_camera_image_gpu_tensor(
            self.sim, env, cam_handle, self.image_type
        )
        cam_tensor = gymtorch.wrap_tensor(cam_tensor)
        return cam_tensor
