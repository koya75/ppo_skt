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
        self.cam_props.horizontal_fov = 100

        #self.camera_pos = gymapi.Vec3(0.0, 0.2, 0.8)
        #self.camera_target = gymapi.Vec3(0.0, 0.2, 0.0)
        self.transform = gymapi.Transform()
        self.transform.p = gymapi.Vec3(-0.01, 0.0, 0.1)#0.33, 0.43
        self.transform.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(1,0,0), np.radians(180.0))*gymapi.Quat.from_axis_angle(gymapi.Vec3(0,0,1), np.radians(180.0))

        self.image_type =image_type
        
    def add(self, env, hand_handle):
        # add camera
        cam_handle = self.gym.create_camera_sensor(env, self.cam_props)
        #self.gym.set_camera_transform(cam_handle, env, self.transform)
        self.gym.attach_camera_to_body(cam_handle, env, hand_handle, self.transform, gymapi.FOLLOW_TRANSFORM)

        # obtain camera tensor
        cam_tensor = self.gym.get_camera_image_gpu_tensor(
            self.sim, env, cam_handle, self.image_type
        )
        cam_tensor = gymtorch.wrap_tensor(cam_tensor)
        return cam_tensor
