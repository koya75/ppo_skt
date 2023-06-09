from pathlib import Path

from isaacgym import gymapi
from isaacgym import gymtorch
import math
import numpy as np


class Table:
    def create(self, gym, sim):
        self.gym = gym
        self.sim = sim

        table_dims = gymapi.Vec3(1.0, 1.4, 0.4)

        self.pose = gymapi.Transform()
        self.pose.p = gymapi.Vec3(0.0, 0.3, 0.5 * table_dims.z)

        self.color = gymapi.Vec3(1.0, 1.0, 1.0)#0.35, 0.35, 0.35#0.24, 0.35, 0.8

        asset_options = gymapi.AssetOptions()
        asset_options.armature = 0.001
        asset_options.flip_visual_attachments = True
        asset_options.fix_base_link = True
        asset_options.thickness = 0.002

        asset_options.mesh_normal_mode = gymapi.COMPUTE_PER_VERTEX

        self.asset = gym.create_box(sim, table_dims.x, table_dims.y, table_dims.z, asset_options)

        self._num_bodies = self.gym.get_asset_rigid_body_count(self.asset)
        self._num_dofs = self.gym.get_asset_dof_count(self.asset)
        self._num_shapes = self.gym.get_asset_rigid_shape_count(self.asset)

    def add(self, env, collisionGroup):
        handle = self.gym.create_actor(env, self.asset, self.pose, "table", collisionGroup, 0)

        self.gym.set_rigid_body_color(
            env, handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, self.color
        )


    @property
    def num_bodies(self):
        return self._num_bodies

    @property
    def num_shapes(self):
        return self._num_shapes

    @property
    def num_dofs(self):
        return self._num_dofs