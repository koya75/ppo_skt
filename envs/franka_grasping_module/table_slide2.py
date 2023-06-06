from pathlib import Path

from isaacgym import gymapi
from isaacgym import gymtorch
import math
import numpy as np


class Table:
    def create(self, gym, sim):
        self.gym = gym
        self.sim = sim

        table1_dims = gymapi.Vec3(0.6, 1.2, 0.4)
        table2_dims = gymapi.Vec3(0.02, 0.3, 0.3)

        self.pose1 = gymapi.Transform()
        self.pose1.p = gymapi.Vec3(0.0, 0.35, 0.5 * table1_dims.z)
        self.pose2 = gymapi.Transform()
        self.pose2.p = gymapi.Vec3(0.0, 0.8, table1_dims.z + (0.5 * table2_dims.z))

        self.color1 = gymapi.Vec3(1.0, 1.0, 1.0)#0.35, 0.35, 0.35#0.24, 0.35, 0.8
        self.color2 = gymapi.Vec3(0.35, 0.35, 0.35)#0.35, 0.35, 0.35#0.24, 0.35, 0.8

        asset_options = gymapi.AssetOptions()
        asset_options.armature = 0.001
        asset_options.flip_visual_attachments = True
        asset_options.fix_base_link = True
        asset_options.thickness = 0.002

        asset_options.mesh_normal_mode = gymapi.COMPUTE_PER_VERTEX

        self.asset1 = gym.create_box(sim, table1_dims.x, table1_dims.y, table1_dims.z, asset_options)
        self.asset2 = gym.create_box(sim, table2_dims.x, table2_dims.y, table2_dims.z, asset_options)

        self._num_bodies1 = self.gym.get_asset_rigid_body_count(self.asset1)
        self._num_dofs1 = self.gym.get_asset_dof_count(self.asset1)
        self._num_shapes1 = self.gym.get_asset_rigid_shape_count(self.asset1)
        self._num_bodies2 = self.gym.get_asset_rigid_body_count(self.asset2)
        self._num_dofs2 = self.gym.get_asset_dof_count(self.asset2)
        self._num_shapes2 = self.gym.get_asset_rigid_shape_count(self.asset2)

    def add(self, env, collisionGroup):
        handle1 = self.gym.create_actor(env, self.asset1, self.pose1, "table1", collisionGroup, 0)
        handle2 = self.gym.create_actor(env, self.asset2, self.pose2, "table2", collisionGroup, 0)

        self.gym.set_rigid_body_color(
            env, handle1, 0, gymapi.MESH_VISUAL_AND_COLLISION, self.color1
        )
        self.gym.set_rigid_body_color(
            env, handle2, 0, gymapi.MESH_VISUAL_AND_COLLISION, self.color2
        )


    @property
    def num_bodies1(self):
        return self._num_bodies1

    @property
    def num_shapes1(self):
        return self._num_shapes1

    @property
    def num_dofs1(self):
        return self._num_dofs1
    
    @property
    def num_bodies2(self):
        return self._num_bodies2

    @property
    def num_shapes2(self):
        return self._num_shapes2

    @property
    def num_dofs2(self):
        return self._num_dofs2