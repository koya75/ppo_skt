import os
import sys
from pathlib import Path

import numpy as np
import imageio

from isaacgym import gymtorch
from isaacgym import gymapi
from isaacgym import gymutil
from isaacgym.torch_utils import *
import torch

from gym import spaces

from .franka_grasping_module.table import Table
from .franka_grasping_module.franka_slide import Franka
#from .franka_grasping_module.item_slide import Item
from .franka_grasping_module.camera_map import Camera

from repos.pfrl.pfrl.env import VectorEnv

import torch.distributed as dist


class FrankaGraspingEnv(VectorEnv):
    def __init__(
        self,
        num_envs,
        width,
        height,
        discrete,
        image_type,
        output_debug_images_dir,
        item_asset_root,
        isaacgym_asset_root,
        num_items,
        item_names,
        use_viewer,
        action_repeat,
        device_id,
        n_actions,
        descentstep,
    ):

        self.max_episode_length = descentstep#50
        self.dt = 1 / 60.0
        self.substep = 4
        self.n_actions = n_actions
        # self.device_type = "cuda"
        self.device_id = device_id
        self.num_envs = num_envs
        self.width = width
        self.height = height
        self.output_debug_images_dir = output_debug_images_dir
        self.discrete = True
        if discrete:
            self.discrete = False
        self.item_asset_root = item_asset_root
        self.isaacgym_asset_root = isaacgym_asset_root
        if image_type == "color":
            self.image_type = gymapi.IMAGE_COLOR
        elif image_type == "depth":
            self.image_type = gymapi.IMAGE_DEPTH
        self.actionRepeat = action_repeat  # 15
        self.num_items = num_items
        self.item_names = item_names
        self.use_viewer = use_viewer
        self.random_int = 0

        self.device = "cuda" + ":" + str(self.device_id)
        print(self.device)

        if self.image_type == gymapi.IMAGE_COLOR:
            self.observation_space = spaces.Box(
                low=0.0, high=1.0, shape=(3, height, width), dtype=np.float32
            )
        elif self.image_type == gymapi.IMAGE_DEPTH:
            self.observation_space = spaces.Box(
                low=0.0, high=1.0, shape=(1, height, width), dtype=np.float32
            )

        if self.discrete:
            self.action_space = spaces.Discrete(self.n_actions)
        else:
            self.action_space = spaces.Box(low=-1, high=1, shape=(2, 1))
        self.action_shape = self.action_space.shape[0]

        self.use_manual_action = False
        self.manual_action = 0

        self.gym = gymapi.acquire_gym()

        # optimization flags for pytorch JIT
        torch._C._jit_set_profiling_mode(False)
        torch._C._jit_set_profiling_executor(False)

        # allocate buffers
        self.obs_buf = torch.zeros(
            (self.num_envs, self.observation_space.shape[2], self.height, self.width),
            device=self.device,
            dtype=torch.float,
        )
        self.rew_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.done_buf = torch.ones(self.num_envs, device=self.device, dtype=torch.long)
        self.infos = [{} for _ in range(self.num_envs)]
        self.progress_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long
        )

        # create envs, sim and viewer
        self._create_sim()
        self.gym.prepare_sim(self.sim)

        if self.use_viewer:
            # todo: read from config
            self.enable_viewer_sync = False

            # subscribe to keyboard shortcuts
            self.viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_ESCAPE, "QUIT"
            )
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_V, "toggle_viewer_sync"
            )
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_LEFT, "left"
            )
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_RIGHT, "right"
            )
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_DOWN, "down"
            )
            self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_UP, "up")
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_A, "counter_clockwise"
            )
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_S, "clockwize"
            )

            # set the camera position based on up axis
            cam_pos = gymapi.Vec3(10.0, 10.0, 3.0)
            cam_target = gymapi.Vec3(5.0, 5.0, 0.0)
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)
        else:
            self.viewer = None

        # get gym GPU state tensors
        actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(
            self.sim
        )  # シミュレーション全体の状態を取得
        rigid_body_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)

        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)

        # create some wrapper tensors for different slices

        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_tensor).view(
            self.num_envs, -1, 13
        )
        self.root_state_tensor = gymtorch.wrap_tensor(actor_root_state_tensor).view(
            self.num_envs, -1, 13
        )

        self.num_bodies = self.rigid_body_states.shape[1]
        self.num_dofs = self.gym.get_sim_dof_count(self.sim) // self.num_envs

        self.franka.set_tensors(
            self.rigid_body_states,
            self.num_envs,
            self.dof_state,
            self.sim_params,
            self.viewer,
        )
        """self.item.set_tensors(
            self.root_state_tensor, self.num_envs, self.rigid_body_states
        )"""

        self.axes_geom = gymutil.AxesGeometry(0.3)

        self.global_indices = torch.arange(
            self.num_envs * (3), dtype=torch.int32, device=self.device
        ).view(self.num_envs, -1)

    def _create_sim(self):
        # initialize sim
        sim_params = gymapi.SimParams()
        sim_params.dt = self.dt
        sim_params.substeps = self.substep
        sim_params.use_gpu_pipeline = True
        sim_params.up_axis = gymapi.UP_AXIS_Z
        sim_params.gravity.x = 0
        sim_params.gravity.y = 0
        sim_params.gravity.z = -9.81

        # sim_params.physx.solver_type = 1
        # sim_params.physx.num_position_iterations = 4
        # sim_params.physx.num_velocity_iterations = 1
        # sim_params.physx.num_threads = 4
        # sim_params.physx.use_gpu = True
        # sim_params.physx.num_subscenes = 0
        # sim_params.physx.max_gpu_contact_pairs = 8 * 1024 * 1024

        sim_params.physx.use_gpu = True
        sim_params.physx.solver_type = 1
        sim_params.physx.num_position_iterations = 8
        sim_params.physx.num_velocity_iterations = 1
        sim_params.physx.rest_offset = 0.0
        sim_params.physx.contact_offset = 0.001
        sim_params.physx.friction_offset_threshold = 0.001
        sim_params.physx.friction_correlation_distance = 0.0005
        sim_params.physx.num_threads = 8

        self.sim_params = sim_params

        self.sim = self.gym.create_sim(
            self.device_id,
            self.device_id,
            gymapi.SIM_PHYSX,
            self.sim_params,
        )

        self._create_ground_plane()

        envSpacing = 0.8
        self._create_envs(envSpacing, int(np.sqrt(self.num_envs)))

        l_color = gymapi.Vec3(0.4, 0.4, 0.4)
        l_ambient = gymapi.Vec3(0.5, 0.5, 0.5)
        l_direction = gymapi.Vec3(1, 1, 3)
        self.gym.set_light_parameters(self.sim, 0, l_color, l_ambient, l_direction)
        l_direction = gymapi.Vec3(-1, 1, 3)
        self.gym.set_light_parameters(self.sim, 1, l_color, l_ambient, l_direction)

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self, spacing, num_per_row):
        #########################################################
        # 一つの環境の大きさ
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        ########## franka ##########
        self.franka = Franka()
        self.franka.create(
            self.gym,
            self.sim,
            self.device,
            self.discrete,
            self.actionRepeat,
            self.isaacgym_asset_root,
            self.max_episode_length,
        )
        ########## table1 ##########
        self.table = Table()
        self.table.create(self.gym, self.sim)
        """########## tray ##########
        self.tray = Tray()
        self.tray.create(self.gym, self.sim, self.device, self.isaacgym_asset_root)
        ########## item ##########
        self.item = Item()
        self.item.create(
            self.gym,
            self.sim,
            self.device,
            self.num_items,
            self.item_asset_root,
            self.item_names,
        )"""
        ########## camera ##########
        self.camera = Camera()
        self.camera.create(self.gym, self.sim, self.width, self.height, self.image_type)

        #########################################################

        print("num envs: ", self.num_envs)
        print("num franka bodies: ", self.franka.num_bodies)
        print("num franka dofs: ", self.franka.num_dofs)
        print("num table bodies: ", self.table.num_bodies)
        print("num table dofs: ", self.table.num_dofs)

        # compute aggregate size

        self.envs = []
        self.cam_tensors = []

        for i in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)
            self.envs.append(env_ptr)
            #self.item.select_item()
            # print("num item bodies: ", self.item.num_bodies)
            # print("num item dofs: ", self.item.num_dofs)

            max_agg_bodies = (
                self.franka.num_bodies
                + self.table.num_bodies
            )#+ self.item.num_bodies
            max_agg_shapes = (
                self.franka.num_shapes
                + self.table.num_shapes
            )#+ self.item.num_shapes

            self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)
            self.franka.add(env_ptr, i)
            self.table.add(env_ptr, i)
            #self.tray.add(env_ptr, i)
            #self.item.add(env_ptr, i)
            self.cam_tensors.append(self.camera.add(env_ptr))#, self.franka.hand_handle
            self.gym.end_aggregate(env_ptr)

        self._init_data()

    def step(self, actions):
        done = self._pre_physics_step(actions)
        self.done_buf[:] = done
        self._post_physics_step()

        return (
            self.obs_buf,#tuple()
            self.rew_buf,#tuple().cpu().numpy()
            self.done_buf.cpu().numpy(),#tuple()
            self.infos,#tuple()
        )

    def _reset(self, env_ids):
        self.franka.reset(env_ids)
        self.progress_buf[env_ids] = 0
        self.done_buf[env_ids] = 0
        for _ in range(80):
            self.gym.refresh_actor_root_state_tensor(self.sim)
            self.gym.refresh_dof_state_tensor(self.sim)
            self.gym.refresh_rigid_body_state_tensor(self.sim)
            self.gym.refresh_jacobian_tensors(self.sim)
            self.franka.apply_target_pose()
            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim, True)
            self.gym.step_graphics(self.sim)
            if self.use_viewer:
                self.gym.draw_viewer(self.viewer, self.sim, True)
        """for actor_idx in range(self.item.num_actors):
            self.item.reset_one(env_ids, actor_idx)
            self.item.reset_one(env_ids, actor_idx)
            for _ in range(20):
                self.gym.refresh_actor_root_state_tensor(self.sim)
                self.gym.refresh_dof_state_tensor(self.sim)
                self.gym.refresh_rigid_body_state_tensor(self.sim)
                self.gym.refresh_jacobian_tensors(self.sim)
                self.franka.apply_target_pose()
                self.gym.simulate(self.sim)
                self.gym.fetch_results(self.sim, True)
                self.gym.step_graphics(self.sim)
                if self.use_viewer:
                    self.gym.draw_viewer(self.viewer, self.sim, True)"""
        self._compute_observations()
        return self.obs_buf

    def reset(self, mask=None):
        if mask is not None:
            if np.any(~mask): # can not partial envs reset.
                self.random_int = torch.randint(4, (1,), device=self.device)
                self._reset(torch.arange(self.num_envs, device=self.device))
        else:
            self.random_int = torch.randint(4, (1,), device=self.device)
            self._reset(torch.arange(self.num_envs, device=self.device))
        return self.obs_buf, self.random_int

    def _init_data(self):

        franka_local_grasp_pose = gymapi.Transform()
        franka_local_grasp_pose.p = gymapi.Vec3(0, 0, 0.1)
        self.franka_local_grasp_pos = to_torch(
            [
                franka_local_grasp_pose.p.x,
                franka_local_grasp_pose.p.y,
                franka_local_grasp_pose.p.z,
            ],
            device=self.device,
        ).repeat((self.num_envs, 1))
        self.franka_local_grasp_rot = to_torch(
            [
                franka_local_grasp_pose.r.x,
                franka_local_grasp_pose.r.y,
                franka_local_grasp_pose.r.z,
                franka_local_grasp_pose.r.w,
            ],
            device=self.device,
        ).repeat((self.num_envs, 1))

    def _compute_reward(self, actions):
        self.rew_buf[:], self.done_buf[:] = compute_franka_reward(
            self.done_buf,
            self.progress_buf,
            self.actions,
            self.global_franka_pos,
            self.max_episode_length,
            self.hand_pos,
            self.hand_rot,
            self.random_int,
        )#self.item_pos,

    def _compute_observations(self):

        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)

        self.hand_pos = self.franka.hand_pos
        self.hand_rot = self.franka.hand_rot
        self.right_finger_pos = self.franka.right_finger_pos
        self.left_finger_pos = self.franka.left_finger_pos
        self.global_franka_rot, self.global_franka_pos = tf_combine(
            self.hand_rot,
            self.hand_pos,
            self.franka_local_grasp_rot,
            self.franka_local_grasp_pos,
        )
        #self.item_pos = self.item.pos
        #self.item_rot = self.item.rot
        if False:
            self.gym.clear_lines(self.viewer)
            for i, env in enumerate(self.envs):
                pose = gymapi.Transform()
                pose.p.x = self.item_pos[i, 0, 0]
                pose.p.y = self.item_pos[i, 0, 1]
                pose.p.z = self.item_pos[i, 0, 2]
                # pose.r.x = self.item_rot[i, 0, 0]
                # pose.r.y = self.item_rot[i, 0, 1]
                # pose.r.z = self.item_rot[i, 0, 2]
                # pose.r.w = self.item_rot[i, 0, 3]
                gymutil.draw_lines(self.axes_geom, self.gym, self.viewer, env, pose)

                pose = gymapi.Transform()
                pose.p.x = self.global_franka_pos[i, 0]
                pose.p.y = self.global_franka_pos[i, 1]
                pose.p.z = self.global_franka_pos[i, 2]
                pose.r.x = self.global_franka_rot[i, 0]
                pose.r.y = self.global_franka_rot[i, 1]
                pose.r.z = self.global_franka_rot[i, 2]
                pose.r.w = self.global_franka_rot[i, 3]
                gymutil.draw_lines(self.axes_geom, self.gym, self.viewer, env, pose)

        # camera buf
        self.gym.render_all_camera_sensors(self.sim)
        self.gym.start_access_image_tensors(self.sim)

        if self.output_debug_images_dir is not None:
            for i, _ in enumerate(self.envs):
                os.makedirs(self.output_debug_images_dir, exist_ok=True)

                cam_img = self.cam_tensors[i].detach().cpu().numpy()
                imageio.imwrite(Path(self.output_debug_images_dir).joinpath("image_env_{}.png".format(i)), cam_img)

        if self.image_type == gymapi.IMAGE_COLOR:
            self.obs_buf = torch.stack(self.cam_tensors)[:, :, :, :3]
            self.obs_buf = self.obs_buf.permute(0, 3, 1, 2) / 255.0
        if self.image_type == gymapi.IMAGE_DEPTH:
            self.obs_buf = torch.stack(self.cam_tensors).unsqueeze(-1)
            self.obs_buf = self.obs_buf.permute(0, 3, 1, 2)

        self.gym.end_access_image_tensors(self.sim)

        return self.obs_buf

    def _pre_physics_step(self, actions):
        self.actions = (
            #torch.from_numpy(actions.astype(np.float32)).clone().to(self.device)
            #torch.from_numpy(np.array(, dtype=np.int32))
            actions.detach().clone().to(self.device)
        )
        if self.use_manual_action:
            self.actions[0] = self.manual_action
        done = self.franka.pre_physics_step(self.actions, self.n_actions)#, self.item.pos

        return done

    def _post_physics_step(self):
        self._compute_observations()
        self._compute_reward(self.actions)
        self.progress_buf += 1

    def get_screen(self):
        return self.obs_buf

    def get_step(self):
        return self.progress_buf

    def seed(self, seeds):
        raise NotImplementedError()

    def close(self):
        pass
        # print("--- env close---")
        # if self.viewer is not None:
        #     self.gym.destroy_viewer(self.viewer)
        # self.gym.destroy_sim(self.sim)


#####################################################################
###=========================jit functions=========================###
#####################################################################


@torch.jit.script
def compute_franka_reward(
    done_buf,
    progress_buf,
    actions,
    global_franka_pos,
    max_episode_length,
    hand_pos,
    hand_rot,
    randint,
):#item_pos,
    # type: (Tensor, Tensor, Tensor, Tensor, float, Tensor, Tensor, Tensor) -> Tuple[Tensor, Tensor]
    flag = "hight" # hight , right , left , pickandplease

    # regularization on the actions (summed for each environment)

    gorl = torch.zeros_like(hand_pos)
    gorl[:, 2] = 5.3
    if randint==0:
        gorl[:, 0] = -0.3
        gorl[:, 1] = 0.6
    elif randint==1:
        gorl[:, 0] = 0.3
        gorl[:, 1] = 0.6
    elif randint==0:
        gorl[:, 0] = -0.2
        gorl[:, 1] = 0.6
    elif randint==1:
        gorl[:, 0] = 0.2
        gorl[:, 1] = 0.6
    """dpos = (item_pos - global_franka_pos.unsqueeze(1))
    distance = torch.norm(dpos,dim=-1)
    item_x = item_pos[:, 1, 0]
    item_y = item_pos[:, 1, 1]
    item_z = item_pos[:, 1, 2]
    item_offset = torch.zeros_like(item_y)
    tem_offset = torch.zeros_like(item_z)
    item_offset[:, 0] = 0.4314
    item_offset[:] = 0.03

    if flag == "pickandplease":
        item_x = item_pos[:, 0, 0]
        item_y = item_pos[:, 0, 1]
        item_z = item_pos[:, 0, 2]
        
        x_low = item_x >= -0.08
        x_up  = item_x <= 0.08
        y_low = item_y >= 0.45
        y_up  = item_y <= 0.65
        z_high = item_z <= 0.5

        condition = (x_low * x_up * y_low * y_up * z_high).sum() > 0
        reward1 = torch.where(
            torch.bitwise_and(condition, progress_buf > 10), 1.0, 0.0
        )
    elif flag == "hight":
        item_z = item_z - item_offset
        distance_condition = distance < 0.5
        height_condition = item_z >= 0.29
        height_distance_condition = (height_condition * distance_condition).sum(dim=1) > 0

        reward2 = torch.where(
            torch.bitwise_and(height_distance_condition, progress_buf > 10), 1.0, 0.0
        )
    elif flag == "right":
    item_y = item_y - item_offset
    distance_condition = distance < 0.5
    y_condition = item_y >= 0.6
    condition_y = item_y < 0.9
    x_condition = item_x < 0
    condition_x = item_x > -0.3
    height_distance_condition = (y_condition * x_condition * condition_y * condition_x * distance_condition).sum(dim=1) > 0

    rewards = torch.where(
        torch.bitwise_and(height_distance_condition, progress_buf > 10), 1.0, 0.0
    )
    elif flag == "left":
    item_y = item_y - item_offset
    y_condition = torch.round((item_y - 0.65), decimals=2)"""
    x_condition = -abs(hand_pos[:, 0] - gorl[:, 0])
    y_condition = -abs(hand_pos[:, 1] - gorl[:, 1])
    condition = x_condition + y_condition

    rewards = torch.where(
        progress_buf >= max_episode_length - 1, condition, 0.0
    ).sum(dim=0)
        
    done_buf = torch.where(
        progress_buf >= max_episode_length - 1, torch.ones_like(done_buf), done_buf
    )
    # print("rewards mean: ", rewards.mean())
    return rewards, done_buf