"""
Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.


Kuka bin perfromance test
-------------------------------
Test simulation perfromance and stability of the robotic arm dealing with a set of complex objects in a bin.
"""

from __future__ import print_function, division, absolute_import

import os
import math
import numpy as np
from isaacgym import gymapi
from isaacgym import gymutil
from isaacgym import gymtorch
from copy import copy

axes_geom = gymutil.AxesGeometry(0.1)

sphere_rot = gymapi.Quat.from_euler_zyx(0.5 * math.pi, 0, 0)
sphere_pose = gymapi.Transform(r=sphere_rot)
sphere_geom = gymutil.WireframeSphereGeometry(0.03, 12, 12, sphere_pose, color=(1, 0, 0))

colors = [gymapi.Vec3(1.0, 0.0, 0.0),
          gymapi.Vec3(1.0, 127.0/255.0, 0.0),
          gymapi.Vec3(1.0, 1.0, 0.0),
          gymapi.Vec3(0.0, 1.0, 0.0),
          gymapi.Vec3(0.0, 0.0, 1.0),
          gymapi.Vec3(39.0/255.0, 0.0, 51.0/255.0),
          gymapi.Vec3(139.0/255.0, 0.0, 1.0)]

tray_color = gymapi.Vec3(2.0, 2.0, 2.0)
banana_color = gymapi.Vec3(0.85, 0.88, 0.2)
brick_color = gymapi.Vec3(0.9, 0.5, 0.1)


# initialize gym
gym = gymapi.acquire_gym()

# parse arguments
args = gymutil.parse_arguments(
    description="Kuka Bin Test",
    custom_parameters=[
        {"name": "--num_envs", "type": int, "default": 16, "help": "Number of environments to create"},
        {"name": "--num_objects", "type": int, "default": 10, "help": "Number of objects in the bin"},
        {"name": "--object_type", "type": int, "default": 0, "help": "Type of bjects to place in the bin: 0 - box, 1 - meat can, 2 - banana, 3 - mug, 4 - brick, 5 - random"}])

num_envs = args.num_envs
num_objects = args.num_objects
box_size = 0.05

item_name = "item11"

# configure sim
sim_type = args.physics_engine
sim_params = gymapi.SimParams()
if sim_type == gymapi.SIM_FLEX:
    sim_params.substeps = 4
    sim_params.flex.solver_type = 5
    sim_params.flex.num_outer_iterations = 4
    sim_params.flex.num_inner_iterations = 20
    sim_params.flex.relaxation = 0.75
    sim_params.flex.warm_start = 0.8
elif sim_type == gymapi.SIM_PHYSX:
    sim_params.substeps = 2
    sim_params.physx.solver_type = 1
    sim_params.physx.num_position_iterations = 25
    sim_params.physx.num_velocity_iterations = 0
    sim_params.physx.num_threads = args.num_threads
    sim_params.physx.use_gpu = args.use_gpu
    sim_params.physx.rest_offset = 0.001

sim_params.use_gpu_pipeline = False
if args.use_gpu_pipeline:
    print("WARNING: Forcing CPU pipeline.")

sim = gym.create_sim(args.compute_device_id, args.graphics_device_id, sim_type, sim_params)
if sim is None:
    print("*** Failed to create sim")
    quit()

# add ground plane
plane_params = gymapi.PlaneParams()
gym.add_ground(sim, plane_params)

# create viewer
viewer = gym.create_viewer(sim, gymapi.CameraProperties())
if viewer is None:
    print("*** Failed to create viewer")
    quit()

# load assets
asset_root = "../../assets"

table_dims = gymapi.Vec3(0.8, 0.4, 1.0)

pose = gymapi.Transform()
pose.p = gymapi.Vec3(0.0, 0.0, 0.0)
pose.r = gymapi.Quat(-0.707107, 0.0, 0.0, 0.707107)

asset_options = gymapi.AssetOptions()
asset_options.armature = 0.001
asset_options.fix_base_link = True
asset_options.thickness = 0.002

asset_options.mesh_normal_mode = gymapi.COMPUTE_PER_VERTEX

table_pose = gymapi.Transform()
table_pose.p = gymapi.Vec3(0.7, 0.5 * table_dims.y + 0.001, 0.0)

bin_pose = gymapi.Transform()
bin_pose.r = gymapi.Quat(-0.707107, 0.0, 0.0, 0.707107)

object_pose = gymapi.Transform()

table_asset = gym.create_box(sim, table_dims.x, table_dims.y, table_dims.z, asset_options)

# load assets of objects in a bin
asset_options.fix_base_link = False

can_asset_file = "urdf/ycb/010_potted_meat_can/010_potted_meat_can.urdf"
banana_asset_file = "urdf/ycb/011_banana/011_banana.urdf"
mug_asset_file = "urdf/ycb/025_mug/025_mug.urdf"
brick_asset_file = "urdf/ycb/061_foam_brick/061_foam_brick.urdf"

object_files = []
object_files.append(can_asset_file)
object_files.append(banana_asset_file)
object_files.append(mug_asset_file)
object_files.append(object_files)

object_assets = []

object_assets.append(gym.create_box(sim, box_size, box_size, box_size, asset_options))
object_assets.append(gym.load_asset(sim, asset_root, can_asset_file, asset_options))
object_assets.append(gym.load_asset(sim, asset_root, banana_asset_file, asset_options))
object_assets.append(gym.load_asset(sim, asset_root, mug_asset_file, asset_options))
object_assets.append(gym.load_asset(sim, asset_root, brick_asset_file, asset_options))

spawn_height = gymapi.Vec3(0.0, 0.3, 0.0)

# load bin asset
bin_asset_file = "urdf/tray/traybox.urdf"

print("Loading asset '%s' from '%s'" % (bin_asset_file, asset_root))
bin_asset = gym.load_asset(sim, asset_root, bin_asset_file, asset_options)

corner = table_pose.p - table_dims * 0.5

item_asset_file = "urdf/item/"+item_name+".urdf"
item_asset = gym.load_asset(sim, asset_root, item_asset_file, asset_options)

asset_options.fix_base_link = True
asset_options.flip_visual_attachments = False
asset_options.collapse_fixed_joints = True
asset_options.disable_gravity = True

if sim_type == gymapi.SIM_FLEX:
    asset_options.max_angular_velocity = 40.

# set up the env grid
spacing = 1.5
env_lower = gymapi.Vec3(-spacing, 0.0, -spacing)
env_upper = gymapi.Vec3(spacing, spacing, spacing)

# cache some common handles for later use
envs = []
kuka_handles = []
tray_handles = []
object_handles = []

# Attractors setup
kuka_attractors = ["iiwa7_link_7"]  # , "thumb_link_3", "index_link_3", "middle_link_3", "ring_link_3"]
attractors_offsets = [gymapi.Transform(), gymapi.Transform(), gymapi.Transform(), gymapi.Transform(), gymapi.Transform()]

# Coordinates to offset attractors to tips of fingers
# thumb
attractors_offsets[1].p = gymapi.Vec3(0.07, 0.01, 0)
attractors_offsets[1].r = gymapi.Quat(0.0, 0.0, 0.216433, 0.976297)
# index, middle and ring
for i in range(2, 5):
    attractors_offsets[i].p = gymapi.Vec3(0.055, 0.015, 0)
    attractors_offsets[i].r = gymapi.Quat(0.0, 0.0, 0.216433, 0.976297)

attractor_handles = {}

print("Creating %d environments" % num_envs)
num_per_row = int(math.sqrt(num_envs))
base_poses = []

for i in range(num_envs):
    # create env
    env = gym.create_env(sim, env_lower, env_upper, num_per_row)
    envs.append(env)

    table_handle = gym.create_actor(env, table_asset, table_pose, "table", i, 0)
    gym.set_actor_scale(env, table_handle, 1.3)
    gym.set_rigid_body_color(env, table_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, tray_color)

    """x = corner.x + table_dims.x * 0.5
    y = table_dims.y + box_size + 0.01
    z = corner.z + table_dims.z * 0.5

    bin_pose.p = gymapi.Vec3(x, y, z)
    tray_handles.append(gym.create_actor(env, bin_asset, bin_pose, "bin", i, 0))
    gym.set_actor_scale(env, tray_handles[-1], 1.3)

    gym.set_rigid_body_color(env, tray_handles[-1], 0, gymapi.MESH_VISUAL_AND_COLLISION, tray_color)"""

    x = 0.65
    y = 0.55
    z = 0.0

    object_pose.p = gymapi.Vec3(x, y, z) + spawn_height
    #object_pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0,1,0), np.radians(45.0))
    #object_pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0,0,1), np.radians(-90.0))# *gymapi.Quat.from_axis_angle(gymapi.Vec3(1,0,0), np.radians(45.0))

    object_handles.append(gym.create_actor(env, item_asset, object_pose, "object", i, 0))

next_kuka_update_time = 0.1
frame = 0

cam_props = gymapi.CameraProperties()
cam_props.width = 128
cam_props.height = 128
cam_props.horizontal_fov = 100
camera_pos = gymapi.Vec3(0.65, 0.7, 0.0)
camera_target = gymapi.Vec3(0.7, -2.0, 0.0)
os.makedirs("../../camera_2",exist_ok=True)
cam_handle = gym.create_camera_sensor(env, cam_props)
gym.set_camera_location(cam_handle, env, camera_pos, camera_target)
power = gymapi.Vec3(0.03, 0.03, 0.03)
color = gymapi.Vec3(0.0, 0.0, 0.0)
target = gymapi.Vec3(-1.5707963268, 0.0, 0.0)
gym.set_light_parameters(sim, 1, power, color, target)

while not gym.query_viewer_has_closed(viewer):
    # check if we should update
    t = gym.get_sim_time(sim)

    # step the physics
    gym.simulate(sim)
    gym.fetch_results(sim, True)

    gym.render_all_camera_sensors(sim)
    gym.start_access_image_tensors(sim)
    gym.write_camera_image_to_file(sim, env, cam_handle, gymapi.IMAGE_COLOR, "../../camera_2/"+item_name+".png")

#    for env in envs:
#        gym.draw_env_rigid_contacts(viewer, env, colors[0], 0.5, True)

    # step rendering
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, False)
    gym.sync_frame_time(sim)
    gym.end_access_image_tensors(sim)



    frame = frame + 1

print("Done")

gym.destroy_viewer(viewer)
gym.destroy_sim(sim)
