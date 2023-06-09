import os
import glob
import time
from datetime import datetime
import pytz
import argparse

from isaacgym import gymtorch
import torch
import numpy as np

import gym

from test_PPO import PPO
import torch.distributed as dist
from pathlib import Path
from make_urdf import URDF


def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=str, default="results")
    parser.add_argument("--isaacgym-assets-dir", type=str, default="../isaacgym/assets")
    parser.add_argument("--item-urdf-dir", type=str, default="./urdf")
    parser.add_argument(
        "--model",
        help="choose model",
        type=str,
        choices=[
            "vanilla",
            "skt",
        ],
    )
    parser.add_argument(
        "--use-lstm", action="store_true", default=False, help="use lstm"
    )
    parser.add_argument("--steps", type=int, default=10 ** 7)
    parser.add_argument("--step_offset", type=int, default=0)
    parser.add_argument("--update-batch-interval", type=int, default=1)
    parser.add_argument("--lr", type=float, default=7e-4)
    parser.add_argument("--gamma", type=float, default=0.99, help="discount factor")
    parser.add_argument(
        "--lambd", type=float, default=0.95, help="Lambda-return factor [0, 1]"
    )
    parser.add_argument("--minibatch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--eval-batch-interval", type=int, default=10 ** 4)
    parser.add_argument("--eval-n-runs", type=int, default=128)
    parser.add_argument("--demo", action="store_true", default=False)
    parser.add_argument("--load", type=str, default="")
    parser.add_argument(
        "--max-grad-norm", type=float, default=40, help="value loss coefficient"
    )
    """parser.add_argument('--gpu-ids', type=str, default='0,1',
                        help='use which gpu to train, must be a \
                        comma-separated list of integers only (default=0)')"""
    parser.add_argument(
        "--gpu",
        "-g",
        type=int,
        default=0,
        help="GPU ID (negative value indicates CPU)",
    )
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument(
        "--log-level",
        type=int,
        default=20,
        help="Logging level. 10:DEBUG, 20:INFO etc.",
    )
    parser.add_argument(
        "--monitor",
        action="store_true",
        default=False,
        help=(
            "Monitor env. Videos and additional information are saved as output files."
        ),
    )
    parser.add_argument(
        "--output-sensor-images-dir",
        type=str,
        default=None,
        help="Output sensor images directory. Image files are updated per step",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        default=False,
        help="Render env states in a GUI window.",
    )
    parser.add_argument(
        "--num-items",
        type=int,
        default=1,
        help="Number of items on tray.",
    )
    parser.add_argument(
        "--item-names", nargs="+", default=None, help=["List of item names."]
    )
    parser.add_argument("--max-step", type=int, default=12)
    
    parser.add_argument(
        "--target",
        type=str,
        default="item21",
        help="target-item",
    )
    parser.add_argument("--drop", action="store_true", default=False, help="use drop")
    parser.add_argument("--hand", action="store_true", default=False, help="use handcamera")
    parser.add_argument("--discrete", action="store_true", default=False, help="use handcamera")
    parser.add_argument(
        "--mode",
        help="choose mode",
        type=str,
        default="normal",
        choices=[
            "normal",
            "hard",
            "veryhard",
        ],
    )

    

    parser.set_defaults(use_lstm=False)
    args = parser.parse_args()

    return args

#################################### Testing ###################################
def test():
    print("============================================================================================")

    ################## hyperparameters ##################

    args = parser()
    urdf = URDF()
    urdf.create_urdf()
    ####### initialize environment hyperparameters ######
    env_name = "Franka"

    has_continuous_action_space = True
    max_ep_len = args.max_step           # max timesteps in one episode
    action_std = 0.1            # set same std for action distribution which was used while saving

    #render = True              # render environment on screen
    frame_delay = 0             # if required; add delay b/w frames

    total_test_episodes = 10    # total num of testing episodes

    K_epochs = 80               # update policy for K epochs
    eps_clip = 0.2              # clip parameter for PPO
    gamma = 0.99                # discount factor

    lr_actor = 0.0003           # learning rate for actor
    lr_critic = 0.001           # learning rate for critic

    #####################################################

    ################################## set device ##################################
    print("============================================================================================")
    # set device to cpu or cuda
    device = torch.device('cpu')
    if(torch.cuda.is_available()): 
        device = torch.device('cuda:{}'.format(args.gpu))
        torch.cuda.empty_cache()
        print("Device set to : " + str(torch.cuda.get_device_name(device)))
    else:
        print("Device set to : cpu")
    args.device = device
    print("============================================================================================")


    height = 128
    width = 128
    action_repeat = 30
    item_asset_root = args.item_urdf_dir
    isaacgym_asset_root = args.isaacgym_assets_dir
    trans_flag = False

    num_items = args.num_items
    item_names = args.item_names
    if item_names is None:
        item_names = sorted(list(Path(item_asset_root).glob("*.urdf")))
        item_names = [path.stem for path in item_names]
    output_debug_images_dir = args.output_sensor_images_dir#directory
    from envs.skt_franka_grasping_env import FrankaGraspingEnv

    def make_batch_env(num_envs):
        env = FrankaGraspingEnv(
            num_envs=num_envs,
            height=height,
            width=width,
            discrete=has_continuous_action_space,
            image_type="color",
            item_asset_root=item_asset_root,
            isaacgym_asset_root=isaacgym_asset_root,
            num_items=num_items,
            item_names=item_names,
            use_viewer=args.render,
            action_repeat=action_repeat,
            output_debug_images_dir=output_debug_images_dir,
            device_id=args.gpu,
            n_actions = 0,
            descentstep = args.max_step,
        )
        return env

    envs = make_batch_env(num_envs=args.num_envs)

    # state space dimension
    state_dim = envs.observation_space.shape[0]

    # action space dimension
    if has_continuous_action_space:
        action_dim = envs.action_space.shape[0]
    else:
        action_dim = envs.action_space.n

    # initialize a PPO agent
    ppo_agent = PPO(args, state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space, action_std)

    # preTrained weights directory

    random_seed = 0             #### set this to load a particular checkpoint trained on random seed
    run_num_pretrained = 0      #### set this to load a particular checkpoint num

    directory = "results/PPO/PPO_preTrained" + '/' + env_name + '/'
    checkpoint_path = directory + "PPO_best.pth"
    #checkpoint_path = directory + "PPO_{}_{}_{}.pth".format(env_name, random_seed, run_num_pretrained)
    print("loading network from : " + checkpoint_path)

    ppo_agent.load(checkpoint_path)

    print("--------------------------------------------------------------------------------------------")

    test_running_reward = 0

    for ep in range(1, total_test_episodes+1):
        ep_reward = 0
        state, rand = envs.reset()
        sketch_querys = ppo_agent.select_query(rand)

        for t in range(1, max_ep_len+1):
            sketch_query = sketch_querys[t-1].unsqueeze(0)
            action = ppo_agent.select_action(state, sketch_query)
            state, reward, done, _ = envs.step(action)
            ep_reward += reward.item()

            if done:
                break

        # clear buffer
        ppo_agent.buffer.clear()

        test_running_reward +=  ep_reward
        print('Episode: {} \t\t Reward: {}'.format(ep, round(ep_reward, 2)))
        ep_reward = 0

    print("============================================================================================")

    avg_test_reward = test_running_reward / total_test_episodes
    avg_test_reward = round(avg_test_reward, 2)
    print("average test reward : " + str(avg_test_reward))

    print("============================================================================================")


if __name__ == '__main__':

    test()
