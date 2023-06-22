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
from utils import min_max, make_en_attention, make_en_img, make_de_attention, make_de_img, sk_make_en_attention
import cv2
from utils import load_args
from envs import task_map


def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--number", type=str, default="results")
    parser.add_argument('--arg_file', type=str, default='args.json', help='json file name saved training settings')
    parser.add_argument('--num_episode', type=int, default=10, help='Number of episodes at the time of testing')
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--abci", action="store_true", default=False)
    parser.add_argument(
        "--task",
        help="choose task",
        type=str,
        choices=[
            "Franka",
            "HSR",
            "Anymal",
        ],
    )
    parser.add_argument(
        "--gpu",
        "-g",
        type=int,
        default=0,
        help="GPU ID (negative value indicates CPU)",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        default=False,
        help="Render env states in a GUI window.",
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
    env_name = args.task
    if args.abci:
        directory = "results/ABCI/PPO/PPO_preTrained" + '/' + env_name + '/PPO_' + env_name + args.number + '/'
    else:
        directory = "results/PPO/PPO_preTrained" + '/' + env_name + '/PPO_' + env_name + args.number + '/'
    # load train args #####################################
    print(os.path.join(directory, args.arg_file))
    train_args = load_args(os.path.join(directory, args.arg_file))

    has_continuous_action_space = True
    max_ep_len = train_args['max_step']         # max timesteps in one episode
    action_std = 0.1            # set same std for action distribution which was used while saving

    #render = True              # render environment on screen
    frame_delay = 0             # if required; add delay b/w frames

    total_test_episodes = args.num_episode    # total num of testing episodes

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
    item_asset_root = train_args['item_urdf_dir']
    isaacgym_asset_root = train_args['isaacgym_assets_dir']
    trans_flag = False

    num_items = train_args['num_items']
    item_names = train_args['item_names']
    if item_names is None:
        item_names = sorted(list(Path(item_asset_root).glob("*.urdf")))
        item_names = [path.stem for path in item_names]
    output_debug_images_dir = train_args['output_sensor_images_dir']#directory

    def make_batch_env(num_envs):
        env = task_map[env_name](
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
            descentstep = train_args['max_step'],
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
    train_args['gpu'] = args.gpu
    ppo_agent = PPO(train_args, state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space, action_std)

    # preTrained weights directory

    random_seed = 0             #### set this to load a particular checkpoint trained on random seed
    run_num_pretrained = 0      #### set this to load a particular checkpoint num

    checkpoint_path = directory + "PPO_best.pth"
    #checkpoint_path = directory + "PPO_{}_{}_{}.pth".format(env_name, random_seed, run_num_pretrained)
    print("loading network from : " + checkpoint_path)

    ppo_agent.load(checkpoint_path)

    print("--------------------------------------------------------------------------------------------")

    image_dir = "results/PPO/PPO_image"
    if not os.path.exists(image_dir):
          os.makedirs(image_dir)

    image_dir = image_dir + '/' + env_name + '/'
    if not os.path.exists(image_dir):
          os.makedirs(image_dir)

    #### get number of log files in log directory
    run_num = 0
    current_num_dir = next(os.walk(image_dir))[1]
    run_num = len(current_num_dir)

    #### create new log file for each run
    image_dir_name = image_dir + '/PPO_' + env_name + str(run_num) + '/'
    if not os.path.exists(image_dir_name):
        os.makedirs(image_dir_name)

    test_running_reward = 0
    done = True

    for ep in range(1, total_test_episodes+1):
        # make directry
        epi_dir = os.path.join(image_dir_name, "epi{}".format(ep))
        if not os.path.exists(epi_dir):
            os.makedirs(epi_dir)
        if not os.path.exists(epi_dir + "/raw_img/"):
            os.makedirs(epi_dir + "/raw_img/")
        if not os.path.exists(epi_dir + "/image_encoder/"):
            os.mkdir(epi_dir + "/image_encoder/")
        #for i in range(10):
        if not os.path.exists(epi_dir + "/sketch_encoder/"):#.format(i)_{}
            os.mkdir(epi_dir + "/sketch_encoder/")#.format(i)_{}
        if not os.path.exists(epi_dir + "/decoder_act/"):
            os.mkdir(epi_dir + "/decoder_act/")

        ep_reward = 0
        state, rand = envs.reset()
        raw_list, en_list, de_list = [], [], []
        sketch_querys, sk_att = ppo_agent.select_query(rand)

        for t in range(1, max_ep_len+1):
            action, raw_img, en_atts, de_atts = ppo_agent.select_action(state, sketch_querys, t-1)

            raw_img = raw_img[0].permute(1, 2, 0).cpu().detach().numpy()
            raw_img = raw_img*255
            raw_img = raw_img.astype(np.uint8)
            raw_list.append(raw_img)

            en_attention = make_en_attention(en_atts)
            en_list.append(en_attention)

            de_attention = make_de_attention(de_atts)
            de_list.append(de_attention)

            state, reward, done, _ = envs.step(action)
            ep_reward += reward.item()

            if done:
                break

        # clear buffer
        ppo_agent.buffer.clear()

        test_running_reward +=  ep_reward
        print('Episode: {} \t\t Reward: {}'.format(ep, round(ep_reward, 2)))
        ep_reward = 0

        cv2.imwrite(epi_dir + "/raw_img/raw_{0:06d}.png".format(0), raw_list[0])
        raw_list, en_list, de_list = np.array(raw_list), np.array(en_list), np.array(de_list)
        de_max, de_min = de_list.reshape(-1).max(), de_list.reshape(-1).min()

        en_mean = np.zeros((en_list.shape[1], en_list.shape[2]))
        for idx in range(en_list.shape[0]):
            en_mean += en_list[idx]
        en_max = max(en_mean.flatten())
        en_min = min(en_mean.flatten())
        for x in range(en_mean.shape[0]):
            for y in range(en_mean.shape[1]):
                en_mean[x][y] = en_mean[x][y] / en_max
    
        #en_mean = np.mean(en_list, axis=0)
        en_max, en_min = en_mean.reshape(-1).max(), en_mean.reshape(-1).min()
        en_mean = min_max(en_mean, en_min, en_max)
        make_en_img(en_mean * 255, raw_list[0], 0, epi_dir, mode="mean")
        sk_make_en_attention(sk_att, epi_dir)

        for idx in range(len(raw_list)):
            raw_img = raw_list[idx]
            en_att = en_list[idx]
            de_att = de_list[idx]
            de_att = min_max(de_att, de_min, de_max)

            cv2.imwrite(epi_dir + "/raw_img/raw_{0:06d}.png".format(idx), raw_img)
            make_en_img(en_att * 255, raw_img, idx, epi_dir)
            make_de_img(de_att * 255, idx, epi_dir)

    print("============================================================================================")

    avg_test_reward = test_running_reward / total_test_episodes
    avg_test_reward = round(avg_test_reward, 2)
    print("average test reward : " + str(avg_test_reward))

    print("============================================================================================")


if __name__ == '__main__':

    test()
