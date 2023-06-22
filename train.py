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

from PPO import PPO
import torch.distributed as dist
from pathlib import Path
from make_urdf import URDF
from utils import log_plot, save_args
from envs import task_map

def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=str, default="results")
    parser.add_argument("--isaacgym-assets-dir", type=str, default="../isaacgym/assets")
    parser.add_argument("--item-urdf-dir", type=str, default="./urdf")
    parser.add_argument("--local-rank", type=int, help="local rank (this is automatically set by torch.distributed.launch)")
    parser.add_argument(
        "--model",
        help="choose model",
        type=str,
        choices=[
            "vanilla",
            "skt",
            "hoge",
        ],
    )
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
        default=None,
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
    args.is_master = args.local_rank == 0

    return args

################################### Training ###################################
def train():
    print("============================================================================================")

    args = parser()
    urdf = URDF()
    urdf.create_urdf()
    ### initialize the distributed backend
    torch.distributed.init_process_group(backend='nccl')
    ### get the number of GPUs
    _num_gpus = dist.get_world_size()
    if args.gpu is not None:
        args.local_rank = args.gpu
    ####### initialize environment hyperparameters ######
    env_name = args.task

    has_continuous_action_space = True  # continuous action space; else discrete

    max_ep_len = args.max_step                   # max timesteps in one episode
    max_training_timesteps = args.steps  # break training loop if timeteps > max_training_timesteps

    print_freq = max_ep_len * 10        # print avg reward in the interval (in num timesteps)
    log_freq = max_ep_len * args.eval_batch_interval           # log avg reward in the interval (in num timesteps)
    save_model_freq = int(1e4)          # save model frequency (in num timesteps)

    action_std = 0.6                    # starting std for action distribution (Multivariate Normal)
    action_std_decay_rate = 0.05        # linearly decay action_std (action_std = action_std - action_std_decay_rate)
    min_action_std = 0.1                # minimum action_std (stop decay after action_std <= min_action_std)
    action_std_decay_freq = int(2.5e5)  # action_std decay frequency (in num timesteps)
    #####################################################

    ## Note : print/log frequencies should be > than max_ep_len

    ################ PPO hyperparameters ################
    update_timestep = max_ep_len * args.update_batch_interval      # update policy every n timesteps
    K_epochs = 80               # update policy for K epochs in one PPO update

    eps_clip = 0.2          # clip parameter for PPO
    gamma = 0.99            # discount factor

    lr_actor = 0.0003       # learning rate for actor network
    lr_critic = 0.001       # learning rate for critic network

    random_seed = 1         # set random seed if required (0 = no random seed)

    eval_episode_interval = args.eval_batch_interval
    #####################################################

    ################### checkpointing ###################
    run_num_pretrained = 0      #### change this to prevent overwriting weights in same env_name folder

    if args.is_master:
        directory = "results/PPO/PPO_preTrained"
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)

        directory = directory + '/' + env_name + '/'
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)

        #### get number of log files in log directory
        run_num = 0
        current_num_dir = next(os.walk(directory))[1]
        run_num = len(current_num_dir)

        #### create new log file for each run
        directory = directory + '/PPO_' + env_name + str(run_num) + '/'
        if not os.path.exists(directory):
            os.makedirs(directory)

        checkpoint_path = directory + "PPO_{}_{}_{}.pth".format(env_name, random_seed, run_num_pretrained)
        print("save checkpoint path : " + checkpoint_path)
        #####################################################

        print("training environment name : " + env_name)

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
            device_id=args.local_rank,
            n_actions = 0,
            descentstep = args.max_step,
        )
        return env

    envs = make_batch_env(num_envs=args.num_envs)
    dist.barrier()

    # state space dimension
    state_dim = envs.observation_space.shape[0]

    # action space dimension
    if has_continuous_action_space:
        action_dim = envs.action_shape
    else:
        action_dim = envs.action_space.n

    ###################### logging ######################
    if args.is_master:

        #### log files for multiple runs are NOT overwritten
        log_dir = "results/PPO/PPO_logs"
        if not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)

        log_dir = log_dir + '/' + env_name + '/'
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        #### get number of log files in log directory
        run_num = 0
        current_num_files = next(os.walk(log_dir))[2]
        run_num = len(current_num_files)

        #### create new log file for each run
        log_f_name = log_dir + '/PPO_' + env_name + "_log_" + str(run_num) + ".csv"

        print("current logging run number for " + env_name + " : ", run_num)
        print("logging at : " + log_f_name)
    #####################################################


    ############# print all hyperparameters #############
        print("--------------------------------------------------------------------------------------------")
        print("max training timesteps : ", max_training_timesteps)
        print("max timesteps per episode : ", max_ep_len)
        print("model saving frequency : " + str(save_model_freq) + " timesteps")
        print("log frequency : " + str(log_freq) + " timesteps")
        print("printing average reward over episodes in last : " + str(print_freq) + " timesteps")
        print("--------------------------------------------------------------------------------------------")
        print("state space dimension : ", state_dim)
        print("action space dimension : ", action_dim)
        print("--------------------------------------------------------------------------------------------")
        if has_continuous_action_space:
            print("Initializing a continuous action space policy")
            print("--------------------------------------------------------------------------------------------")
            print("starting std of action distribution : ", action_std)
            print("decay rate of std of action distribution : ", action_std_decay_rate)
            print("minimum std of action distribution : ", min_action_std)
            print("decay frequency of std of action distribution : " + str(action_std_decay_freq) + " timesteps")
        else:
            print("Initializing a discrete action space policy")
        print("--------------------------------------------------------------------------------------------")
        print("PPO update frequency : " + str(update_timestep) + " timesteps")
        print("PPO K epochs : ", K_epochs)
        print("PPO epsilon clip : ", eps_clip)
        print("discount factor (gamma) : ", gamma)
        print("--------------------------------------------------------------------------------------------")
        print("optimizer learning rate actor : ", lr_actor)
        print("optimizer learning rate critic : ", lr_critic)
        if random_seed:
            print("--------------------------------------------------------------------------------------------")
            print("setting random seed to ", random_seed)
            torch.manual_seed(random_seed)
            #envs.seed(random_seed)
            np.random.seed(random_seed)
        #####################################################

        print("============================================================================================")

    ################# training procedure ################

    # initialize a PPO agent
    ppo_agent = PPO(args, state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space, action_std)

    # track total training time
    start_time = datetime.now(pytz.timezone('Asia/Tokyo')).replace(microsecond=0)
    if args.is_master:
        print("Started training at (GMT) : ", start_time)

        print("============================================================================================")

        # logging file
        log_f = open(log_f_name,"w+")
        log_f.write('episode,timestep,reward\n')

        save_args(os.path.join(directory, 'args.json'), args)

    best_reward = -9999

    time_step = 0
    i_episode = 0

    # training loop
    while time_step <= max_training_timesteps:

        dist.barrier()
        state, rand  = envs.reset()#
        current_ep_reward = 0
        sketch_querys = ppo_agent.select_query(rand)

        for t in range(1, max_ep_len+1):
            # select action with policy
            action = ppo_agent.select_action(state, sketch_querys, t-1)
            state, reward, done, _ = envs.step(action)

            # saving reward and is_terminals
            ppo_agent.buffer.rewards.append(reward)
            ppo_agent.buffer.is_terminals.append(done)

            time_step +=1
            current_ep_reward += reward.mean()

            # update PPO agent
            if time_step % update_timestep == 0:
                ppo_agent.update()

            # if continuous action space; then decay action std of ouput action distribution
            if has_continuous_action_space and time_step % action_std_decay_freq == 0:
                ppo_agent.decay_action_std(action_std_decay_rate, min_action_std)

            # break; if the episode is over
            if done.all():
                break
        
        current_ep_reward = current_ep_reward.round(decimals=2)#torch.round(print_avg_reward, decimals=2)
        dist.all_reduce(current_ep_reward, op=dist.ReduceOp.SUM)
        current_ep_reward = current_ep_reward/_num_gpus

        if args.is_master:
            print("Episode : {} \t\t Timestep : {} \t\t Average Reward : {:.2f}".format(i_episode, time_step, current_ep_reward))

        #################### eval ####################
        if i_episode%eval_episode_interval==0 and i_episode > 0:
            dist.barrier()
            state, rand  = envs.reset()#
            eval_reward = 0
            sketch_querys = ppo_agent.select_query(rand)
            for t in range(1, max_ep_len+1):
                # select action with policy
                action = ppo_agent.select_action(state, sketch_querys, t-1)
                state, reward, done, _ = envs.step(action)

                # saving reward and is_terminals
                ppo_agent.buffer.rewards.append(reward)
                ppo_agent.buffer.is_terminals.append(done)

                eval_reward += reward.mean()
            ppo_agent.buffer.clear()
            dist.all_reduce(eval_reward, op=dist.ReduceOp.SUM)
            eval_reward = eval_reward/_num_gpus
            # save model weights
            if args.is_master:
                print("Episode : {} \t\t Timestep : {} \t\t Average Reward : {:.2f}".format(i_episode, time_step, eval_reward.round(decimals=2)))
                log_f.write('{},{},{}\n'.format(i_episode, time_step, eval_reward.round(decimals=2)))
                log_f.flush()
                log_plot(log_f_name, directory, env_name)

                if eval_reward > best_reward:
                    best_reward = eval_reward
                    checkpoint_path = directory + "PPO_best.pth"
                    print("--------------------------------------------------------------------------------------------")
                    print("saving model at : " + checkpoint_path)
                    ppo_agent.save(checkpoint_path)
                    print("best_model saved || score:{:.2f}".format(best_reward))
                    print("Elapsed Time  : ", datetime.now(pytz.timezone('Asia/Tokyo')).replace(microsecond=0) - start_time)
                    print("--------------------------------------------------------------------------------------------")
                if i_episode % save_model_freq == 0:
                    checkpoint_path = directory + "PPO_{}_{}_{}.pth".format(env_name, random_seed, time_step)
                    print("--------------------------------------------------------------------------------------------")
                    print("saving model at : " + checkpoint_path)
                    ppo_agent.save(checkpoint_path)
                    print("model saved")
                    print("Elapsed Time  : ", datetime.now(pytz.timezone('Asia/Tokyo')).replace(microsecond=0) - start_time)
                    print("--------------------------------------------------------------------------------------------")                    
        ##############################################

        i_episode += 1

    if args.is_master:
        log_f.close()

    # print total training time
    if args.is_master:
        print("============================================================================================")
        end_time = datetime.now().replace(microsecond=0)
        print("Started training at (GMT) : ", start_time)
        print("Finished training at (GMT) : ", end_time)
        print("Total training time  : ", end_time - start_time)
        print("============================================================================================")


if __name__ == '__main__':

    train()
    
    
    
    
    
    
    
