import torch
import torch.nn as nn
import torch.distributed as dist
import time
from utils import save_checkpoint, load_checkpoint

################################## PPO Policy ##################################
class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.state_values = []
        self.is_terminals = []
        self.random = []
    
    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.state_values[:]
        del self.is_terminals[:]
        del self.random[:]

class PPO:
    def __init__(self, args, state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space, action_std_init=0.6):

        ################################## set device ##################################
        dist.barrier()
        if args.is_master:
            print("============================================================================================")
        # set device to cpu or cuda
        self.device = torch.device('cpu')
        if(torch.cuda.is_available()): 
            self.device = torch.device('cuda:{}'.format(args.local_rank))
            torch.cuda.empty_cache()
            print("Device set to : " + str(torch.cuda.get_device_name(self.device)))
        else:
            print("Device set to : cpu")
        print("============================================================================================")

        self.has_continuous_action_space = has_continuous_action_space
        self.num_envs = args.num_envs

        if has_continuous_action_space:
            self.action_std = action_std_init

        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        
        self.buffer = RolloutBuffer()

        if args.model == "vanilla":
            from agent.ppo_vanilla import ActorCritic
            policy = ActorCritic(self.device, state_dim, action_dim, has_continuous_action_space, action_std_init).to(self.device)
            ### distributed
            self.policy = torch.nn.parallel.DistributedDataParallel(
                policy, device_ids=[args.local_rank], output_device=args.local_rank
            )
            self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=args.lr)

            policy_old = ActorCritic(self.device, state_dim, action_dim, has_continuous_action_space, action_std_init).to(self.device)
            ### distributed
            self.policy_old = torch.nn.parallel.DistributedDataParallel(
                policy_old, device_ids=[args.local_rank], output_device=args.local_rank
            )

        elif args.model == "skt":
            from agent.ppo_sketch_transformer import ActorCritic
            policy = ActorCritic(args.task, self.device, state_dim, action_dim, has_continuous_action_space, action_std_init).to(self.device)
            ### distributed
            self.policy = torch.nn.parallel.DistributedDataParallel(
                policy, device_ids=[args.local_rank], output_device=args.local_rank
            )
            self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=args.lr)

            policy_old = ActorCritic(args.task, self.device, state_dim, action_dim, has_continuous_action_space, action_std_init).to(self.device)
            ### distributed
            self.policy_old = torch.nn.parallel.DistributedDataParallel(
                policy_old, device_ids=[args.local_rank], output_device=args.local_rank
            )
        else:
            from agent.ppo_anymal import ActorCritic
            policy = ActorCritic(self.device, state_dim, action_dim, has_continuous_action_space, action_std_init).to(self.device)
            ### distributed
            self.policy = torch.nn.parallel.DistributedDataParallel(
                policy, device_ids=[args.local_rank], output_device=args.local_rank
            )
            self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=args.lr)

            policy_old = ActorCritic(self.device, state_dim, action_dim, has_continuous_action_space, action_std_init).to(self.device)
            ### distributed
            self.policy_old = torch.nn.parallel.DistributedDataParallel(
                policy_old, device_ids=[args.local_rank], output_device=args.local_rank
            )
            self.sketch_encoder=None
            self.optimizer2=None

        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.MseLoss = nn.MSELoss().to(self.device)

    def set_action_std(self, new_action_std):
        if self.has_continuous_action_space:
            self.action_std = new_action_std
            self.policy.module.set_action_std(new_action_std)
            self.policy_old.module.set_action_std(new_action_std)
        else:
            print("--------------------------------------------------------------------------------------------")
            print("WARNING : Calling PPO::set_action_std() on discrete action space policy")
            print("--------------------------------------------------------------------------------------------")

    def decay_action_std(self, action_std_decay_rate, min_action_std):
        print("--------------------------------------------------------------------------------------------")
        if self.has_continuous_action_space:
            self.action_std = self.action_std - action_std_decay_rate
            self.action_std = round(self.action_std, 4)
            if (self.action_std <= min_action_std):
                self.action_std = min_action_std
                print("setting actor output action_std to min_action_std : ", self.action_std)
            else:
                print("setting actor output action_std to : ", self.action_std)
            self.set_action_std(self.action_std)

        else:
            print("WARNING : Calling PPO::decay_action_std() on discrete action space policy")
        print("--------------------------------------------------------------------------------------------")

    def select_query(self, rand):
        sketch_query = self.policy_old.module.create_query(rand)
        self.buffer.random.append(rand)

        return sketch_query

    def select_action(self, state, skq, t):#

        if self.has_continuous_action_space:
            with torch.no_grad():
                state = state.to(self.device)#torch.FloatTensor()
                action, action_logprob, state_val = self.policy_old.module.act(state, skq, t)#

            self.buffer.states.append(state)
            self.buffer.actions.append(action)
            self.buffer.logprobs.append(action_logprob)
            self.buffer.state_values.append(state_val)

            return action.detach()#.cpu().numpy()#.flatten()
        else:
            with torch.no_grad():
                state = state.to(self.device)#torch.FloatTensor()
                action, action_logprob, state_val = self.policy_old.module.act(state)
            
            self.buffer.states.append(state)
            self.buffer.actions.append(action)
            self.buffer.logprobs.append(action_logprob)
            self.buffer.state_values.append(state_val)

            return action.item()

    def update(self):
        # Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal.all():
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
            
        # Normalizing the rewards
        rewards = torch.stack(rewards, dim=0)#, dtype=torch.float32, device=self.device
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # convert list to tensor
        old_states, old_actions, old_logprobs, old_state_values, old_randoms, rewards = self.convert_list_to_tensor(rewards)

        # calculate advantages
        advantages = rewards.detach() - old_state_values.detach()

        # Optimize policy for K epochs
        for _ in range(self.K_epochs):

            old_sketch_querys = self.policy.module.create_batch_query(old_randoms)
            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.policy.module.evaluate(old_states, old_actions, old_sketch_querys)#

            # match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)
            
            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss  
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages

            # final loss of clipped objective PPO
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values.squeeze(), rewards.squeeze()) - 0.01 * dist_entropy
            
            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            
        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # clear buffer
        self.buffer.clear()
        
    
    def save(self, checkpoint_path):
        save_checkpoint(checkpoint_path, self.policy_old, self.optimizer)
   
    def load(self, checkpoint_path):
        load_checkpoint(checkpoint_path, self.policy, self.optimizer, self.policy_old, self.device)
        
    def convert_list_to_tensor(self, reward):
        old_state = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(self.device)
        old_action = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(self.device)
        old_logprob = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(self.device)
        old_state_value = torch.squeeze(torch.stack(self.buffer.state_values, dim=0)).detach().to(self.device)
        old_random = torch.stack(self.buffer.random, dim=0).detach().to(self.device)

        old_states, old_actions, old_logprobs, old_state_values, old_randoms, rewards = [],[],[],[],[],[]

        for i in range(self.num_envs):
            old_states.append(old_state[:,i])
            old_actions.append(old_action[:,i])
            old_logprobs.append(old_logprob[:,i])
            old_state_values.append(old_state_value[:, i])
            old_randoms.append(old_random[:,i])
            rewards.append(reward[:,i])
        old_states = torch.cat(old_states, dim=0)
        old_actions = torch.cat(old_actions, dim=0)
        old_logprobs = torch.cat(old_logprobs, dim=0)
        old_state_values = torch.cat(old_state_values, dim=0)
        old_randoms = torch.cat(old_randoms, dim=0)
        rewards = torch.cat(rewards, dim=0)

        return old_states, old_actions, old_logprobs, old_state_values, old_randoms, rewards


