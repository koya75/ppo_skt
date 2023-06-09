import torch
import torch.nn as nn
import torch.distributed as dist
import time

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
        print("============================================================================================")
        # set device to cpu or cuda
        self.device = torch.device('cpu')
        if(torch.cuda.is_available()): 
            self.device = torch.device('cuda:{}'.format(args.gpu))
            torch.cuda.empty_cache()
            print("Device set to : " + str(torch.cuda.get_device_name(self.device)))
        else:
            print("Device set to : cpu")
        print("============================================================================================")

        self.has_continuous_action_space = has_continuous_action_space

        if has_continuous_action_space:
            self.action_std = action_std_init

        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        
        self.buffer = RolloutBuffer()

        if args.model == "vanilla":
            from agent.ppo_vanilla import ActorCritic
            self.policy = ActorCritic(args, state_dim, action_dim, has_continuous_action_space, action_std_init).to(self.device)
            self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=args.lr)

            self.policy_old = ActorCritic(args, state_dim, action_dim, has_continuous_action_space, action_std_init).to(self.device)
        elif args.model == "skt":
            from agent.ppo_sketch_transformer import ActorCritic
            from agent.sketch_encoder import Sketch_Encoder
            self.sketch_encoder = Sketch_Encoder(self.device).to(self.device)
            self.policy = ActorCritic(args, state_dim, action_dim, has_continuous_action_space, action_std_init).to(self.device)
            self.optimizer1 = torch.optim.Adam(self.sketch_encoder.parameters(), lr=args.lr)
            self.optimizer2 = torch.optim.Adam(self.policy.parameters(), lr=args.lr)

            self.policy_old = ActorCritic(args, state_dim, action_dim, has_continuous_action_space, action_std_init).to(self.device)

        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.MseLoss = nn.MSELoss().to(self.device)

    def set_action_std(self, new_action_std):
        if self.has_continuous_action_space:
            self.action_std = new_action_std
            self.policy.set_action_std(new_action_std)
            self.policy_old.set_action_std(new_action_std)
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
        sketch_query = self.sketch_encoder.create_query(rand)
        self.buffer.random.append(rand)

        return sketch_query

    def select_action(self, state, skq):

        if self.has_continuous_action_space:
            with torch.no_grad():
                state = state.to(self.device)#torch.FloatTensor()
                action, action_logprob, state_val = self.policy_old.act(state, skq)

            self.buffer.states.append(state)
            self.buffer.actions.append(action)
            self.buffer.logprobs.append(action_logprob)
            self.buffer.state_values.append(state_val)

            return action.detach().cpu().numpy()#.flatten()
        else:
            with torch.no_grad():
                state = state.to(self.device)#torch.FloatTensor()
                action, action_logprob, state_val = self.policy_old.act(state)
            
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
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # convert list to tensor
        old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(self.device)
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(self.device)
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(self.device)
        old_state_values = torch.squeeze(torch.stack(self.buffer.state_values, dim=0)).detach().to(self.device)
        old_randoms = torch.squeeze(torch.stack(self.buffer.random, dim=0)).detach().to(self.device)


        # calculate advantages
        advantages = rewards.detach() - old_state_values.detach()

        # Optimize policy for K epochs
        for _ in range(self.K_epochs):

            old_sketch_querys = self.sketch_encoder.create_batch_query(old_randoms)
            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions, old_sketch_querys)

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
            self.optimizer1.zero_grad()
            self.optimizer2.zero_grad()
            loss.mean().backward()
            self.optimizer1.step()
            self.optimizer2.step()
            
        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # clear buffer
        self.buffer.clear()
    
    def save(self, checkpoint_path):
        torch.save({'optimizer1_state_dict': self.optimizer1.state_dict(),
                    'policy_old_state_dict': self.policy_old.state_dict(),
                    'optimizer2_state_dict': self.optimizer2.state_dict(),
                    'sketch_encoder_state_dict': self.sketch_encoder.state_dict()}, checkpoint_path)
   
    def load(self, checkpoint_path, device=None):
        ### load checkpoint
        if device is not None:
            _ckpt = torch.load(checkpoint_path, map_location=torch.device(device))
        else:
            _ckpt = torch.load(checkpoint_path)

        ### load state dicts
        self.policy_old.load_state_dict(_ckpt['policy_old_state_dict'])
        self.policy.load_state_dict(_ckpt['policy_old_state_dict'])
        self.optimizer1.load_state_dict(_ckpt['optimizer1_state_dict'])
        self.sketch_encoder.load_state_dict(_ckpt['sketch_encoder_state_dict'])
        self.optimizer2.load_state_dict(_ckpt['optimizer2_state_dict'])
        
        #self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        #self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        
        
       


