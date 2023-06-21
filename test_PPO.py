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
        print("============================================================================================")
        # set device to cpu or cuda
        self.device = torch.device('cpu')
        if(torch.cuda.is_available()): 
            self.device = torch.device('cuda:{}'.format(args['gpu']))
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

        if args['model'] == "vanilla":
            from agent.ppo_vanilla import ActorCritic
            self.policy = ActorCritic(self.device, state_dim, action_dim, has_continuous_action_space, action_std_init).to(self.device)

            self.policy_old = ActorCritic(self.device, state_dim, action_dim, has_continuous_action_space, action_std_init).to(self.device)
        elif args['model'] == "skt":
            from agent.ppo_sketch_transformer import ActorCritic
            policy = ActorCritic(args['task'], self.device, state_dim, action_dim, has_continuous_action_space, action_std_init)
            ### distributed
            self.policy = nn.DataParallel(policy).to(self.device)

            policy_old = ActorCritic(args['task'], self.device, state_dim, action_dim, has_continuous_action_space, action_std_init)
            ### distributed
            self.policy_old = nn.DataParallel(policy_old).to(self.device)

        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.MseLoss = nn.MSELoss().to(self.device)

    def select_query(self, rand):
        enc_attn_weights = []
        hooks = [
            self.policy_old.module.transformer_encoder.layers[-1].self_attn.register_forward_hook(
                lambda self, input, output: enc_attn_weights.append(output[1])
            ),
        ]

        sketch_query = self.policy_old.module.create_query(rand)

        for hook in hooks:
            hook.remove()

        self.buffer.random.append(rand)

        enc_attn_weights = enc_attn_weights[0]

        return sketch_query, enc_attn_weights

    def select_action(self, state, skq, t):

        enc_attn_weights, dec_attn_weights = [], []
        # Last depth Attention
        hooks = [
            self.policy_old.module.image_transformer_encoder.layers[-1].self_attn.register_forward_hook(
                lambda self, input, output: enc_attn_weights.append(output[1])
            ),
            self.policy_old.module.transformer_decoder.layers[-1].multihead_attn.register_forward_hook(
                lambda self, input, output: dec_attn_weights.append(output[1])
            ),
        ]
        if self.has_continuous_action_space:
            with torch.no_grad():
                state = state.to(self.device)#torch.FloatTensor()
                action, action_logprob, state_val = self.policy_old.module.act(state, skq, t)

            self.buffer.states.append(state)
            self.buffer.actions.append(action)
            self.buffer.logprobs.append(action_logprob)
            self.buffer.state_values.append(state_val)

            for hook in hooks:
                hook.remove()
            
            raw_img = self.policy_old.module.input_image
            enc_attn_weights = enc_attn_weights[0]
            dec_attn_weights = dec_attn_weights[0]

            return action.detach(), raw_img, enc_attn_weights, dec_attn_weights#.cpu().numpy()
        else:
            with torch.no_grad():
                state = state.to(self.device)#torch.FloatTensor()
                action, action_logprob, state_val = self.policy_old.act(state)
            
            self.buffer.states.append(state)
            self.buffer.actions.append(action)
            self.buffer.logprobs.append(action_logprob)
            self.buffer.state_values.append(state_val)

            return action.item()
        
    def save(self, checkpoint_path):
        save_checkpoint(checkpoint_path, self.policy_old, self.optimizer)
   
    def load(self, checkpoint_path):
        load_checkpoint(checkpoint_path, self.policy, self.policy_old, None, self.device)
        
        
       


