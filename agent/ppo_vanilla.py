import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical

class ActorCritic(nn.Module):
    def __init__(self, args, state_dim, action_dim, has_continuous_action_space, action_std_init):
        super(ActorCritic, self).__init__()
        if args.is_master:
            print("PPO_Vannilla")
            print("============================================================================================")

        self.has_continuous_action_space = has_continuous_action_space
        self.device = args.device
        
        if has_continuous_action_space:
            self.action_dim = action_dim
            self.action_var = torch.full((action_dim,), action_std_init * action_std_init).to(self.device)
        # actor
        hidden_dim = 256
        if has_continuous_action_space :

            self.actor_conv = nn.Sequential(
                            nn.Conv2d(3, 32, kernel_size=8, stride=4),
                            nn.ReLU(),
                            nn.Conv2d(32, 64, kernel_size=4, stride=2),
                            nn.ReLU(),
                            nn.Conv2d(64, hidden_dim, kernel_size=3, stride=1),
                            nn.ReLU()
                        )
            self.actor = nn.Sequential(
                            nn.Linear(hidden_dim*12*12, 64),
                            nn.Tanh(),
                            nn.Linear(64, 64),
                            nn.Tanh(),
                            nn.Linear(64, action_dim),
                            nn.Tanh()
                        )
        else:
            self.actor = nn.Sequential(
                            nn.Linear(state_dim, 64),
                            nn.Tanh(),
                            nn.Linear(64, 64),
                            nn.Tanh(),
                            nn.Linear(64, action_dim),
                            nn.Softmax(dim=-1)
                        )
        # critic
        self.critic_conv = nn.Sequential(
                        nn.Conv2d(3, 32, kernel_size=8, stride=4),
                        nn.ReLU(),
                        nn.Conv2d(32, 64, kernel_size=4, stride=2),
                        nn.ReLU(),
                        nn.Conv2d(64, hidden_dim, kernel_size=3, stride=1),
                        nn.ReLU()
                    )
        self.critic = nn.Sequential(
                        nn.Linear(hidden_dim*12*12, 64),
                        nn.Tanh(),
                        nn.Linear(64, 64),
                        nn.Tanh(),
                        nn.Linear(64, 1)
                    )
        
    def set_action_std(self, new_action_std):
        if self.has_continuous_action_space:
            self.action_var = torch.full((self.action_dim,), new_action_std * new_action_std).to(self.device)
        else:
            print("--------------------------------------------------------------------------------------------")
            print("WARNING : Calling ActorCritic::set_action_std() on discrete action space policy")
            print("--------------------------------------------------------------------------------------------")

    def forward(self):
        raise NotImplementedError
    
    def act(self, state):

        if self.has_continuous_action_space:
            action_hidden = self.actor_conv(state)
            action_mean = self.actor(action_hidden.flatten(1))
            cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)
            dist = MultivariateNormal(action_mean, cov_mat)
        else:
            action_probs = self.actor(state)
            dist = Categorical(action_probs)

        action = dist.sample()
        action_logprob = dist.log_prob(action)
        state_hidden = self.critic_conv(state)
        state_val = self.critic(state_hidden.flatten(1))

        return action.detach(), action_logprob.detach(), state_val.detach()
    
    def evaluate(self, state, action):

        if self.has_continuous_action_space:
            action_hidden = self.actor_conv(state)
            action_mean = self.actor(action_hidden.flatten(1))
            
            action_var = self.action_var.expand_as(action_mean)
            cov_mat = torch.diag_embed(action_var).to(self.device)
            dist = MultivariateNormal(action_mean, cov_mat)
            
            # For Single Action Environments.
            if self.action_dim == 1:
                action = action.reshape(-1, self.action_dim)
        else:
            action_probs = self.actor(state)
            dist = Categorical(action_probs)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_hidden = self.critic_conv(state)
        state_values = self.critic(state_hidden.flatten(1))
        
        return action_logprobs, state_values, dist_entropy