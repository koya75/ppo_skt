import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical
import numpy as np
from einops import rearrange
from einops.layers.torch import Rearrange
from agent.module.transformer import Transformer, TransformerEncoder, TransformerDecoder, TransformerEncoderLayer, TransformerDecoderLayer

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

class ActorCritic(nn.Module):
    def __init__(self, device, state_dim, action_dim, has_continuous_action_space, action_std_init):
        super(ActorCritic, self).__init__()
        print("PPO_SKT")
        print("============================================================================================")

        self.has_continuous_action_space = has_continuous_action_space
        self.device = device
        image_size = 128
        patch_size = 8
        channels = 3
        num_encoder_layers = 1
        num_decoder_layers = 1
        self.t = 0
        
        if has_continuous_action_space:
            self.action_dim = action_dim
            self.action_var = torch.full((action_dim,), action_std_init * action_std_init).to(self.device)
        # actor
        hidden_dim = 256
        if has_continuous_action_space :

            image_height, image_width = pair(image_size)#copy the number
            patch_height, patch_width = pair(patch_size)#copy the number
            assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.' 

            num_patches = (image_height // patch_height) * (image_width // patch_width)#how many patches
            patch_dim = channels * patch_height * patch_width#patch image to vector

            self.to_patch_embedding = nn.Sequential(
                Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width), #(h,w) patch number
                nn.Linear(patch_dim, hidden_dim),
                Rearrange('b n d -> n b d')
            )

            self.pos_embedding = nn.Parameter(torch.randn(num_patches, 1, hidden_dim))

            encoder_layer = TransformerEncoderLayer(hidden_dim, nhead=4, dim_feedforward=64,
                                                    dropout=0.1, activation="relu", normalize_before=False)
            self.image_transformer_encoder = TransformerEncoder(encoder_layer, num_encoder_layers)

            decoder_layer = TransformerDecoderLayer(hidden_dim, nhead=4, dim_feedforward=64,
                                                    dropout=0.1, activation="relu", normalize_before=False)
            decoder_norm = nn.LayerNorm(hidden_dim)
            self.transformer_decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm)

            self.actor = nn.Sequential(nn.Linear(hidden_dim*hidden_dim, action_dim),nn.Tanh())
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
    
    def act(self, state):#, skq

        if self.has_continuous_action_space:
            # construct positional encodings
            self.input_image = state * 1.0

            src = self.to_patch_embedding(state)

            n, bs, p = src.shape
            src += self.pos_embedding

            memory = self.image_transformer_encoder(src)

            #sketch_query = skq.repeat(1, bs, 1)
            #actor_out = self.transformer_decoder(sketch_query, memory)[0] # 3,bs,256
            action_mean = self.actor(memory.permute(1, 0, 2).flatten(1,2))

            cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)
            dist = MultivariateNormal(action_mean, cov_mat)
        else:
            action_probs = self.actor(state)
            dist = Categorical(action_probs)

        action = dist.sample()
        action_logprob = dist.log_prob(action)
        state_hidden = self.critic_conv(state)
        state_val = self.critic(state_hidden.flatten(1))
        self.t += 1

        return action.detach(), action_logprob.detach(), state_val.detach()
    
    def evaluate(self, state, action):#, skq

        if self.has_continuous_action_space:
            # construct positional encodings
            self.input_image = state * 1.0

            src = self.to_patch_embedding(state)

            n, bs, p = src.shape
            src += self.pos_embedding

            memory = self.image_transformer_encoder(src)

            #sketch_query = skq.unsqueeze(0)
            #actor_out = self.transformer_decoder(sketch_query, memory)[0] # 3,bs,256
            action_mean = self.actor(memory.permute(1, 0, 2).flatten(1,2))
            
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