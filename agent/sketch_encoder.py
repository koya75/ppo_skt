import torch
import numpy as np
from torch import nn
import repos.pfrl.pfrl as pf
from einops import rearrange
from einops.layers.torch import Rearrange
import cv2
from agent.module.transformer import Transformer, TransformerEncoder, TransformerDecoder, TransformerEncoderLayer, TransformerDecoderLayer

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

class sketch_encoder(nn.Module):
    def __init__(
        self,
        device="cuda",
        image_size=128,
        patch_size=8,
        channels=3,
        hidden_dim = 256,
        num_encoder_layers=1,
        num_decoder_layers=1,
        seed=0,
    ):
        super().__init__()
        print("seg_sketch")
        img = create_mask("image/image_env_5_color_mask.png")
        self.seg_img = img.clone().to(device).to(torch.float32)

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
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_encoder_layers)

        decoder_layer = TransformerDecoderLayer(hidden_dim, nhead=4, dim_feedforward=64,
                                                dropout=0.1, activation="relu", normalize_before=False)
        decoder_norm = nn.LayerNorm(hidden_dim)
        self.transformer_decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm)

        g_point = torch.from_numpy(np.loadtxt('image/green6.csv', delimiter=",")).clone().to(device).to(torch.float32)
        self.g_point = g_point.reshape([10, 4, 2],-1).flatten(1,2)
        self.sketch_encoder = nn.Linear(self.g_point.shape[1], hidden_dim)
        self.sketch_pos_embedding = nn.Parameter(torch.randn(self.g_point.shape[0],  1, hidden_dim))
        
        self._reset_parameters()
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def rich_sketch(self):
        src = self.to_patch_embedding(self.seg_img)
        
        n, bs, p = src.shape
        src += self.pos_embedding

        seq_memory = self.transformer_encoder(src)
        
        # sketch_transformer encoder
        sketch_query = self.sketch_encoder(self.g_point).unsqueeze(1).repeat(1, bs, 1)
        sketch_query += self.sketch_pos_embedding
        
        out = self.transformer_decoder(sketch_query, seq_memory)[0] # 10,bs,256

        return out

    def forward(self, flag=None):
        if flag is not None:
            if np.any(~flag): # can not partial envs reset.
                self.out = self.rich_sketch()
        else:
            self.out = self.rich_sketch()
        return self.out