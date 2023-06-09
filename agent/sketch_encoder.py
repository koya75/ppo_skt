import torch
import torch.nn as nn
import numpy as np
from agent.module.randomization import MyRandomization
from agent.module.transformer import TransformerEncoder, TransformerEncoderLayer

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

class Sketch_Encoder(nn.Module):
    def __init__(
        self,
        device="cuda",
        hidden_dim = 256,
        num_encoder_layers=1,
        num_decoder_layers=1,
        seed=0,
    ):
        super(Sketch_Encoder, self).__init__()
        print("seg_sketch")
        self.sketch_token = MyRandomization(device)

        self.sketch_encoder = nn.Linear(self.sketch_token.pattern1.shape[1], hidden_dim)
        self.sketch_pos_embedding = nn.Parameter(torch.randn(self.sketch_token.pattern1.shape[0], 1, hidden_dim))

        encoder_layer = TransformerEncoderLayer(hidden_dim, nhead=4, dim_feedforward=64,
                                                dropout=0.1, activation="relu", normalize_before=False)
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_encoder_layers)
        
        self._reset_parameters()
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self):
        raise NotImplementedError

    def create_query(self, rand):
        # sketch_transformer encoder
        sketch_query = self.sketch_encoder(self.sketch_token.select(rand)).unsqueeze(1)
        sketch_query += self.sketch_pos_embedding
        
        out = self.transformer_encoder(sketch_query) # 10,bs,256

        return out

    def create_batch_query(self, rand):
        # sketch_transformer encoder
        sketch_query = self.sketch_encoder(self.sketch_token.batch_select(rand)).permute(1, 0, 2)
        sketch_query += self.sketch_pos_embedding
        
        out = self.transformer_encoder(sketch_query).flatten(0,1) # 10,bs,256

        return out