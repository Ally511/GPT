import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from torch.nn import Module

class Causal_self_attention(nn.Module):
    """Causal Self Attention class that defines masked multi-head attention
    in Transformer Decoder Blocks"""
    def __init__(self,n_embd,n_head,dropout_rate,attn_pdrop,resid_pdrop,block_size):
        super().__init__()
        assert n_embd % n_head == 0
        #projections into key, queries and values for all heads
        self.c_attention = nn.Linear(n_embd, 3* n_embd)
        #output projection
        self.c_projection = nn.Linear(n_embd, n_embd)

        #dropouts
        self.attention_dropout = nn.Dropout(attn_pdrop)
        self.residual_dropout = nn.Dropout(resid_pdrop)
        self.dropout_rate = dropout_rate

        #number of heads and number of embeddings
        self.n_head = n_head
        self.n_embd = n_embd

    def forward(self,x):
       # B = batch_size, T = sequence_len, C = embedding_dim
       B,T,C = x.size()

       #calculate key, queries, values
       k, q, v = self.c_attention(x).split(self.n_embd, dim = 2)
       k = k.view(B, T, self.n_head, C //self.n_head).transpose(1,2)
       q = q.view(B, T, self.n_head, C //self.n_head).transpose(1,2)
       v = v.view(B, T, self.n_head, C //self.n_head).transpose(1,2)

       #scaled dot-product attention
       scale = 1.0 / math.sqrt(k.size(-1))
       attention = (q @ k.transpose(-2,-1)) * scale

       #apply a causal mask(to mask out future tokens)
       causal_mask = torch.triu(torch.ones(T,T,), diagonal=1).bool()
       masked_attention = attention.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0),float('-inf'))   

       #apply softmax to last dimension
       attention_weights = F.softmax(masked_attention, dim=-1)       

       #dropout of attention weights
       attention_weights = self.attention_dropout(attention_weights)

       #compute attention weights * values = output
       output = attention_weights @ v

       #recombine heads back to single vector
       output = output.transpose(1,2).contiguous().view(B,T,C)

       #output projections and residual dropout
       output = self.residual_dropout(self.c_projection(output))
       
       return output

