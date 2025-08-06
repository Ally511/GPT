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
        self.n_head = n_head
        self.n_embd = n_embd

        # using scaled dot product instead of flash attention
        #not sure if we need this -> test it out
        """self.flash = hasattr(
                        torch.nn.functional, 
                        'scaled_dot_product_attention')
        if not self.flash:
            print(
              "WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence"""
        self.register_buffer(
            "mask", 
            torch.tril(torch.ones(block_size, block_size)
        ).view(1, 1, block_size, block_size))
        
        
        
    def forward(self,x):
       # B = batch_size, T = sequence_len, C = embedding_dim
       B,T,C = x.size()

       #calculate key, queries, values
       k, q, v = self.c_attention(x).split(self.n_embd, dim = 2)
       k = k.view(B, T, self.n_head, C //self.n_head).transpose(1,2)
       q = q.view(B, T, self.n_head, C //self.n_head).transpose(1,2)
       v = v.view(B, T, self.n_head, C //self.n_head).transpose(1,2) 

       #not sure if we're supposed to use this
       # 
       attention_mask = (q @ k.transpose(-2,-1))* (1.0 / math.sqrt(k.size(-1)))
       attention_mask = attention_mask.masked_fill(self.mask[:,:,:T,:T] == 0, float('-inf'))
       y = F.scaled_dot_product_attention(query=q,key=k,value=v, attn_mask = attention_mask)

       #instead manually this ? :

       """attention = (q @ k.transpose(-2,-1))* (1.0 / math.sqrt(k.size(-1)))
       #diagonal mask
       #fill 0 mask with super small numbers so it won't affect softmax
       attention = attention.masked_fill(self.mask[:,:,:T,:T] == 0, float('-inf'))
       attention = F.softmax(attention, dim = 1)
       attention = self.attention_dropout(attention)

       y = attention @ v"""

       y = y.transpose(1,2).contiguous().view(B,T,C)

       y = self.residual_dropout(self.c_projection(y))

       return y

