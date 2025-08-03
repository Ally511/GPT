import numpy as np
import torch as th
import torch.nn as nn
from torch.nn import functional as F

from decoder_block import Block

class GPT(nn.Module):
    # are we completely skipping optimizer, and initialising it in main, have it be passed to the trainer?
    # could also use very simple configure_optimizer() function that returns one?
    def __init__(self, config, device):
        super().__init__()
        self.block_size = config.block_size
        self.device = device

        # build transformer layers
        self.token_embd = nn.Embedding(config.vocab_size, config.n_embd)
        self.posit_embd = nn.Embedding(config.block_size, config.n_embd)
        self.dropout = nn.Dropout(config.embd_prdop)
        self.all_blocks = nn.ModuleList([Block(config.n_embd, config.resid_pdrop) for _ in range(config.n_layer)])
        self.layer_norm = nn.LayerNorm(config.n_embd)
        
        # add linear layer
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # weights initialisation can be skipped completely? I think it just takes the default

        self.count_params(nn.ModuleList([self.token_embd, self.posit_embd, self.dropout, self.all_blocks, self.layer_norm]))
    
    def count_params(self, layers):
        # no. parameters without lm_head
        n_params = 0

        for layer in layers:
            n_params += sum(param.numel() for param in layer.parameters())
        
        print("Number of parameters: %.2fM" % (n_params/1e6,))

    def configure_optimizer(self, params, train_config):
        optimizer = th.optim.AdamW(params=params, lr=train_config.learning_rate, betas=train_config.betas)
        return optimizer
    
    def transfomer_forward(self, idx, pos):
        """ forward pass through the transformer layers"""
        # forward the GPT model itself
        tok_embd = self.token_embd(idx) # shape (b, t, n_embd)
        pos_embd = self.posit_embd(pos) # (1, t, n_embd)
        
        x = self.dropout(tok_embd + pos_embd)
        for block in self.all_blocks:
            x = block(x)
        
        return self.layer_norm(x)

    def forward(self, idx, targets=None):
        # b=batch_size, t=chunk_size
        # set device
        device = self.device
        # get batch and chunk size from idx
        batch_size, chunk_size = idx.size()
        assert chunk_size <= self.block_size, f"Sequence length of {chunk_size} exceeds block size {self.block_size}."

        # get positional token, shape (1,t)
        pos = th.arange(0, chunk_size, dtype=th.long, device=device).unsqueeze(0)

        # forward GPT model itself
        x = self.transformer_forward(self, idx, pos)

        # (b, t, n_embd) -- > # (b, t, vocab_size)
        logits = self.lm_head(x)

        # calculate loss if targets
        loss = None
        if targets is not None:
            # compute cross entropy, ignore -1 at output
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        return logits, loss
    
    @th.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, sampling=False, top_k=None):
        ...