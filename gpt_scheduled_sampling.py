import numpy as np
import torch as th
import torch.nn as nn
from torch.nn import functional as F

from decoder_block import Block

class GPT(nn.Module):
    def __init__(self, config, device):
        super().__init__()
        self.block_size = config.block_size
        self.device = device

        # build transformer layers
        self.token_embd = nn.Embedding(config.vocab_size, config.n_embd)
        self.posit_embd = nn.Embedding(config.block_size, config.n_embd)
        self.dropout = nn.Dropout(config.embd_pdrop)
        self.all_blocks = nn.ModuleList([Block(config.n_embd, config.dropout, config.attn_pdrop, config.resid_pdrop, config.n_head, config.block_size) for _ in range(config.n_layer)])
        self.layer_norm = nn.LayerNorm(config.n_embd)
        
        # add linear layer
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # weights initialisation can be skipped completely? I think it just takes the default
        trans_layers = nn.ModuleList([self.token_embd, self.posit_embd, self.dropout, self.all_blocks, self.layer_norm])
        self.optimizer = self.configure_optimizer(trans_layers, config)
        self.count_params(trans_layers)
    
    def count_params(self, layers):
        # no. parameters without lm_head
        n_params = 0

        for layer in layers:
            n_params += sum(param.numel() for param in layer.parameters())
        
        print("Number of parameters: %.2fM" % (n_params/1e6,))

    def configure_optimizer(self, layers, train_config):
        params = []
        for layer in layers:
            for param in layer.parameters():
                params.append(param)

        optimizer = th.optim.AdamW(params=params, lr=train_config.learning_rate, betas=train_config.betas)
        return optimizer
    
    def transformer_forward(self, idx, pos):
        """ forward pass through the transformer layers"""
        # forward the GPT model itself
        tok_embd = self.token_embd(idx) # shape (b, t, n_embd)
        pos_embd = self.posit_embd(pos) # (1, t, n_embd)
        
        x = self.dropout(tok_embd + pos_embd)
        for block in self.all_blocks:
            x = block(x)
        
        return self.layer_norm(x)

    def forward(self, idx, targets=None):
        """
        Annealed teacher forcing / scheduled sampling:
            Currently takes (batch_size, chunk_size) tensors, and can process all tokens at once
            --> Would need sequential decoding, sacrifing training efficiency
            --> Build input sequence step by step, for each example in a batch
                At each position, with probability p, take ground-truth next token; with 1-p sample from model's predicted distribution
                Embed sequence so far, run model forward the next step, repeat
        """
        # b=batch_size, t=chunk_size
        # set device

        device = self.device
        # get batch and chunk size from idx
        batch_size, chunk_size = idx.size()
        assert chunk_size <= self.block_size, f"Sequence length of {chunk_size} exceeds block size {self.block_size}."

        # get positional token, shape (1,t)
        pos = th.arange(0, chunk_size, dtype=th.long, device=device).unsqueeze(0)
        # forward GPT model itself
        x = self.transformer_forward(idx, pos)

        # (b, t, n_embd) --> (b, t, vocab_size)
        logits = self.lm_head(x)

        # calculate loss if targets given
        loss = None
        if targets is not None:
            # compute cross entropy, ignore -1 at output
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        return logits, loss
    
    def anneal_teacher(self, ep, total_train):
        k = total_train / 3 # large k: slower decay, small k: faster decay
        ep_prob = k / (k + np.exp(ep/k))
        return ep_prob
    
    def forward_annealed(self, idx, epoch, total_train_epochs, targets=None, temp=1.0, sampling=False, top_k=None):
        device = self.device

        teacher_prob = self.anneal_teacher(epoch, total_train_epochs)
        batch_size, chunk_size = idx.size()
        assert chunk_size <= self.block_size, f"Sequence length of {chunk_size} exceeds block size {self.block_size}."
        
        loss = None
        inputs = idx[:, :1] # get start tokens for batch from idx, after shape (B,1)
        for t in range(chunk_size):
            logits, _ = self(inputs)

            # logits at final step, scale by temp
            logits = logits[:, -1, :] / temp # (batch, vocab_size)

            # accumulate loss, etc.
            # calculate loss if targets given
            if targets is not None:
                target_token = targets[:, t] # (batch, )
                # compute cross entropy, ignore -1 at output
                loss = F.cross_entropy(logits, target_token, ignore_index=-1)

            # if top_k, crop logits
            if top_k is not None:
                v, _ = th.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float('Inf')

            # normalise logit probabilities
            soft_logits = F.softmax(logits, dim=-1)

            # sample from distribution or argmax
            if sampling:
                student_tokens = th.multinomial(soft_logits, num_samples=1) # (b, 1)
            else:
                _, student_tokens = th.topk(soft_logits, k=1, dim=-1) # (b, 1)

            teacher_tokens = targets[:, t].unsqueeze(1) # (b, 1)
            use_teacher = th.rand(batch_size, 1, device=device) < teacher_prob
            next = th.where(use_teacher, teacher_tokens, student_tokens)
            concat_inputs = th.cat([inputs, next], dim=1)  # inputs stays a tensor of size (B, curr_seq)
            
        return concat_inputs, loss

    
    @th.no_grad()
    def generate(self, idx, max_new_tokens, temp=1.0, sampling=False, top_k=None):
        """
        Args:
            idx (LongTensor): sequence of indices, shape (b,t)
            max_new_tokens (int): maximum amount of tokens to generate
            temp (float): temperature to scale probabilities of logits by
            sampling (boolean): boolean controlling whether next token is sampled from distribution or argmax
            top_k (int): if not None, only top k tokens are used in sampling
        """
        for _ in range(max_new_tokens):
            # cut off at block-size
            cropped_idx = idx if idx.size(1) <= self.block_size else idx[:, -self.block_size:]
            # forward model, get logits for index
            logits, _ = self(cropped_idx)
            # logits at final step, scale by temp
            logits = logits[:, -1, :] / temp

            # if top_k, crop logits
            if top_k is not None:
                v, _ = th.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float('Inf')

            # normalise logit probabilities
            soft_logits = F.softmax(logits, dim=-1)

            # sample from distribution or argmax
            if sampling:
                next_idx = th.multinomial(soft_logits, num_samples=1)
            else:
                _, next_idx = th.topk(soft_logits, k=1, dim=-1)

            idx = th.cat((idx, next_idx), dim=1)

        return idx

