import torch as th
import numpy as np

# Dummy config object

class GPTConfig:
    def __init__(self, vocab_size, **kwargs):
        self.vocab_size = vocab_size
        for key, value in kwargs.items():
            setattr(self, key, value)

class CustomConfig(GPTConfig):
    n_layer = 8
    n_head = 4
    n_embd = 256
    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1
    dropout = 0.1
    compile = True
    device = 'cpu'
    num_workers = 0
    max_iters = 2e4
    batch_size = 32
    block_size = 64
    learning_rate = 3e-4
    betas = (0.9, 0.95)
    weight_decay = 1e-1
    grad_norm_clip = 1.0

# Fake Block stub (replace this with your real decoder block!)
import torch.nn as nn
class Block(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.net = nn.Identity()
    def forward(self, x):
        return self.net(x)

from gpt.gpt_scheduled_sampling import GPT 

# Instantiate model
device = "cpu"
config = CustomConfig(vocab_size=2000)
model = GPT(config, device)

# Fake data: batch of indices
BATCH_SIZE = 3
SEQ_LEN = 5
vocab_size = config.vocab_size

# random indices in [0, vocab_size)
idx = th.randint(0, vocab_size, (BATCH_SIZE, SEQ_LEN))
targets = th.randint(0, vocab_size, (BATCH_SIZE, SEQ_LEN))

# Call forward_annealed
EPOCH = 2
TOTAL_EPOCHS = 10
LOSS = None   # not used initially

inputs, loss = model.forward_annealed(idx, EPOCH, TOTAL_EPOCHS, targets, temp=1.0, sampling=True)

print("\nFinal inputs shape:", type(inputs), len(inputs) if isinstance(inputs, list) else inputs.shape)
if loss is not None:
    print("Final loss:", loss.item())