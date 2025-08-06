import torch
from gpt import GPT
from trainer import Trainer
from utility_functions import get_batch, decode_characters
import ast


class GPTConfig:
    def __init__(self, vocab_size, **kwargs):
        self.vocab_size = vocab_size
        for key, value in kwargs.items():
            setattr(self, key, value)

class CustomConfig(GPTConfig):
    n_layer = 8
    n_head = 8
    n_embd = 256
    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1
    dropout = 0.1
    compile = True
    device = 'cuda'
    num_workers = 0
    max_iters = 2e4
    batch_size = 4
    block_size = 64
    learning_rate = 6e-4
    betas = (0.9, 0.95)
    weight_decay = 1e-1
    grad_norm_clip = 1.0

with open('corpora/vocab_train.txt', 'r', encoding='utf-8') as f:
    vocab = f.read()

with open('corpora/indices_text.txt', 'r') as f:
    train_dataset = f.read()


train_dataset = ast.literal_eval(train_dataset)
device = 'cpu'
vocab_size = len(vocab)
config = CustomConfig(vocab_size=vocab_size)
our_gpt = GPT(config=config, device=device)



our_trainer = Trainer(our_gpt, train_dataset, device)
epochs = 1
train_steps = 100

xbatch, _ = get_batch(train_dataset, 1, 4)

xbatch = torch.tensor(xbatch, dtype=torch.long, device=device)

# our_trainer.run(epochs, train_steps, config.batch_size, config.block_size)
generated = our_gpt.generate(xbatch, 1000, 1.0, True)
generated = generated[0].tolist()
decoded = decode_characters(generated, vocab)
print(decoded)


