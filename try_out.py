import torch
# from gpt import GPT
from gpt_scheduled_sampling import GPT
from trainer import Trainer
from utility_functions import get_batch, decode_characters
import ast
import numpy as np
import matplotlib.pyplot as plt


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
    device = 'cuda'
    num_workers = 0
    max_iters = 2e4
    batch_size = 32
    block_size = 64
    learning_rate = 3e-4
    betas = (0.9, 0.95)
    weight_decay = 1e-1
    grad_norm_clip = 1.0

with open('corpora/vocab_train.txt', 'r', encoding='utf-8') as f:
    vocab = eval(f.read())

with open('corpora/indices_text.txt', 'r') as f:
    train_dataset = f.read()

with open('corpora/indices_text_val.txt', 'r') as f:
    validation_set = f.read()


train_dataset = ast.literal_eval(train_dataset)
validation_set = ast.literal_eval(validation_set)
device = 'cpu'
vocab_size = len(vocab)
print(vocab_size, "ln: ", np.log(vocab_size))
config = CustomConfig(vocab_size=vocab_size)
our_gpt = GPT(config=config, device=device)



our_trainer = Trainer(our_gpt, train_dataset, vocab, device, val_dataset=validation_set)
epochs = 1
train_steps = 6500

# xbatch, _ = get_batch(train_dataset, 1, 128)
#
# xbatch = torch.tensor(xbatch, dtype=torch.long, device=device)
xbatch = torch.tensor([[0]], dtype=torch.long).to(device)
loss = our_trainer.run(epochs, train_steps, config.batch_size, config.block_size)
generated = our_gpt.generate(xbatch, 100, 0.8, True, 20)
generated = generated[0].tolist()
decoded = decode_characters(generated, vocab)
print(decoded)

# Might be all we need for perplexity
losses = np.array(loss)
perplexities = np.exp(losses)

# Plot perplexity
y = np.arange(1, len(perplexities)+1)
plt.plot(y, perplexities)
plt.xlabel('Training Steps')
plt.ylabel('Perplexity')
plt.title('Perplexity of the GPT Model over Time')
plt.savefig('perplexity_scheduled_sampling.png')
plt.show()


y = np.arange(1, len(loss)+1)
plt.plot(y, loss)
plt.xlabel('Training Steps')
plt.ylabel('Loss')
plt.title('Loss of the GPT Model over Time')
plt.savefig('loss_scheduled sampling.png')
plt.show()

with open('output_scheduled_sampling.txt', 'w') as f:
    f.write(decoded)


