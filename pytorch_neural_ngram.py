import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np

block_size = 8
batch_size = 128

def get_batch(split):

    data = train_data if split =='train' else val_data
    ix = torch.randint(len(data)-block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])

    return x, y

class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        # self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)
        # ensure the initial embedding is small and centered around 0
        limit = 1 / np.sqrt(vocab_size)
        self.token_embedding_table = (
                (np.random.rand(vocab_size, vocab_size).astype(np.float32) * 2 - 1) * limit
        )
        # self.token_embedding_table = np.random.rand(vocab_size, vocab_size).astype(np.float32)
        # In ein torch.nn.Parameter überführen
        self.embedding_param = torch.nn.Parameter(torch.from_numpy(self.token_embedding_table))
        self.embedding_param.data = self.embedding_param.data.float()
        self.optimizer = torch.optim.AdamW([self.embedding_param], lr=1e-3)
        self.vocab_size = vocab_size

    def calculate_softmax(self, x):
        """Takes input array x and returns softmax."""
        exp_x = torch.exp(x - torch.max(x, dim=-1, keepdim=True).values)
        return exp_x / torch.sum(exp_x, dim=-1, keepdim=True)


    def calculate_cross_entropy(self, y_hatless, y_hat):
        """
        Takes target (y_hatless) and prediction (y_hat) and computes cross entropy loss.
        """
        # get vocab_size
        _, _, vocab_size = y_hat.shape
        y_hat = y_hat.reshape(y_hat.shape[0] * y_hat.shape[1], y_hat.shape[2])
        y_hatless_flat = y_hatless.reshape(-1).long()
        # one-hot encode targets
        y_hatless_hot = torch.eye(vocab_size)[y_hatless_flat]

        #y_hat = self.calculate_softmax(y_hat)

        # y_hat = torch.log_softmax(y_hat.float(), dim=1)
        y_hat = self.calculate_softmax(y_hat)
        y_hat = torch.log(y_hat)

        return torch.mean(-torch.sum(y_hatless_hot * y_hat, dim=1))

    def forward(self, idx, targets=None):
        # logits = self.token_embedding_table(idx)
        logits = self.embedding_param[idx]



        if targets is not None:
            B, T, C = logits.shape
            # logits = logits.view(B*T, C)
            # targets = targets.view(B*T)
            # loss = F.cross_entropy(logits, targets)
            loss = self.calculate_cross_entropy(targets, logits)
            
            return logits, loss

        return logits

    def backward(self, inputs, targets, input_logits):
        # need to do the same reshaping as we did for cross entropy
        targets_flat = targets.reshape(-1)
        inputs_flat = inputs.reshape(-1)
        one_hot_targets = torch.eye(self.vocab_size, dtype=torch.float32)[targets_flat]
        one_hot_inputs = torch.eye(self.vocab_size, dtype=torch.float32)[inputs_flat]

        # reshape to B*C, T
        soft_input = self.calculate_softmax(input_logits).float() # WE ARE UNSURE IF SOFTMAX IS NEEDED HERE
        soft_input = soft_input.reshape(soft_input.shape[0] * soft_input.shape[1], soft_input.shape[2])

        # derivation of softmax & crossentropy
        delta = soft_input - one_hot_targets

        # want shape (vocab_size, vocab_size) for matrix multiplication, but with correct indices (use one-hot inputs for that)
        # delta_indexed = torch.dot(one_hot_inputs.transpose(), delta)
        delta_indexed = one_hot_inputs.T @ delta

        # compute gradient for weight matrix: dot product between the transpose of the to layer and delta vector computed above
        # gradient = delta_indexed @ self.embedding_param.T  # WE ARE UNSURE ABOUT WHETHER THIS STEP IS NECESSARY, AND IF THE ORDER OF THE MATMULT + TRANSPOSE IS CORRECT
        gradient = delta_indexed
        return gradient

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logits = self(idx)
            logits = logits[:, -1, :] # becomes (B,C)

            probs = F.softmax(logits, dim=-1) # (B,C)

            idx_next = torch.multinomial(probs, num_samples=1) # (B,1)
            idx = torch.cat(([idx, idx_next]), dim=1) # (B,T+1)
        return idx

    def train(self, steps):
        for _ in range(steps):
            xb, yb = get_batch('train')

            logits, loss = self.forward(xb, yb)
            self.optimizer.zero_grad(set_to_none=True)
            # loss.backward()
            gradient = self.backward(xb, yb, logits)
            self.embedding_param.grad = torch.tensor(gradient, dtype=torch.float32)
            self.optimizer.step()
            print(loss.item())

def decode_characters(input):
    """Decodes a list of indices back to their corresponding characters
    given the abive defined vocabulary"""

    decoded = [] #given the input, we will decode it back to characters
    for i in range(0,len(input)):
        decoded.append(value_byte[input[i]])#using the translation dctionary: value_byte
#make its prettier by joining list to actual words and replacing underscores with spaces
    decoded = ''.join(decoded)
    decoded = decoded.replace('_', ' ')
    return decoded


with open('corpora/indices_text.txt', 'r') as f:
    text = eval(f.read())

with open('corpora/vocab_train.txt', 'r') as f:
    vocab = eval(f.read())

text = torch.tensor(text, dtype=torch.long)
n = int(0.9*len(text))
train_data = text[:n]
val_data = text[n:]
# train_data = torch.tensor([1, 4, 3, 2, 1, 2], dtype=torch.long)

xb, yb = get_batch('train')
m = BigramLanguageModel(len(vocab))
# m = BigramLanguageModel(5)
starting_character = torch.zeros((1,1), dtype=torch.long)

generated_characters = m.generate(idx = starting_character, max_new_tokens=100)
generated_characters = generated_characters[0].tolist()

with open (r"corpora/Shakespeare_byte.txt", 'r') as f:
   shakespeare_byte_train = eval(f.read())

with open (r"corpora/vocab_train.txt", 'r') as f:
   vocab_train = eval(f.read())

vocab = vocab_train
print(len(vocab))
indices = np.arange(0,len(vocab),1)
inidces = indices.astype(int)
indices = indices.tolist()
key_byte = dict(zip(vocab, indices))
value_byte = dict(zip(indices,vocab))

# Map each token in shakespeare_byte_train to its index using key_byte
indices_translation = [key_byte[token] for token in shakespeare_byte_train if token in key_byte]

with open('corpora/indices_text.txt', 'w') as indices_text:
    indices_text.write(str(indices_translation))

with open (r"corpora/indices_text.txt", 'r') as f:
  indices_text = eval(f.read())


bytes_translation = [value_byte[token] for token in indices_text if token in value_byte]

with open('corpora/bytes_text.txt', 'w') as bytes_text:
    bytes_text.write(str(bytes_translation))

decoded_characters =decode_characters(generated_characters)
print(decoded_characters)

m.train(100000)
generated_characters = m.generate(idx = starting_character, max_new_tokens=100)
generated_characters = generated_characters[0].tolist()
print(decode_characters(generated_characters))





