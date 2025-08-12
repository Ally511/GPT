import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import os
import re

block_size = 8
batch_size = 128

""" SPLIT THE TEXT INTO TRAIN AND TEST SET"""
with open('corpora/indices_text.txt', 'r') as f:
    text = eval(f.read())

text = torch.tensor(text, dtype=torch.long)
n = int(0.9*len(text))
train_data = text[:n]
val_data = text[n:]


def get_batch(split):
    """
    Generates a batch of input and target sequences for training or validation.

    Args:
        split (str): Either 'train' or 'val', determines which dataset to use.

    Returns:
        x (torch.Tensor): Batch of input sequences of shape (batch_size, block_size).
        y (torch.Tensor): Batch of target sequences (next tokens) of shape (batch_size, block_size).
    """
    # Select the appropriate dataset based on the split
    data = train_data if split == 'train' else val_data
    # Randomly choose starting indices for each sequence in the batch
    ix = torch.randint(len(data) - block_size, (batch_size,))
    # Create input sequences of length block_size
    x = torch.stack([data[i:i+block_size] for i in ix])
    # Create target sequences by shifting input by one position
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y

with open (r"corpora/Shakespeare_byte.txt", 'r') as f:
   shakespeare_byte_train = eval(f.read())

with open('corpora/vocab_train.txt', 'r') as f:
    vocab = eval(f.read())

indices = np.arange(0,len(vocab),1)
inidces = indices.astype(int).tolist()
value_byte = dict(zip(indices,vocab))

def decode_characters(input):
    """
    Decodes a list of integer indices into their corresponding characters using a predefined vocabulary mapping.

    Args:
        input (list of int): A list of integer indices representing encoded characters.

    Returns:
        str: The decoded string, where underscores ('_') are replaced with spaces.

    Notes:
        - The function relies on a global dictionary `value_byte` that maps indices to characters.
        - The decoded characters are joined into a single string.
        - Underscores in the decoded string are replaced with spaces for readability.
    """
    decoded = [] #given the input, we will decode it back to characters
    for i in range(0,len(input)):
        decoded.append(value_byte[input[i]])#using the translation dctionary: value_byte
    # make it prettier by joining list to actual words and replacing underscores with spaces
    decoded = ''.join(decoded)
    decoded = decoded.replace('_', ' ')
    return decoded

class BigramLanguageModel(nn.Module):
    """
    A simple bigram language model implemented using PyTorch.
    This model learns a (vocab_size x vocab_size) embedding matrix that predicts the next token
    given the current token, using a bigram approach.
    """

    def __init__(self, vocab_size):
        """
        Initializes the BigramLanguageModel.

        Args:
            vocab_size (int): The size of the vocabulary.
        """
        super().__init__()
        # Initialize the embedding matrix with small random values centered around 0
        limit = 1 / np.sqrt(vocab_size)
        self.token_embedding_table = (
            (np.random.rand(vocab_size, vocab_size).astype(np.float32) * 2 - 1) * limit
        )
        # Convert the embedding matrix to a torch.nn.Parameter for optimization
        self.embedding_param = torch.nn.Parameter(torch.from_numpy(self.token_embedding_table))
        self.embedding_param.data = self.embedding_param.data.float()

        # AdamW optimizer for parameter updates
        self.optimizer = torch.optim.AdamW([self.embedding_param], lr=1e-3)
        self.vocab_size = vocab_size
        
        # save loss for later reference
        self.current_loss = 0

    def calculate_softmax(self, x):
        """
        Computes the softmax of the input tensor along the last dimension.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Softmax probabilities.
        """
        exp_x = torch.exp(x - torch.max(x, dim=-1, keepdim=True).values)
        return exp_x / torch.sum(exp_x, dim=-1, keepdim=True)

    def calculate_cross_entropy(self, y_hatless, y_hat):
        """
        Computes the cross-entropy loss between the target and predicted logits.

        Args:
            y_hatless (torch.Tensor): Target tensor of shape (batch, seq_len).
            y_hat (torch.Tensor): Predicted logits of shape (batch, seq_len, vocab_size).

        Returns:
            torch.Tensor: Scalar loss value.
        """
        # Get vocab size from logits shape
        _, _, vocab_size = y_hat.shape
        # Flatten logits and targets for loss computation
        y_hat = y_hat.reshape(y_hat.shape[0] * y_hat.shape[1], y_hat.shape[2])
        y_hatless_flat = y_hatless.reshape(-1).long()
        # One-hot encode targets
        y_hatless_hot = torch.eye(vocab_size)[y_hatless_flat]
        # Compute softmax and log probabilities
        y_hat = self.calculate_softmax(y_hat)
        y_hat = torch.log(y_hat)
        # Compute mean cross-entropy loss
        return torch.mean(-torch.sum(y_hatless_hot * y_hat, dim=1))

    def forward(self, idx, targets=None):
        """
        Forward pass of the model.

        Args:
            idx (torch.Tensor): Input indices of shape (batch, seq_len).
            targets (torch.Tensor, optional): Target indices for loss computation.

        Returns:
            logits (torch.Tensor): Output logits of shape (batch, seq_len, vocab_size).
            loss (torch.Tensor, optional): Cross-entropy loss if targets are provided.
        """
        # Lookup embeddings for input indices
        logits = self.embedding_param[idx]
        if targets is not None:
            # Compute loss if targets are provided
            loss = self.calculate_cross_entropy(targets, logits)
            return logits, loss
        return logits

    def backward(self, inputs, targets, input_logits):
        """
        Computes the gradient of the loss with respect to the embedding parameters.

        Args:
            inputs (torch.Tensor): Input indices of shape (batch, seq_len).
            targets (torch.Tensor): Target indices of shape (batch, seq_len).
            input_logits (torch.Tensor): Logits from the forward pass.

        Returns:
            torch.Tensor: Gradient tensor for the embedding parameters.
        """
        # Flatten inputs and targets
        targets_flat = targets.reshape(-1)
        inputs_flat = inputs.reshape(-1)
        # One-hot encode inputs and targets
        one_hot_targets = torch.eye(self.vocab_size, dtype=torch.float32)[targets_flat]
        one_hot_inputs = torch.eye(self.vocab_size, dtype=torch.float32)[inputs_flat]
        # Compute softmax probabilities for logits
        soft_input = self.calculate_softmax(input_logits).float()
        soft_input = soft_input.reshape(soft_input.shape[0] * soft_input.shape[1], soft_input.shape[2])
        # Compute delta for gradient (softmax output - one-hot targets)
        delta = soft_input - one_hot_targets
        # Compute gradient for the embedding matrix using matrix multiplication
        delta_indexed = one_hot_inputs.T @ delta
        gradient = delta_indexed
        return gradient

    def generate(self, idx, max_new_tokens):
        """
        Generates a sequence of tokens from the model, given a starting index.

        Args:
            idx (torch.Tensor): Starting token indices of shape (batch, 1).
            max_new_tokens (int): Number of new tokens to generate.

        Returns:
            torch.Tensor: Generated sequence of token indices.
        """
        for _ in range(max_new_tokens):
            # Get logits for current sequence
            logits = self(idx)
            logits = logits[:, -1, :]  # Use last token's logits
            # Compute probabilities and sample next token
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            # Append new token to sequence
            idx = torch.cat(([idx, idx_next]), dim=1)
        return idx

    def train(self, training_steps, validation_steps):
        """
        Trains the model for a specified number of steps.

        Args:
            steps (int): Number of training steps.
        """
        patience = np.inf
        for step in range(training_steps):
            # Get a batch of data
            xb, yb = get_batch('train')
            # Forward pass
            logits, current_loss = self.forward(xb, yb)
            m.current_loss = (m.current_loss + current_loss)
            # Zero gradients
            self.optimizer.zero_grad(set_to_none=True)
            # Compute gradients manually
            gradient = self.backward(xb, yb, logits)
            self.embedding_param.grad = torch.tensor(gradient, dtype=torch.float32)
            # Update parameters
            self.optimizer.step()

            if(step % validation_steps == 0):
                val_loss = 0
                
                for i in range( validation_steps//10 ):
                    xb, yb = get_batch('val')
                    _, loss = self.forward(xb, yb)
                    val_loss = val_loss + loss
                val_loss = val_loss / (validation_steps //10)
                m.current_loss = m.current_loss / validation_steps
                print(f"loss {self.current_loss.item()} val_loss {val_loss} ")
                
                if(patience < val_loss):
                    break
                patience = val_loss
                m.current_loss = 0
                

# train the model and save the embedding
training_steps = 1000000
validation_steps = 500
m = BigramLanguageModel(len(vocab))
m.train(training_steps, validation_steps)
trained_embedding = m.token_embedding_table
loss = m.current_loss

# check if less than k models have been saved or if the loss is better
k = 5
save_dir = 'n_grams'
pattern = re.compile(r'trained_embedding_loss_([0-9.]+)\.npy')
existing_files = [f for f in os.listdir(save_dir) if pattern.match(f)]

# Extract losses from filenames
losses = []
for fname in existing_files:
    match = pattern.match(fname)
    if match:
        try:
            losses.append(float(match.group(1)))
        except ValueError:
            continue

save_embedding = False
if len(existing_files) < k: #less than k models saved
    save_embedding = True
elif losses and loss < max(losses): # lower loss
    max_loss = max(losses)
    max_loss_file = [f for f in existing_files if f'trained_embedding_loss_{max_loss}' in f]
    for f in max_loss_file:
        os.remove(os.path.join(save_dir, f))
    save_embedding = True
if save_embedding:
    np.save(os.path.join(save_dir, f'trained_embedding_loss_{loss}.npy'), trained_embedding)

embedding = np.load(f'n_grams/trained_embedding_loss_{loss}.npy')
m = BigramLanguageModel(len(vocab))
m.token_embedding_table = embedding

# generate a sentence
starting_character = torch.zeros((1,1), dtype=torch.long)
generated_characters = m.generate(idx = starting_character, max_new_tokens=100)
generated_characters = generated_characters[0].tolist()
print(decode_characters(generated_characters))

# print(len(vocab))


# # Map each token in shakespeare_byte_train to its index using key_byte
# indices_translation = [key_byte[token] for token in shakespeare_byte_train if token in key_byte]

# with open('corpora/indices_text.txt', 'w') as indices_text:
#     indices_text.write(str(indices_translation))

# with open (r"corpora/indices_text.txt", 'r') as f:
#   indices_text = eval(f.read())


# bytes_translation = [value_byte[token] for token in indices_text if token in value_byte]

# with open('corpora/bytes_text.txt', 'w') as bytes_text:
#     bytes_text.write(str(bytes_translation))

# decoded_characters =decode_characters(generated_characters)
# print(decoded_characters)


# generated_characters = m.generate(idx = starting_character, max_new_tokens=100)
# generated_characters = generated_characters[0].tolist()
# print(decode_characters(generated_characters))





