"""this file contains the Trainer class that handles the training loops for the GPT or any other PyTorch model"""
import numpy as np
import torch
from tqdm import tqdm
from utility_functions import get_batch, decode_characters
# ACHTUNG: added val_dataset to init

class Trainer:
    """
        Trainer class for handling model training loops.

        This class encapsulates the training process for a given PyTorch model,
        including dataset batching, forward/backward passes, and optimizer steps.
    """

    def __init__(self, model, train_dataset, vocab, device, val_dataset=None):
        """
            Initialize the Trainer.

            Args:
                model (torch.nn.Module): The model to be trained. Must have an `optimizer` attribute.
                train_dataset (numpy.ndarray or similar): The raw training data from which batches will be sampled.
                device (torch.device): The device to run training on (e.g., torch.device("cuda") or torch.device("cpu")).
        """
        self.optimizer = model.optimizer
        self.train_dataset = train_dataset
        self.model = model
        self.vocab = vocab
        self.device = device
        self.model = self.model.to(self.device)

        self.val_dataset = val_dataset


    def run(self, epochs, train_steps, batch_size, chunk_size):
        """
                Execute the training loop for the model.

                For each epoch and training step, this method:
                - Samples a batch from the training dataset.
                - Moves the batch to the specified device.
                - Performs a forward pass to compute logits and loss.
                - Backpropagates the loss.
                - Updates model parameters using the optimizer.
                - Records loss values.

                Args:
                    epochs (int): Number of full passes through the dataset.
                    train_steps (int): Number of training steps per epoch.
                    batch_size (int): Number of sequences per training batch.
                    chunk_size (int): Number of tokens per sequence in a batch.

                Returns:
                    list[float]: A list containing the loss values recorded at each training step.
        """
        # set the model to training mode for Dropout etc.
        self.model.train()
        losses = []

        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}/{epochs}")
            progress_bar = tqdm(range(train_steps), desc="Training", leave=False)

            for train_step in range(train_steps):

                batch = get_batch(self.train_dataset, batch_size, chunk_size)
                batch = [torch.tensor(b, dtype=torch.int64) for b in batch]
                batch = [t.to(self.device) for t in batch]
                xb, yb = batch

                # forward step, returns the logits and the loss since we are passing targets
                logits, loss = self.model(xb, yb)
                losses.append(float(loss.detach()))

                # do the backward step
                # self.model.zero_grad()
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                if self.val_dataset and train_step % 1000 == 0:
                    validation_loss = self.evaluate_loss(self.val_dataset, batch_size, chunk_size)
                    val_perplexity = np.exp(validation_loss)
                    print(f"Validation Loss: {validation_loss:.4f}, Perplexity: {val_perplexity:.2f}")

                val_loss = loss.item()
                # tqdm.write(f"Loss: {val_loss:.4f}")
                progress_bar.set_postfix(loss=val_loss)
                progress_bar.update(1)
                if train_step % 1000 == 0:
                    generated = self.model.generate(torch.tensor([[0]], dtype=torch.long).to(self.device), 100, 0.8, True, 20)
                    generated = generated[0].tolist()
                    decoded = decode_characters(generated, self.vocab)
                    print(decoded)
        return losses
    
    def evaluate_loss(self, dataset, batch_size, chunk_size, num_batches=50):
        # set model to evaluation mode
        self.model.eval()
        losses = []
        with torch.no_grad():
            for _ in range(num_batches):
                xb, yb = get_batch(dataset, batch_size, chunk_size)
                xb, yb = xb.to(self.device), yb.to(self.device)
                _, loss = self.model(xb, yb)
                losses.append(loss.item())
        # set model to training mode again
        self.model.train()  
        return np.mean(losses)
