import torch
import torch.nn as nn
from tqdm import tqdm
from utility_functions import get_batch


class Trainer:

    def __init__(self, model, train_dataset, device):
        self.optimizer = model.optimizer
        self.train_dataset = train_dataset
        self.model = model
        self.device = device
        # self.train_dataset = self.train_dataset.to(device)
        self.model = self.model.to(self.device)


    def run(self, epochs, train_steps, batch_size, chunk_size):

        # set the model to training mode for Dropout etc.
        self.model.train()

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

                # do the backward step
                self.model.zero_grad()
                loss.backward()
                self.optimizer.step()

                progress_bar.set_postfix(loss=loss.item())