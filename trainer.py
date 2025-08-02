import torch.nn as nn
from tqdm import tqdm

# ToDO: include tqdm
class Trainer:

    def __init__(self, model, optimizer, train_dataset, device):
        self.model = model
        # ToDo: potentially this has to be adapted depending on how this is solved in the model
        self.optimizer = optimizer
        self.train_dataset = train_dataset
        self.device = device
        self.model = self.model.to(self.device)


    def run(self, epochs, train_steps):

        # set the model to training mode for Dropout etc.
        self.model.train()

        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}/{epochs}")
            progress_bar = tqdm(range(train_steps), desc="Training", leave=False)

            for train_step in range(train_steps):
                # ToDO: adapt to how we handle batches
                batch = get_batch(self.train_dataset)

                batch = [t.to(self.device) for t in batch]
                xb, yb = batch

                # forward step, returns the logits and the loss since we are passing targets
                logits, self.loss = self.model(xb, yb)

                # do the backward step
                self.model.zero_grad()
                self.loss.backward()
                self.optimizer.step()

                progress_bar.set_postfix(loss=self.loss.item())