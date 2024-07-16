import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm

class Trainer:
    def __init__(self, model, train_loader, criterion=nn.CrossEntropyLoss(), optimizer=None, lr=0.001, output_file_name=None):
        """
        Initializes the Trainer class.

        Parameters:
        model (nn.Module): The neural network model to train.
        train_loader (DataLoader): DataLoader for the training dataset.
        criterion (nn.Module): Loss function. Default is CrossEntropyLoss.
        optimizer (torch.optim.Optimizer, optional): Optimizer for training. Default is Adam.
        lr (float): Learning rate for the optimizer. Default is 0.001.
        output_file_name (str, optional): Filename to save the model after training.
        """
        self.model = model
        self.train_loader = train_loader
        self.criterion = criterion
        self.optimizer = optimizer or Adam(model.parameters(), lr=lr)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.output_file_name = output_file_name

    def train(self, num_epochs=30):
        """
        Trains the model for a specified number of epochs.

        Parameters:
        num_epochs (int): Number of epochs to train the model. Default is 30.
        """
        self.model.to(self.device)
        self.model.train()  # Set the model to training mode
        
        for epoch in tqdm(range(num_epochs), desc="Training", unit=" epochs"):
            total_loss = 0.0
            for images, labels in self.train_loader:
                self.optimizer.zero_grad()  # Reset previously stored gradients
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(images)  # Make predictions on the batch
                loss = self.criterion(outputs, labels)  # Compute the loss
                loss.backward()  # Compute gradients of the loss
                self.optimizer.step()  # Update the model weights
                total_loss += loss.item() * images.size(0)

        if self.output_file_name is not None:
            torch.save(self.model, f"{self.output_file_name}_{num_epochs}.pth")
