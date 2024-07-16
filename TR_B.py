import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm

class Trainer:
    def __init__(self, model, train_loader, criterion=nn.CrossEntropyLoss(), optimizer=None, lr=0.001, output_file_name=None):
        self.model = model
        self.train_loader = train_loader
        self.criterion = criterion
        self.optimizer = optimizer or Adam(model.parameters(), lr=lr)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.output_file_name = output_file_name

    def train(self, num_epochs=30):
        self.model.to(self.device)
        self.model.train()  # Mettre le modèle en mode entraînement
        for epoch in tqdm(range(num_epochs),   desc="Training", unit=" epochs"):
            total_loss = 0.0
            for images, labels, _ in self.train_loader:
                self.optimizer.zero_grad()  # Réinitialiser les gradients précédemment stockés
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(images)  # Faire les prédictions sur le lot
                loss = self.criterion(outputs, labels)  # Calculer la perte
                loss.backward()  # Calculer les gradients de la perte
                self.optimizer.step()  # Mettre à jour les poids du modèle
                total_loss += loss.item() * images.size(0)
        if self.output_file_name != None:
            torch.save(self.model, f"{self.output_file_name}_{num_epochs}.pth")

