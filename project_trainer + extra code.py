#This is a demo created to train on a separate program the AI created that has been developed;
#you can still find everything yoou need to train your Artificial Intelligence in the final project file
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import random
import os
import numpy as np

# ---------------------------------------------------------
# 1. SETUP E CARICAMENTO DATI
# ---------------------------------------------------------
IMG_SIZE = 320 
BATCH_SIZE = 16

transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor()
])

dataset_path = "./data/brain_tumor_data"

# Gestione caricamento dati
try:
    full_dataset = datasets.ImageFolder(root=dataset_path, transform=transform)
    print(f"Classi trovate: {full_dataset.class_to_idx}")
except FileNotFoundError:
    print("ERRORE: Cartella dati non trovata. Uso dati finti per test.")
    full_dataset = datasets.FakeData(size=100, image_size=(1, IMG_SIZE, IMG_SIZE), num_classes=4, transform=transforms.ToTensor())

# Divisione Train/Val
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

# Dataloaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ---------------------------------------------------------
# 2. DEFINIZIONE DEL MODELLO
# ---------------------------------------------------------
class BrainMRI_MLP(nn.Module):
    def __init__(self):
        super(BrainMRI_MLP, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(320 * 320, 128) 
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 4) 
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BrainMRI_MLP().to(device)

# 1. Crei un modello vuoto (struttura)
model = BrainMRI_MLP().to(device)

# 2. Carichi i pesi che hai "sudato" ieri
model.load_state_dict(torch.load("brain_tumor_classifier_320px.pth"))

# ---------------------------------------------------------
# 3. TRAINING LOOP (Prima alleniamo, poi testiamo!)
# ---------------------------------------------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
epoch_number = 10 

print("\nInizio Training...")
model.train() # Mette il modello in modalità allenamento

for epoch in range(epoch_number):
    running_loss = 0.0
    for images, labels in train_loader: # CORRETTO: indentazione del blocco
        images, labels = images.to(device), labels.to(device)
        
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad() 
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    # Stampa media loss per epoca (più pulito)
    avg_loss = running_loss / len(train_loader)
    print(f"Epoch: {epoch+1}, Average Loss: {avg_loss:.4f}")

print("Training completato.")

# Salvataggio
torch.save(model.state_dict(), "brain_tumor_classifier_320px.pth")
print("Modello salvato.")

#COMANDO DA INSERIRE SE BISOGNA TESTARE LA MACCHINA SENZA AVERE DATI UPLOADATI
#except FileNotFoundError:
   # print("ERRORE: Cartella dati non trovata. Assicurati di aver creato la cartella './data/brain_tumor_data'.")
    # Creiamo un dataset fittizio solo per far compilare il codice se non hai i dati
    #full_dataset = datasets.FakeData(size=100, image_size=(1, IMG_SIZE, IMG_SIZE), num_classes=4, transform=transforms.ToTensor())

