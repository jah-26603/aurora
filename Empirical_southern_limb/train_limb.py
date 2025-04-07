# -*- coding: utf-8 -*-
"""
Created on Sat Mar  1 16:19:40 2025

@author: JDawg
"""


import os
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from scipy.signal import medfilt 
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean




# Directory with saved arrays
save_dir = r'E:\all_limbs'

# List all available files
file_list = sorted([f for f in os.listdir(save_dir) if f.endswith(".npy")])

# Initialize empty lists to store the loaded data
south_scans, sza_scans, ema_scans = [], [], []

# Load each saved file
for file in tqdm(file_list):
    file_path = os.path.join(save_dir, file)
    # Only use allow_pickle if absolutely necessary
    data = np.load(file_path, allow_pickle=True)
    
    # Append data to the corresponding list
    if "south_scans" in file:
        south_scans.extend(data)  # Keep as list
    elif "sza_scans" in file:
        sza_scans.extend(data)
    elif "ema_scans" in file:
        ema_scans.extend(data)

# Find the indices where the array lengths match across all three lists
valid_indices = []
for i in range(len(south_scans)):
    if len(south_scans[i]) == len(sza_scans[i]) == len(ema_scans[i]) == 124:
        valid_indices.append(i)

# Filter the lists to keep only valid entries
# Convert lists of arrays into uniform float32 numpy arrays
south_scans = np.array([np.asarray(south_scans[i], dtype=np.float32) for i in valid_indices])
sza_scans = np.array([np.asarray(sza_scans[i], dtype=np.float32) for i in valid_indices])
ema_scans = np.array([np.asarray(ema_scans[i], dtype=np.float32) for i in valid_indices])

# Apply MinMaxScaler to each 1D signal in south_scans
south_scans = [scan/np.max(scan) for scan in south_scans]
south_scans = [medfilt(scan, kernel_size=5) for scan in south_scans]  # size=5 is the window size, you can adjust it


from sklearn.model_selection import train_test_split


# Split data into train and test sets (80% train, 20% test)
sza_train, sza_test, south_train, south_test = train_test_split(
    sza_scans, south_scans, test_size=0.01, random_state=42, shuffle=True
)

class ScanDataset(Dataset):
    def __init__(self, sza, south):
        self.X = sza # Features: SZA and EMA signals stacked
        self.Y = south  # Target: South signals
    def __len__(self):
        return len(self.Y)
    
    def __getitem__(self, idx):
        
        
        X_tensor = torch.tensor(self.X[idx], dtype=torch.float32)
        Y_tensor = torch.tensor(self.Y[idx], dtype=torch.float32)
        return X_tensor, Y_tensor

# Create train and test datasets
train_dataset = ScanDataset(sza_train,  south_train)
test_dataset = ScanDataset(sza_test, south_test)

# # #comment out below for only 1 variable
# sza_train, sza_test, south_train, south_test, ema_train, ema_test = train_test_split(
#     sza_scans, south_scans, ema_scans, test_size=0.01, random_state=42, shuffle=True
# )

# # # Define a custom Dataset class
# # Define a custom Dataset class
# class ScanDataset(Dataset):
#     def __init__(self, sza, south, ema):
#         self.X = np.column_stack((sza, ema)) # Features: SZA and EMA signals stacked
#         self.Y = south  # Target: South signals
#     def __len__(self):
#         return len(self.Y)
    
#     def __getitem__(self, idx):
#         X_tensor = torch.tensor(self.X[idx], dtype=torch.float32)
#         Y_tensor = torch.tensor(self.Y[idx], dtype=torch.float32)
#         return X_tensor, Y_tensor

# # Create train and test datasets
# train_dataset = ScanDataset(sza_train,  south_train, ema_train)
# test_dataset = ScanDataset(sza_test, south_test, ema_test)

# Create DataLoaders
batch_size = 100  # Adjust based on available GPU memory
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)


# Define the Model
# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class SimpleNN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SimpleNN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            # Adding 3 more layers:
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, output_dim)
        )


    
    def forward(self, x):
        return self.model(x)




# Initialize model, loss function, optimizer
input_dim = 124  # SZA and EMA as features (each with 124 length)
output_dim = 124  # South Scans as output (target signal length is 124)
model = SimpleNN(input_dim, output_dim).to(device)
criterion = nn.SmoothL1Loss()
optimizer = torch.optim.RMSprop(model.parameters(), lr=0.1, alpha=0.99)
optimizer = torch.optim.Adagrad(model.parameters(), lr=0.001, weight_decay = 1e-5)

# Training Loop
epochs = 300  # Set the number of epochs
for epoch in range(epochs):
    model.train()  # Set model to training mode
    running_loss = 0.0
    
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        
        # Compute loss
        loss = criterion(outputs, targets)
        # breakpoint()


        # Backward pass
        loss.backward()
        
        # Optimize the model
        optimizer.step()
        
        running_loss += loss.item()
    
    avg_loss = running_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")

        # Evaluate on the test set every 5 epochs
    if (epoch + 1) % 5 == 0:
        model.eval()  # Set model to evaluation mode
        test_loss = 0.0
        
        with torch.no_grad():  # Disable gradient calculation
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                
                # Forward pass
                outputs = model(inputs)
                
                # Compute test loss
                loss = criterion(outputs, targets)
                test_loss += loss.item()
        
        avg_test_loss = test_loss / len(test_loader)
        print(f"==> Test Loss after Epoch {epoch+1}: {avg_test_loss:.4f}")
        


# Save the trained model
torch.save(model.state_dict(), 'scan_predictor_model_2019_2020.pth')
print("Model saved!")

import matplotlib.pyplot as plt
import random
import torch

# Set the model to evaluation mode
model.eval()

# Initialize lists to store predictions and true values for selected samples
predictions = []
true_values = []

# Get a random subset of 10 samples from the test_loader
num_samples = 50
sample_indices = random.sample(range(len(test_loader.dataset)), num_samples)


num_samples = min(50, len(test_dataset))  # Ensure we don't exceed dataset size
sample_indices = random.sample(range(len(test_dataset)), num_samples)  # Use test_dataset, not test_loader

# # Disable gradient calculation for evaluation
# with torch.no_grad():
#     for idx in sample_indices:
#         inputs, targets = test_loader.dataset[idx]
#         inputs, targets = inputs.to(device), targets.to(device)
        
#         # Get predictions from the model
#         outputs = model(inputs.unsqueeze(0))  # Add batch dimension
        
#         # Store the predictions and true values
#         predictions.append(outputs.cpu().numpy().flatten())  # Flatten to 1D
#         true_values.append(targets.cpu().numpy().flatten())  # Flatten to 1D


with torch.no_grad():
    for idx in sample_indices:
        inputs, targets = test_dataset[idx]  # Access from test_dataset
        inputs, targets = inputs.to(device), targets.to(device)

        # Forward pass
        outputs = model(inputs.unsqueeze(0))  # Add batch dimension

        # Store results
        predictions.append(outputs.cpu().numpy().flatten())
        true_values.append(targets.cpu().numpy().flatten())

# Plotting the results for each sample in a new plot
for i in range(num_samples):
    plt.figure(figsize=(10, 6))
    plt.plot(true_values[i], label="True Value", color='blue')
    plt.plot(predictions[i], label="Prediction", color='red')
    plt.xlabel("Time Step")
    plt.ylabel("Signal Value")
    plt.title(f"Sample {i+1}: True Value vs Prediction")
    plt.legend()
    plt.show()
