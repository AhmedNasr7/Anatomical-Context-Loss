import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from models import UNet
from utils import * 
from loss import Anatomical_Context_Loss
from dataset import RandomDataGenerator
from torch.utils.data import Dataset, DataLoader
from loguru import logger
from tqdm import tqdm 

# Define training parameters
batch_size = 1
learning_rate = 0.001
epochs = 10
device = "cuda"


# Define hyperparameters
num_samples = 10  # Number of random samples
image_size = 512    # Size of the image (512, 512)
in_channels = 1
out_channels = 19

# Create an instance of the random data generator
random_data = RandomDataGenerator(num_samples, image_size)

# Create a DataLoader for training
train_loader = DataLoader(random_data, batch_size=batch_size, shuffle=True)


# Initialize model, loss function, and optimizer
model = UNet(in_channels, out_channels).to(device)
criterion = Anatomical_Context_Loss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(epochs):

    running_loss = 0.0
    with tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}", unit="batch") as pbar:

        for i, data in enumerate(pbar):
            inputs, targets = data
            inputs, targets = inputs.to(device), targets.to(device)

            # logger.debug(f"inputs, targets: {inputs.shape, targets.shape}")
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            # logger.debug(f"outputs: {outputs.shape}")

            loss = criterion(outputs, targets)

            # Backpropagation and optimization
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
    
            pbar.set_postfix(loss=f"{running_loss:.4f}")



print("Training finished!")
