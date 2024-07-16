# src/train.py
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from dataset import ImageDataset
from model import LappedTransform
from utils import count_parameters
import os
import matplotlib.pyplot as plt

dataset_colored_path = 'data/colored'
dataset_gray_path = 'data/gray'
fixed_size = (256, 256)
batch_size = 32
learning_rate = 1e-3

transform = transforms.Compose([
    transforms.Resize(fixed_size),
    transforms.ToTensor()
])

dataset_colored = ImageDataset(folder_path=dataset_colored_path, transform=transform)
dataloader_colored = DataLoader(dataset_colored, batch_size=batch_size, shuffle=True)

dataset_gray = ImageDataset(folder_path=dataset_gray_path, transform=transform)
dataloader_gray = DataLoader(dataset_gray, batch_size=batch_size, shuffle=True)

lt = LappedTransform(kernel_size=16)
optimizer = optim.Adam(lt.parameters(), lr=learning_rate)
criterion = nn.MSELoss()

count_parameters(lt)

def train(model, dataloader, criterion, optimizer, colored=False, epochs=1):
    losses = []
    model.train()
    
    for epoch in range(epochs):
        for images, _ in dataloader:
            if images.dim() != 4:
                raise ValueError(f"Expected: [batch_size, channels, height, width], received: {images.shape}")
            outputs = lt(images)
            if outputs.shape != images.shape:
                raise ValueError(f"Output shape {outputs.shape} mismatch with input shape {images.shape}")
        
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, images)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item()}')
    
    return model, losses

model_colored, losses_colored = train(lt, dataloader_colored, criterion, optimizer, epochs=10)
torch.save(model_colored.state_dict(), 'models/lapped_transform_colored.pth')
model_gray, losses_gray = train(lt, dataloader_gray, criterion, optimizer, epochs=10)
torch.save(model_gray.state_dict(), 'models/lapped_transform_gray.pth')