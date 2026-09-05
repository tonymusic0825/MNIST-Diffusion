from dataset import create_dataloader
from diffusion import NoiseSchedule
from model import UNet
import torch
import torch.nn as nn
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
T = 1000
train_dataloader, test_dataloader = create_dataloader(test=False)
noise_schedule = NoiseSchedule(T=T)
model = UNet().to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)
loss = nn.MSELoss()
epochs = 50
pre_loss = float("inf")
os.makedirs("./checkpoints", exist_ok=True)

# Train loop
for epoch in range(epochs):
    for images, _ in train_dataloader:
        images = images.to(device)
        t = torch.randint(0, (T-1), (images.shape[0],)).to(device)
        noisy_images, real_noise = noise_schedule.forward(images, t)

        pred_noise = model(noisy_images, t)
        pred_loss = loss(pred_noise, real_noise)

        # Backprop
        optimizer.zero_grad()
        pred_loss.backward()
        optimizer.step()

    if pred_loss.item() < pre_loss:
        print("New Loss Record detected weights have been saved to ./checkpoints as 'mnist_diffusion_weights.pth'")
        torch.save(model.state_dict(), "./checkpoints/mnist_diffusion_weights.pth")
    
        pre_loss = pred_loss.item()
    
    print(f"Epoch {epoch+1}/{epochs} | Loss: {pred_loss.item():.4f}")


