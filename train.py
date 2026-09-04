from dataset import create_dataloader
from diffusion import NoiseSchedule
from model import UNet
import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_dataloader, test_dataloader = create_dataloader(test=False)
noise_schedule = NoiseSchedule(T=500)
model = UNet().to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss = nn.MSELoss()
epochs = 10

# Train loop

for epoch in range(epochs):
    for images, _ in train_dataloader:
        images.to(device)
        t = torch.randint(0, 499, (images.shape[0],)).to(device)
        noisy_images, real_noise = noise_schedule.forward(images, t)

        pred_noise = model(noisy_images, t)
        pred_loss = loss(pred_noise, real_noise)

        # Backprop
        optimizer.zero_grad()
        pred_loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}/{epochs} | Loss: {pred_loss.item():.4f}")
