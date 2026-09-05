import matplotlib.pyplot as plt
from model import UNet
import torch
from diffusion import NoiseSchedule
from PIL import Image
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
T = 1000
model = UNet().to(device)
model.load_state_dict(torch.load("./checkpoints/mnist_diffusion_weights.pth"))
model.eval()
noise_schedule = NoiseSchedule(T) 
batch_size = 1

frames = []

# Random Gaussian Noise
x = torch.randn((batch_size, 1, 28, 28)).to(device)

with torch.no_grad():
    for i in reversed(range(T)):
        t = torch.full((batch_size,), i, device=device, dtype=torch.long)

        pred_noise = model(x, t)

        # Extract and Reshape
        alpha_t = noise_schedule.alpha[i].view(-1, 1, 1, 1)
        alpha_cumprod_t = noise_schedule.alpha_cumprod[i].view(-1, 1, 1, 1)
        beta_t = noise_schedule.beta[i].view(-1, 1, 1, 1)

        # Apply DDPM formula
        scaling_fac = 1.0 / torch.sqrt(alpha_t)
        noise_weight = (1.0 - alpha_t) / torch.sqrt(1.0 - alpha_cumprod_t)

        x = scaling_fac * (x - noise_weight * pred_noise)

        if i > 0:
            z = torch.randn_like(x)
            x = x + torch.sqrt(beta_t) * z

        if i % 10 == 0 or i == 0:
            frame = x[0].detach().cpu().squeeze().numpy()

            frame = (frame + 1) / 2
            frame = np.clip(frame, 0, 1)
            frame = (frame * 255).astype(np.uint8)

            frame = Image.fromarray(frame)

            frame = frame.resize((280, 280), Image.Resampling.NEAREST)

            frames.append(frame)

    x = (x.clamp(-1, 1) + 1) / 2
    x = x.cpu().numpy()

# Save GIF
frames[0].save(
    "./readme/diffusion.gif",
    save_all=True,
    append_images=frames[1:],
    duration=50,
    loop=0
)