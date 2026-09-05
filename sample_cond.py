import matplotlib.pyplot as plt
from model_cond import ConditionalUNet
import torch
from diffusion import NoiseSchedule
from PIL import Image
import numpy as np
from utils import *
import argparse

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
T = 1000
model = ConditionalUNet().to(device)
model.load_state_dict(torch.load("./checkpoints/cond_mnist_diffusion_weights.pth"))
model.eval()
noise_schedule = NoiseSchedule(T) 
batch_size = 1

def main(args):

    with torch.no_grad():

        while True:
            frames = []
            user_input = input("Please input your desired number: ")

            if user_input == 'q':
                break

            if not user_input.isnumeric() or int(user_input) < 0 or int(user_input) > 9:
                print("That is an invalid option please choose agian...")
                continue

            # Random Gaussian Noise
            x = torch.randn((batch_size, 1, 28, 28)).to(device)

            num = torch.tensor(int(user_input), dtype=torch.long, device=device)

            for i in reversed(range(T)):
                t = torch.full((batch_size,), i, device=device, dtype=torch.long)

                pred_noise = model(x, t, num)

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

                # Save frame every 10 steps
                if args.gif:
                    if i % 10 == 0 or i == 0:
                        frame = x[0].detach().cpu().squeeze().numpy()

                        # Convert [-1, 1] -> [0, 255]
                        frame = (frame + 1) / 2
                        frame = np.clip(frame, 0, 1)
                        frame = (frame * 255).astype(np.uint8)

                        # Convert to PIL image
                        frame = Image.fromarray(frame)

                        # Make it larger so the GIF isn't tiny
                        frame = frame.resize((280, 280), Image.Resampling.NEAREST)

                        frames.append(frame)

            if args.display:
                terminal_display(x)

            # Save GIF
            if args.gif:
                frames[0].save(
                    f"./readme/diffusion_{user_input}.gif",
                    save_all=True,
                    append_images=frames[1:],
                    duration=50,
                    loop=0
                )


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Sample Condition UNet"
    )

    parser.add_argument(
        "-g", "--gif",
        action="store_true",
        help="Gif of image reconstruction is saved"
    )

    parser.add_argument(
        "-d", "--display",
        action="store_true",
        help="Display reconstructed images"
    )

    args = parser.parse_args()

    main(args)