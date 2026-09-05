from dataset import create_dataloader
from diffusion import NoiseSchedule
from model_cond import ConditionalUNet
import torch
import torch.nn as nn
import os
import argparse

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    T = args.timesteps

    batch_size = args.batch_size
    train_dataloader, _ = create_dataloader(test=False, batch_size=batch_size)
    noise_schedule = NoiseSchedule(T=T)
    model = ConditionalUNet().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss = nn.MSELoss()
    epochs = args.epochs

    pre_loss = float("inf")
    os.makedirs(args.save_dir, exist_ok=True)
    save_path = os.path.join(args.save_dir, "cond_mnist_diffusion_weights.pth")

    # Train loop
    for epoch in range(epochs):
        epoch_loss = 0.0
        for images, labels in train_dataloader:
            images = images.to(device)
            labels = labels.to(device)
            
            t = torch.randint(0, (T-1), (images.shape[0],)).to(device)
            noisy_images, real_noise = noise_schedule.forward(images, t)

            pred_noise = model(noisy_images, t, labels)
            pred_loss = loss(pred_noise, real_noise)

            # Backprop
            optimizer.zero_grad()
            pred_loss.backward()
            optimizer.step()

            epoch_loss += pred_loss.item()

        avg_loss = epoch_loss / len(train_dataloader)
        
        if avg_loss < pre_loss:
            print(f"New Loss Record detected. Weights saved to {save_path}")
            torch.save(model.state_dict(), save_path)
            pre_loss = avg_loss

        print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Conditional MNIST Diffusion Model")
    parser.add_argument("--epochs", type=int, default=50, help="Total number of training epochs")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate for the optimizer")
    parser.add_argument("--timesteps", type=int, default=1000, help="Total diffusion timesteps (T)")
    parser.add_argument("--save_dir", type=str, default="./checkpoints", help="Directory to save model weights")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for dataloader")
    
    args = parser.parse_args()
    main(args)