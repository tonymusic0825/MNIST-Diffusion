import torch
import torch.nn as nn
import math

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class SelfAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim=channels, num_heads=4, batch_first=True)
        self.norm = nn.GroupNorm(num_groups=8, num_channels=channels)

    def forward(self, x):
        B, C, H, W = x.shape
        flat_x = self.norm(x).view(B, C, H * W).swapaxes(1, 2)
        attn_out, _ = self.mha(flat_x, flat_x, flat_x)
        attn_out = attn_out.swapaxes(1, 2).view(B, C, H, W)
        return x + attn_out

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_ch)
        self.conv1 = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, padding=1, kernel_size=3)
        self.batchNorm = nn.BatchNorm2d(num_features=out_ch)
        self.silu = nn.SiLU() # Upgraded from ReLU

    def forward(self, x, t_emb):
        x = self.conv1(x)
        time_val = self.time_mlp(t_emb).view(-1, x.shape[1], 1, 1)
        x = x + time_val 
        x = self.batchNorm(x)
        return self.silu(x)

class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        time_emb_dim = 32
        
        # Upgraded time processing using continuous math instead of random embeddings
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim)
        )

        # Encoder (Doubled Depth)
        self.down1_1 = ConvBlock(1, 32, time_emb_dim)
        self.down1_2 = ConvBlock(32, 32, time_emb_dim)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        
        self.down2_1 = ConvBlock(32, 64, time_emb_dim)
        self.down2_2 = ConvBlock(64, 64, time_emb_dim)
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        # Bottleneck with Attention
        self.bot1 = ConvBlock(64, 128, time_emb_dim)
        self.attn = SelfAttention(128)
        self.bot2 = ConvBlock(128, 128, time_emb_dim)

        # Decoder (Doubled Depth)
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.up_conv1_1 = ConvBlock(128, 64, time_emb_dim) # 64 (up) + 64 (skip)
        self.up_conv1_2 = ConvBlock(64, 64, time_emb_dim)
        
        self.up2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.up_conv2_1 = ConvBlock(64, 32, time_emb_dim) # 32 (up) + 32 (skip)
        self.up_conv2_2 = ConvBlock(32, 32, time_emb_dim)

        self.out = nn.Conv2d(32, 1, kernel_size=1) 

    def forward(self, x, t):
        t_emb = self.time_mlp(t)

        # Down
        x1 = self.down1_1(x, t_emb)
        x1 = self.down1_2(x1, t_emb)
        p1 = self.pool1(x1)
        
        x2 = self.down2_1(p1, t_emb)
        x2 = self.down2_2(x2, t_emb)
        p2 = self.pool2(x2)

        # Bot
        bot = self.bot1(p2, t_emb)
        bot = self.attn(bot)
        bot = self.bot2(bot, t_emb)

        # Up
        u1 = self.up1(bot)
        u1 = torch.cat([u1, x2], dim=1)
        u1 = self.up_conv1_1(u1, t_emb)
        u1 = self.up_conv1_2(u1, t_emb)
        
        u2 = self.up2(u1)
        u2 = torch.cat([u2, x1], dim=1)
        u2 = self.up_conv2_1(u2, t_emb)
        u2 = self.up_conv2_2(u2, t_emb)

        return self.out(u2)