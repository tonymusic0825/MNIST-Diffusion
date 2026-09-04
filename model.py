import torch.nn as nn
import torch


class ConvBlock(nn.Module):

    def __init__(self, in_ch, out_ch, time_emb_dim):
        super().__init__()

        self.time_mlp = nn.Linear(time_emb_dim, out_ch)
        self.conv1 = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, padding=1, kernel_size=3)
        self.batchNorm = nn.BatchNorm2d(num_features=out_ch)
        self.relu = nn.ReLU()



    def forward(self, x, t_emb):
        x = self.conv1(x)
        time_val = self.time_mlp(t_emb).view(-1, x.shape[1], 1, 1)
        x = x + time_val 
        x = self.batchNorm(x)


        return self.relu(x)


class UNet(nn.Module):

    def __init__(self):

        super().__init__()

        self.time_emb = nn.Embedding(num_embeddings=1000, embedding_dim=32)
        self.time_proj = nn.Linear(32, 64)

        # Encoder
        self.down1 = ConvBlock(1, 16)
        self.pool1 = nn.MaxPool2d(kernel_size=2) # 14x14
        self.down2 = ConvBlock(16, 32)
        self.pool2 = nn.MaxPool2d(kernel_size=2) # 7x7

        # Bottleneck
        self.bot = ConvBlock(32, 64)

        # Encoder
        self.up1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.up_conv1 = ConvBlock(64, 32)
        self.up2 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.up_conv2 = ConvBlock(32, 16)

        # Output Layer
        self.out = nn.Conv2d(16, 1, kernel_size=1) 

    def forward(self, x, t):
        t_emb = self.time_emb(t)

        x1 = self.down1(x, t_emb)
        x2 = self.pool1(x1)
        x2 = self.down2(x2, t_emb)
        x3 = self.pool2(x2)

        bot_features = self.bot(x3, t_emb)

        x4 = self.up1(bot_features)
        x4 = torch.cat([x4, x2], dim=1)
        x4 = self.up_conv1(x4, t_emb)
        x4 = self.up2(x4)
        x4 = torch.cat([x4, x1], dim=1)
        x4 = self.up_conv2(x4, t_emb)

        return self.out(x4) 



        
