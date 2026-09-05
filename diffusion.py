import torch

class NoiseSchedule():

    def __init__(self, T=1000, beta_s=1e-4, beta_e=0.02, device="cuda"):
        self.device = device
        self.T = T
        self.beta = torch.linspace(beta_s, beta_e, self.T, device=device) 
        self.alpha = 1.0 - self.beta
        self.alpha_cumprod = torch.cumprod(self.alpha, dim=0)


    def forward(self, x, t):
        noise = torch.randn_like(x).to(self.device)

        acp = self.alpha_cumprod[t].view(-1, 1, 1, 1)
        acp_sqrt = torch.sqrt(acp)
        acp_sqrt2 = torch.sqrt(1 - acp)

        final = (acp_sqrt*x) + (acp_sqrt2*noise)

        return final, noise

