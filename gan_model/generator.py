import argparse

import torch
import numpy as np
from .model import Generator
from tqdm import tqdm


def renorm(x):
    x = x.clamp_(-1, 1)
    x = (x + 1) / 2
    x = x[0].permute(1, 2, 0).detach().numpy()
    x = (x*255).astype(np.uint8)
    return x

def generate(g_ema, device, mean_latent):
    with torch.no_grad():
        g_ema.eval()
        sample_z = torch.randn(1, 512, device=device)
        sample, _ = g_ema([sample_z], truncation=0.8, truncation_latent=mean_latent)
        sample = renorm(sample)
        return sample


if __name__ == "__main__":
    latent = 512
    n_mlp = 8
    size = 512
    weight_path = "./170000_e-ema.pt"
    device = "cpu"
    truncation_mean = 0.8

    g_ema = Generator(size, latent, n_mlp, channel_multiplier=2)

    g_ema.load_state_dict(torch.load(weight_path))
    g_ema = g_ema.to(device)

    with torch.no_grad():
        mean_latent = g_ema.mean_latent(truncation_mean)

    sample = generate(g_ema, device, mean_latent)
    """
    sample: 
    """
