import torch
import torch.nn as nn
#from PIL import Image

class Noise_scheduler():
    def __init__(self, timesteps = 1000, beta_start = 1e-4, beta_end = 0.02, device = "cpu"):
        self.T = timesteps

        self.device = device

        self.betas = torch.linspace(beta_start, beta_end, self.T).to(device)

        self.alphas = 1.0 - self.betas.to(device)

        self.alpha_bars = torch.cumprod(self.alphas, dim = 0).to(device)

    def add_noise(self, x0, t):

        alpha_bar = self.alpha_bars[t].view(-1, 1, 1) # reshaping alpha_bar to make it broadcast-compatible at 't' timestep.
        noise = torch.randn_like(x0)

        x_t = torch.sqrt(alpha_bar)*x0 + torch.sqrt(1 - alpha_bar)*noise

        return x_t, noise

    def reverse_step(self, x_t, t, predicted_noise):
        beta_t = self.betas[t].view(-1, 1, 1)
        alpha_t = self.alphas[t].view(-1, 1, 1)
        alpha_bar_t = self.alpha_bars[t].view(-1, 1, 1)

        prev_x_t = (1/torch.sqrt(alpha_t)) * (x_t - (beta_t/torch.sqrt(1-alpha_bar_t)) * predicted_noise)

        if t[0].item() > 0:
            noise = torch.randn_like(x_t)
            prev_x_t = prev_x_t + torch.sqrt(beta_t) * noise
        
        return prev_x_t

'''
# some test code here:

batch_size = 4
x0 = torch.randn(batch_size, 142, 2)
t = torch.randint(0, 1000, (4,))

noise_scheduler = Noise_scheduler()
x_t, noise = noise_scheduler.add_noise(x0, t)

print(t.shape)
print(x_t.shape)
print(noise.shape)
'''