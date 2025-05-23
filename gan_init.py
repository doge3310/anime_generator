from torch import nn
import numpy as np


class Descriminator(nn.Module):
    def __init__(self, im_size):
        super().__init__()

        self.ff = nn.Sequential(nn.Linear(np.prod(im_size), 500),
                                nn.LeakyReLU(),

                                nn.Linear(500, 300),
                                nn.LeakyReLU(),

                                nn.Linear(300, 100),
                                nn.LeakyReLU(),

                                nn.Linear(100, 1),
                                nn.Sigmoid())

    def forward(self, x):
        x = self.ff(x)

        return x


class Generator(nn.Module):
    def __init__(self, latent_dim, im_size):
        super().__init__()

        self.ff = nn.Sequential(nn.Linear(latent_dim, latent_dim * 3),
                                nn.ReLU(),

                                nn.Linear(latent_dim * 3, latent_dim * 4),
                                nn.ReLU(),

                                nn.Linear(latent_dim * 4, np.prod(im_size)),
                                nn.Tanh())

    def forward(self, x):
        x = self.ff(x)

        return x
