from torch import nn
from torchinfo import summary
import numpy as np


class Descriminator(nn.Module):
    def __init__(self, im_size):
        super().__init__()

        self.ff = nn.Sequential(nn.Linear(np.prod(im_size), 300),
                                nn.LeakyReLU(),

                                nn.Linear(300, 400),
                                nn.LeakyReLU(),

                                nn.Linear(400, 500),
                                nn.LeakyReLU(),

                                nn.Linear(500, 100),
                                nn.LeakyReLU(),

                                nn.Linear(100, 1),
                                nn.Sigmoid())

    def forward(self, x):
        x = self.ff(x)

        return x


class Generator(nn.Module):
    def __init__(self, latent_dim, im_size):
        super().__init__()

        self.latent_dim = latent_dim

        self.ff = nn.Sequential(nn.ConvTranspose1d(latent_dim, latent_dim * 3, 4),
                                nn.InstanceNorm1d(latent_dim * 3),
                                nn.LeakyReLU(),

                                nn.ConvTranspose1d(latent_dim * 3, latent_dim * 4, 4),
                                nn.InstanceNorm1d(latent_dim * 4),
                                nn.LeakyReLU(),

                                nn.ConvTranspose1d(latent_dim * 4, latent_dim * 5, 4),
                                nn.InstanceNorm1d(latent_dim * 5),
                                nn.LeakyReLU(),

                                nn.ConvTranspose1d(latent_dim * 5, latent_dim * 5, 4),
                                nn.InstanceNorm1d(latent_dim * 5),
                                nn.LeakyReLU(),

                                nn.ConvTranspose1d(latent_dim * 5, np.prod(im_size), 4),
                                nn.Tanh())

    def forward(self, x):
        if x.dim() == 1:
            x = x.reshape((1, self.latent_dim, 1))

        x = self.ff(x)

        return x


if __name__ == "__main__":
    summary(Generator(100, (3, 64, 64)), input_size=(1, 100, 1))
