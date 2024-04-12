import torch
import torch.nn as nn


class ResidualBlock(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(dim, dim, 3, 1, 1)
        self.conv2 = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        tmp = self.relu(x)
        tmp = self.conv1(tmp)
        tmp = self.relu(tmp)
        tmp = self.conv2(tmp)
        return x + tmp


class VQVAE(nn.Module):

    def __init__(self, input_dim, dim, n_embedding):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(input_dim, 32, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(64, dim, 3, 1, 1),
            nn.ReLU()
            # ResidualBlock(dim),
            # ResidualBlock(dim)
        )
        self.vq_embedding = nn.Embedding(n_embedding, dim)
        self.vq_embedding.weight.data.uniform_(-1.0 / n_embedding, 1.0 / n_embedding)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(dim, 32, 3, 1, 1),
            nn.ReLU(),
            # ResidualBlock(dim),
            # ResidualBlock(dim),
            nn.ConvTranspose2d(32, 64, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, input_dim, 4, 2, 1),
        )
        self.n_downsample = 2

    def forward(self, x):
        # encode
        ze = self.encoder(x)

        # ze: [N, C, H, W]
        # embedding [K, C]
        embedding = self.vq_embedding.weight.data
        N, C, H, W = ze.shape
        K, _ = embedding.shape
        embedding_broadcast = embedding.reshape(1, K, C, 1, 1)
        ze_broadcast = ze.reshape(N, 1, C, H, W)
        distance = torch.sum((embedding_broadcast - ze_broadcast) ** 2, 2)
        nearest_neighbor = torch.argmin(distance, 1)
        # make C to the second dim
        zq = self.vq_embedding(nearest_neighbor).permute(0, 3, 1, 2)
        # stop gradient
        decoder_input = ze + (zq - ze).detach()

        # decode
        x_hat = self.decoder(decoder_input)
        return x_hat, ze, zq
