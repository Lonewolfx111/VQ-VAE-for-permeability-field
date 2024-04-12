import torch
import torch.nn as nn
import time
import scipy.io as scio
import matplotlib.pyplot as plt
import numpy as np
from vqvae import VQVAE


data0 = scio.loadmat('D:\\PycharmFiles\\VQVAE\\prior_model' + '/data.mat')
data1 = data0['data']
data_true = data1[:, 0].reshape(-1, 1)
data_sample = data1[:, 1:]
data_sample = data_sample.T

min = np.min(data_sample, axis=0).reshape(1, -1)
max = np.max(data_sample, axis=0).reshape(1, -1)
data_loader = (data_sample - np.tile(min, (data_sample.shape[0], 1))) / np.tile((max - min), (data_sample.shape[0], 1))

data_loader = data_loader.reshape(500, 1, 60, 60)
data_loader = torch.from_numpy(data_loader)
data_loader = data_loader.to(torch.float32)

batch_size = 64
lr = 1e-3
n_epochs = 100
l_w_embedding = 1
l_w_commitment = 0.25

model = VQVAE(input_dim=1, dim=16, n_embedding=64)

optimizer = torch.optim.Adam(model.parameters(), lr)

mse_loss = nn.MSELoss()

tic = time.time()
for e in range(n_epochs):
    total_loss = 0
    for x in data_loader:
        current_batch_size = x.shape[0]
        x = x.reshape(current_batch_size, 1, 60, 60)
        x_hat, ze, zq = model(x)
        l_reconstruct = mse_loss(x, x_hat)
        l_embedding = mse_loss(ze.detach(), zq)
        l_commitment = mse_loss(ze, zq.detach())
        loss = l_reconstruct + l_w_embedding * l_embedding + l_w_commitment * l_commitment
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * current_batch_size
        total_loss /= len(data_loader)
        toc = time.time()
    print(f'Epoch {e+1}, Total loss: {total_loss}, Elapsed {(toc - tic):.2f}s')
print('Done')

x_hat, ze, zq = model(data_loader)
print("Compressed array shape:", ze.shape)
print("Reconstructed array shape:", x_hat.shape)

data_loader = data_loader.detach().numpy().reshape(500, 3600)
data_loader = (data_loader * np.tile((max - min), (data_sample.shape[0], 1))) + (np.tile(min, (data_sample.shape[0], 1)))
x_hat = x_hat.detach().numpy().reshape(500, 3600)
x_hat = (x_hat * np.tile((max - min), (data_sample.shape[0], 1))) + (np.tile(min, (data_sample.shape[0], 1)))


plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title('Original Data')
plt.imshow(data_loader[0].reshape(60, 60), cmap='jet')
plt.colorbar(label='Permeability')
plt.subplot(1, 2, 2)
plt.title('Reconstructed Data')
plt.imshow(x_hat[0].reshape(60, 60), cmap='jet')
plt.colorbar(label='Permeability')
plt.show()
