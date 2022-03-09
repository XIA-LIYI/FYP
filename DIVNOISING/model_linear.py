import torch
import torch.nn as nn

class DIVNOISING(nn.Module):
	def __init__(self, data_mean, data_std):
		super(DIVNOISING, self).__init__()
		self.Encoder = nn.Sequential(
			nn.Linear(784, 256),
			nn.ReLU(),
			nn.Linear(256, 64),
			nn.ReLU(),
			nn.Linear(64, 32),
			nn.ReLU(),
		)
		self.mean = nn.Linear(32, 32)
		self.logvar = nn.Linear(32, 32)
		self.Decoder = nn.Sequential(
			nn.Linear(32, 64),
			nn.ReLU(),
			nn.Linear(64, 256),
			nn.ReLU(),
			nn.Linear(256, 784),
			nn.Sigmoid()
		)
		self.data_mean = data_mean
		self.data_std = data_std

	def encode(self, x):
		x = (x - self.data_mean) / self.data_std
		h = self.Encoder(x)
		mean = self.mean(h)
		logvar = self.logvar(h)
		return mean, logvar 

	def predict(self, x):
		x_normalized = (x - self.data_mean) / self.data_std
		recon_normalized = self.forward(x_normalized)[2]
		recon = recon_normalized * self.data_std + self.data_mean
		return recon

	def reparameterize(self, mean, logvar):
		std = torch.exp(0.5 * logvar)
		eps = torch.randn_like(mean)
		z = mean + eps * std
		return z

	def forward(self, x):
		h = self.Encoder(x)
		mean = self.mean(h)
		logvar = self.logvar(h)
		z = self.reparameterize(mean, logvar)
		recon = self.Decoder(z)
		return mean, logvar, recon

