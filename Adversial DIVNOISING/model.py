import torch
import torch.nn as nn

class Encoder(nn.Module):
	def __init__(self):
		super(Encoder, self).__init__()
		self.layer = nn.Sequential(
			nn.Conv2d(1, 32, 3, 1, 1),
			nn.ReLU(),
			#nn.BatchNorm2d(32),
			nn.Conv2d(32, 32, 3, 1, 1),
			nn.ReLU(),
			# nn.BatchNorm2d(32),
			nn.Conv2d(32, 64, 3, 1, 1),
			nn.ReLU(),
			# nn.BatchNorm2d(64),
			nn.MaxPool2d(2, 2),
			nn.Conv2d(64, 64, 3, 1, 1),
			nn.ReLU(),
			# nn.BatchNorm2d(64),				
			nn.Conv2d(64, 64, 3, 1, 1),
			nn.ReLU(),
			# nn.BatchNorm2d(64),
			nn.MaxPool2d(2, 2)
		)
		self.mean = nn.Conv2d(64, 64, 3, 1, 1)
		self.logvar = nn.Conv2d(64, 64, 3, 1, 1)

	def forward(self, x):
		h = self.layer(x)
		mean = self.mean(h)
		logvar = self.logvar(h)
		return mean, logvar

class Decoder(nn.Module):
	def __init__(self):
		super(Decoder, self).__init__()
		self.layer = nn.Sequential(
			nn.Conv2d(64, 64, 3, 1, 1),
			nn.ReLU(),
			# nn.BatchNorm2d(64),
			nn.Conv2d(64, 64, 3, 1, 1),
			nn.ReLU(),
			# nn.BatchNorm2d(64),
			nn.ConvTranspose2d(64, 64, 2, 2),
			nn.Conv2d(64, 32, 3, 1, 1),
			nn.ReLU(),
			# nn.BatchNorm2d(32),
			nn.Conv2d(32, 32, 3, 1, 1),
			nn.ReLU(),
			# nn.BatchNorm2d(32),
			nn.ConvTranspose2d(32, 32, 2, 2),
			nn.Conv2d(32, 1, 3, 1, 1)
		)

	def forward(self, z):
		return self.layer(z)

class DIVNOISING(nn.Module):
	def __init__(self, data_mean, data_std):
		super(DIVNOISING, self).__init__()
		self.Encoder = Encoder()
		self.Decoder = Decoder()
		self.data_mean = data_mean
		self.data_std = data_std


	def encode(self, x):
		x_normalized = (x - self.data_mean) / self.data_std
		return self.Encoder(x_normalized)

	def decode(self, z):
		recon_normalized = self.Decoder(z)
		return recon_normalized * self.data_std + self.data_mean

	def predict(self, x):
		x_normalized = (x - self.data_mean) / self.data_std
		recon_normalized = self.forward(x_normalized)
		recon = recon_normalized * self.data_std + self.data_mean
		return recon

	def reparameterize(self, mean, logvar):
		std = torch.exp(0.5 * logvar)
		eps = torch.randn_like(mean)
		z = mean + eps * std
		return z

	def forward(self, x):
		mean, logvar = self.Encoder(x)
		z = self.reparameterize(mean, logvar)
		recon = self.Decoder(z)
		return recon

class Discriminator(nn.Module):
	def __init__(self):
		super(Discriminator, self).__init__()
		self.layer = nn.Sequential(
			nn.Conv2d(64, 32, 3, 1, 1),
			nn.ReLU(),
			# nn.BatchNorm2d(32),
			nn.Conv2d(32, 32, 3, 1, 1),
			nn.ReLU(),
			# nn.BatchNorm2d(32),
			nn.Conv2d(32, 1, 3, 1, 1),
			nn.Sigmoid()
		)
	def forward(self, x):
		return self.layer(x)