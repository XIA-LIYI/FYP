import torch
import numpy as np 
import matplotlib.pyplot as plt
import torch.nn.functional as F
from tqdm import tqdm
import predict

def compute_recon_loss_with_noise(noisy, signal, data_mean, data_std, noise):
	signal = signal * data_std + data_mean
	noisy = noisy * data_std + data_mean

	signal = signal.permute(1, 0, 2, 3)
	noisy = noisy.permute(1, 0, 2, 3)

	likelihood = torch.log(noise.compute_likelihood(signal, noisy))

	result = -torch.mean(likelihood)
	return result

def compute_recon_loss_without_noise(noisy, signal, gaussian_std, data_std):
	return torch.mean((noisy - signal) ** 2) / (2.0* (gaussian_std / data_std) ** 2)

def compute_recon_loss(noisy, signal, data_mean, data_std, noise, gaussian_std):
	if (noise is not None):
		result = compute_recon_loss_with_noise(noisy ,signal, data_mean, data_std, noise)
	else:
		result = compute_recon_loss_without_noise(noisy, signal ,gaussian_std, data_std)
	return result

def compute_kl_loss(mean, var):
	### not used ###
	result = 0.5 * torch.sum(mean ** 2 - var + var.exp() - 1)
	return result

def train(model, discriminator, path, model_name, loss_name, data_mean, data_std, train_loader, val_loader, noise, gaussian_std,
	epochs, batch_size, learning_rate, kl_limit):
	
	### Configure the training ###
	encoder_optim = torch.optim.Adam(model.Encoder.parameters(), lr = learning_rate)
	g_optim = torch.optim.Adam(model.Encoder.parameters(), lr = learning_rate)
	decoder_optim = torch.optim.Adam(model.Decoder.parameters(), lr = learning_rate)
	dis_optim = torch.optim.Adam(discriminator.parameters(), lr = learning_rate)

	data_mean = torch.Tensor([data_mean]).cuda()
	data_std = torch.Tensor([data_std]).cuda()
	val_loss_total = []
	recon_loss_total = []
	d_loss_total = []
	g_loss_total = []
	counter = 0

	for epoch in range(1, epochs + 1):
		recon_loss_per_epoch = []
		d_loss_per_epoch = []
		g_loss_per_epoch = []

		for step, (noisy, _) in enumerate(tqdm(train_loader)):

			noisy = noisy.cuda()
			noisy = (noisy - data_mean) / data_std
			mean, logvar = model.Encoder(noisy)
			z = model.reparameterize(mean, logvar)
			signal = model.Decoder(z)
			recon_loss = compute_recon_loss(noisy, signal, data_mean, data_std, noise, gaussian_std)
			recon_loss_per_epoch.append(recon_loss.item())
			encoder_optim.zero_grad()
			decoder_optim.zero_grad()
			recon_loss.backward()
			encoder_optim.step()
			decoder_optim.step()

			# Discriminator
			z_real = torch.autograd.Variable(torch.randn(noisy.shape[0], 64, 32, 32)).cuda()
			mean, logvar = model.Encoder(noisy)
			z_fake = model.reparameterize(mean, logvar)

			d_real, d_fake = discriminator(z_real), discriminator(z_fake)
			d_loss = -torch.mean(torch.log(d_real + 1e-15) + torch.log(1 - d_fake  + 1e-15))
			d_loss_per_epoch.append(d_loss.item())
			dis_optim.zero_grad()
			d_loss.backward()
			dis_optim.step()

			# Encoder
			if (epoch > 2):
				mean, logvar = model.Encoder(noisy)
				z_fake = model.reparameterize(mean, logvar)
				d_fake = discriminator(z_fake)

				g_loss = -torch.mean(torch.log(d_fake + 1e-15))
				g_loss_per_epoch.append(g_loss.item())
				g_optim.zero_grad()
				g_loss.backward()
				g_optim.step()

		recon_loss_avg = np.mean(recon_loss_per_epoch).item()
		d_loss_avg = np.mean(d_loss_per_epoch).item()
		g_loss_avg = np.mean(g_loss_per_epoch).item()
		output = "Epoch[{}] Reconstruction Loss: {} D Loss: {} G Loss {}"
		print(output.format(epoch, recon_loss_avg, d_loss_avg, g_loss_avg))
		recon_loss_total.append(recon_loss_avg)
		d_loss_total.append(d_loss_avg)
		g_loss_total.append(g_loss_avg)
		if (epoch % 5 == 0):
			torch.save(model, path + str(epoch) + model_name)

		val_loss_per_epoch = []
		with torch.no_grad():
			for step, (noisy, _) in enumerate(tqdm(val_loader)):
				counter += 1
				noisy = noisy.cuda()
				noisy = (noisy - data_mean) / data_std
				mean, logvar = model.Encoder(noisy)
				z = model.reparameterize(mean, logvar)
				signal = model.Decoder(z)
				val_recon_loss = compute_recon_loss(noisy, signal, data_mean, data_std, noise, gaussian_std)

				val_loss = val_recon_loss
				val_loss_per_epoch.append(val_loss.item())

		val_loss_avg = np.mean(val_loss_per_epoch).item()
		output = "Epoch[{}] Val Loss: {}"
		print(output.format(epoch, val_loss_avg))
		val_loss_total.append(val_loss_avg)
		
	np.savez(path + loss_name, recon_loss = recon_loss_total, g_loss = g_loss_total, d_loss = d_loss_total, val_loss = val_loss_total)

	return recon_loss_total, d_loss_total, g_loss_total, val_loss_total

