import torch
import numpy as np 
import matplotlib.pyplot as plt
import torch.nn.functional as F
from tqdm import tqdm

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
	result = 0.5 * torch.sum(mean ** 2 - var + var.exp() - 1)
	return result

def train(model, model_path, model_name, data_mean, data_std, train_loader, val_loader, noise, gaussian_std,
	epochs, batch_size, learning_rate, kl_limit):
	
	### Configure the training ###
	optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
	data_mean = torch.Tensor([data_mean]).cuda()
	data_std = torch.Tensor([data_std]).cuda()
	train_loss_total = []
	val_loss_total = []
	recon_loss_total = []
	kl_loss_total = []

	for epoch in range(1, epochs + 1):
		train_loss_per_epoch = []
		recon_loss_per_epoch = []
		kl_loss_per_epoch = []

		for step, (noisy, _) in enumerate(tqdm(train_loader)):

			noisy = noisy.cuda()
			noisy = (noisy - data_mean) / data_std
			mean, logvar, signal = model(noisy)
			recon_loss = compute_recon_loss(noisy, signal, data_mean, data_std, noise, gaussian_std)
			if (epoch > 10):
				kl_loss = compute_kl_loss(mean, logvar) / noisy.numel()
				if (kl_loss <= kl_limit):
					print("Training stops due to posterior collapse")
					return None, None, None, None
			else:
				kl_loss = torch.Tensor([0]).cuda()



			loss = recon_loss + kl_loss
			train_loss_per_epoch.append(loss.item())
			recon_loss_per_epoch.append(recon_loss.item())
			kl_loss_per_epoch.append(kl_loss.item())

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()


		train_loss_avg = np.mean(train_loss_per_epoch).item()
		recon_loss_avg = np.mean(recon_loss_per_epoch).item()
		kl_loss_avg = np.mean(kl_loss_per_epoch).item()
		output = "Epoch[{}] Training Loss: {} Reconstruction Loss: {} KL Loss: {}"
		print(output.format(epoch, train_loss_avg, recon_loss_avg, kl_loss_avg))

		if (epoch % 5 == 0):
			torch.save(model, model_path + str(epoch) + model_name)

		train_loss_total.append(train_loss_avg)
		recon_loss_total.append(recon_loss_avg)
		kl_loss_total.append(kl_loss_avg)

		val_loss_per_epoch = []
		with torch.no_grad():
			for step, (noisy, _) in enumerate(tqdm(val_loader)):
				noisy = noisy.cuda()
				noisy = (noisy - data_mean) / data_std
				mean, var, signal = model(noisy)
				val_recon_loss = compute_recon_loss(noisy, signal, data_mean, data_std, noise, gaussian_std)
				val_kl_loss = compute_kl_loss(mean, var) / noisy.numel()
				if (epoch <= 10):
					val_kl_loss = val_kl_loss * 0
				val_loss = val_recon_loss + val_kl_loss
				val_loss_per_epoch.append(val_loss.item())

		val_loss_avg = np.mean(val_loss_per_epoch).item()
		output = "Epoch[{}] Val Loss: {}"
		print(output.format(epoch, val_loss_avg))
		val_loss_total.append(val_loss_avg)

	return train_loss_total, recon_loss_total, kl_loss_total, val_loss_total

