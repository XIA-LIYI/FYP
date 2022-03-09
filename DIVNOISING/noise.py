import torch
import matplotlib.pyplot as plt
import numpy as np
import pickle

class GaussianMixtureModel:
	
	def __init__(self, **params):


		self.gaussian = params.get("gaussian")
		self.coeff = params.get("coeff")
		max_signal = params.get("max_signal")
		min_signal = params.get("min_signal")
		self.max_signal = torch.Tensor([max_signal]).cuda()
		self.min_signal = torch.Tensor([min_signal]).cuda()
		if (params.get("weight") is None):
			self.weight = self.create_weight(self.gaussian, self.coeff, max_signal, min_signal)
		else:
			self.weight = torch.Tensor(params.get("weight")).cuda()
		self.min_variance = params.get("min_variance")
		self.limit = torch.Tensor([1e-10]).cuda()


	def create_weight(self, gaussian, coeff, max_signal, min_signal):
		weight = np.random.randn(gaussian * 3, coeff)
		weight[gaussian: gaussian * 2, 1] = np.log(max_signal - min_signal)
		weight = weight.astype(np.float32)
		weight = torch.from_numpy(weight).float().cuda()
		weight.requires_grad = True
		return weight

	def get_signal_noisy_pair(self, noisy, signal):
		stepsize = noisy[0].size
		result = np.zeros((stepsize * noisy.shape[0], 2))
		for i in range(noisy.shape[0]):
			result[stepsize * i: stepsize * (i + 1), 0] = signal[0].reshape(1, stepsize)
			result[stepsize * i: stepsize * (i + 1), 1] = noisy[i].reshape(1, stepsize)
		np.random.shuffle(result)
		return result

	def polynomial_regressor(self, weight, signals):
		value = 0 
		for i in range(weight.shape[0]):
			value += weight[i] * (((signals - self.min_signal) / (self.max_signal- self.min_signal)) ** i)
		return value

	def compute_gaussian(self, signals):
		params = {}
		mean = []
		std = []
		fraction = []
		for i in range(self.gaussian):
			mean_temp = self.polynomial_regressor(self.weight[i, :], signals)
			variance_temp = self.polynomial_regressor(torch.exp(self.weight[i + self.gaussian, :]), signals)
			variance_temp = torch.clamp(variance_temp, min = self.min_variance)
			std_temp = torch.sqrt(variance_temp)
			fraction_temp  = torch.exp(self.limit + self.polynomial_regressor(self.weight[i + 2 * self.gaussian, :], signals))

			mean.append(mean_temp)
			std.append(std_temp)
			fraction.append(fraction_temp)
		
		fraction_sum = 0
		for i in range(self.gaussian):
			fraction_sum += fraction[i]
		for i in range(self.gaussian):
			fraction[i] = fraction[i] / fraction_sum

		mean_sum = 0
		for i in range(self.gaussian):
			mean_sum += fraction[i] * mean[i]
		for i in range(self.gaussian):
			mean[i] = mean[i] - mean_sum + signals

		params["mean"] = mean
		params["std"] = std
		params["fraction"] = fraction
		return params

	def density(self, noisy, mean, std):
		result = -((noisy - mean) ** 2) / (2.0 * std * std)
		return torch.exp(result) / torch.sqrt((2.0 * np.pi) * std * std)

	def compute_likelihood(self, signals, noisy):
		params = self.compute_gaussian(signals)
		likelihood = 0
		mean = params["mean"]
		std = params["std"]
		fraction = params["fraction"]
		for i in range(self.gaussian):
			likelihood += self.density(noisy, mean[i], std[i]) * fraction[i]
		return likelihood + self.limit

	def train(self, noisy, signal, learning_rate, batch_size, epochs, save_path):
		pairs = self.get_signal_noisy_pair(noisy, signal)
		print(pairs.shape[0])
		counter = 0
		optimizer = torch.optim.Adam([self.weight], lr = learning_rate)

		for epoch in range(epochs):
			loss = 0
			if (counter + 1) * batch_size >= pairs.shape[0]:
				counter = 0
				np.random.shuffle(pairs)

			batch = pairs[counter * batch_size: (counter + 1) * batch_size, :]
			batch_noisy = batch[:, 1].astype(np.float32)
			batch_noisy = torch.from_numpy(batch_noisy).float().cuda()
			batch_signal = batch[:, 0].astype(np.float32)
			batch_signal = torch.from_numpy(batch_signal).float().cuda()
			likelihood = self.compute_likelihood(batch_signal, batch_noisy)
			loss += torch.mean(-torch.log(likelihood))
			
			if (epoch % 100 == 0):
				print(epoch, loss.item())

			counter += 1
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
		trained_weight = self.weight.cpu().detach().numpy()
		np.savez(save_path, weight = trained_weight)

