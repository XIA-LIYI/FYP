import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.utils.data as Data 
from tifffile import imread

from noise import GaussianMixtureModel
from data import preprocess
from model import DIVNOISING
from predict import predict
import train

############################
## Step 0: Configure Path ##
############################
data_path = "../Convallaria_diaphragm/"
calibration_data_name = "20190726_tl_50um_500msec_wf_130EM_FD.tif"
data_name = "20190520_tl_25um_50msec_05pc_488_130EM_Conv.tif"
model_path = "./Convallaria/"
noise_model_name = "noise_model.npz"
divnoising_model_name = "divnoising_model_last.net"
divnoising_data_parameters = "divnoising_data_parameters.npz"
noise_model_trained = True
divnoising_model_trained = False


#########################################
## Step 1: Train Gaussian Noise Model  ##
#########################################

### import data ###

noisy = imread(data_path + calibration_data_name)

### configure noise model parameters ###
num_of_gaussian = 3
num_of_coeff = 2
epochs = 5000
batch_size = 25000
min_variance = 50
weight = None

### prepare data ###
signal = np.mean(noisy[:, ...],axis=0)[np.newaxis,...]

### visualize the signal ###
plt.figure(figsize=(12, 12))
plt.subplot(1, 2, 2)
plt.title(label='average (ground truth)')
plt.imshow(signal[0],cmap='gray')
plt.subplot(1, 2, 1)
plt.title(label='single raw image')
plt.imshow(noisy[0],cmap='gray')
plt.show()

### train noise model ###
max_signal = np.max(signal)
min_signal = np.min(signal)
if (noise_model_trained):
	weight = np.load(model_path + noise_model_name)["weight"]
noise_model = GaussianMixtureModel(weight = weight, gaussian = num_of_gaussian, coeff = num_of_coeff, max_signal = max_signal, min_signal = min_signal, min_variance = min_variance)
if (not noise_model_trained):
	noise_model.train(noisy, signal, 0.001, batch_size, epochs, model_path + noise_model_name)


##############################
## Step 2: Train DIVNOISING ##
##############################

### import data ###
noisy = imread(data_path + data_name)

### configure training parameters ###
patch_size = 128
train_fraction = 0.85
batch_size = 32
epochs = 10
learning_rate = 0.01
kl_limit = 1e-5


if (divnoising_model_trained):
	model = torch.load(model_path + divnoising_model_name)
else:
	### preprocess data ###
	train_loss, recon_loss, kl_loss, val_loss = None, None, None, None
	train_tensor, val_tensor, mean, std = preprocess(noisy, patch_size, train_fraction)
	train_loader = Data.DataLoader(dataset = Data.TensorDataset(train_tensor, train_tensor), batch_size = batch_size, shuffle = True)
	val_loader = Data.DataLoader(dataset = Data.TensorDataset(val_tensor, val_tensor), batch_size = batch_size, shuffle = True)

	### training ###
	while train_loss is None:
		model = DIVNOISING(mean, std).cuda()
		train_loss, recon_loss, kl_loss, val_loss = train.train(model, model_path + divnoising_model_name, mean, std, train_loader, val_loader, noise_model,
		epochs, batch_size, learning_rate, kl_limit)


	### plot loss ###
	plt.figure(figsize=(18, 3))
	plt.subplot(1,3,1)
	plt.plot(train_loss, label='training')
	plt.plot(val_loss, label='validation')
	plt.xlabel("epochs")
	plt.ylabel("loss")
	plt.legend()

	plt.subplot(1,3,2)
	plt.plot(recon_loss, label='training')
	plt.xlabel("epochs")
	plt.ylabel("reconstruction loss")
	plt.legend()

	plt.subplot(1,3,3)
	plt.plot(kl_loss, label='training')
	plt.xlabel("epochs")
	plt.ylabel("KL loss")
	plt.legend()
	plt.show()

##############################
## Step 3: Predicte Signal  ##
##############################

### import data ###
noisy = imread(data_path + data_name).astype("float32")

### configure parameters###
plot = True
num_samples = 100
num_display = 5
predict_input = noisy[: 1]

### predict ###
predict(predict_input, model, noise_model, num_samples, num_display, plot = True)