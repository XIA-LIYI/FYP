import torch
import numpy as np
from tqdm import tqdm
from sklearn.feature_extraction.image import extract_patches_2d
import matplotlib.pyplot as plt

def get_patch(data, patch_size, num_patch):
	print("Extracting patches ......")
	patches = np.zeros(shape = (data.shape[0] * num_patch, patch_size, patch_size))
	for i in tqdm(range(data.shape[0])):
		patches[i * num_patch: (i + 1) * num_patch] = extract_patches_2d(data[i], (patch_size, patch_size), max_patches = num_patch, random_state = i)
	return patches

def get_num_patch(data, patch_size):
	height = data.shape[1]
	width = data.shape[2]
	num_patch = int(height * width / patch_size / patch_size * 2)
	return num_patch


def getMean(train_set, val_set):
	data = np.concatenate((train_set, val_set), axis = 0)
	return np.mean(data)

def getStd(train_set, val_set):
	data = np.concatenate((train_set, val_set), axis = 0)
	return np.std(data)

def preprocess(data, patch_size, fraction):
	num_train = int(fraction * data.shape[0])
	train_images = data[: num_train]
	val_images = data[num_train:]
	print("Shape of train images:", train_images.shape, "Shape of validation images:", val_images.shape)

	num_patch = get_num_patch(data, patch_size)
	print("Number of patch:", num_patch)
	train_set = get_patch(train_images, patch_size, num_patch).astype("float32")[:,np.newaxis]
	val_set = get_patch(val_images, patch_size, num_patch).astype("float32")[:,np.newaxis]

	mean = getMean(train_set, val_set)
	std = getStd(train_set, val_set)
	print("Data mean:", mean, "Data std:", std)

	train_tensor = torch.from_numpy(train_set)
	val_tensor = torch.from_numpy(val_set)
	print("Shape of train tensor:", train_tensor.shape, "Shape of val tensor:", val_tensor.shape)

	return train_tensor, val_tensor, mean, std



