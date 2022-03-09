import torch
import numpy as np
import matplotlib.pyplot as plt

from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def compute_mmse(img, model, image_size, num_samples):
	predicted_imgs = []
	img_tensor = torch.Tensor(img)
	img_tensor = img_tensor.view(1, 1, img.shape[0], img.shape[1]).cuda()

	for i in range(num_samples):
		predicted_tensor = model.predict(img_tensor)
		predicted_img = predicted_tensor.cpu().detach().numpy().reshape(image_size, image_size)
		predicted_imgs.append(predicted_img)

	return np.mean(predicted_imgs, axis = 0), predicted_imgs


def compute_ssim(original, predicted):
	return structural_similarity(original, predicted)

def compute_psnr(original, predicted):
    mse = np.mean(np.square(original - predicted))
    return 20 * np.log10(np.max(original)-np.min(original)) - 10 * np.log10(mse)

def predict(noisy, signal, model, image_size, num_samples, num_display, plot = False):
	mmse_list = []
	psnr_gt_mmse_list = []
	psnr_noisy_mmse_list = []
	ssim_list = []
	num_imgs = noisy.shape[0]
	for index, img in enumerate(noisy):
		print("Image", index)
		if (image_size == noisy.shape[1]):
			shiftx = 0
			shifty = 0
		else:
			shiftx = int(np.random.randint(0, noisy.shape[1] - image_size))
			shifty = int(np.random.randint(0, noisy.shape[1] - image_size))
		img = img[shiftx: shiftx + image_size, shifty: shifty + image_size]
		ground_truth = signal[shiftx: shiftx + image_size, shifty: shifty + image_size]

		mmse, predicted_imgs = compute_mmse(img, model, image_size, num_samples)

		if (plot == True):
			plt.figure(figsize = (20, 6.75))

			ax = plt.subplot(1, num_display + 3 , 1)			
			ax.get_xaxis().set_visible(False)
			ax.get_yaxis().set_visible(False)
			plt.imshow(img, cmap='magam')
			plt.title('input')

			for i in range(num_display):
				ax = plt.subplot(1, num_display + 3, i + 2)
				ax.get_xaxis().set_visible(False)
				ax.get_yaxis().set_visible(False)
				plt.imshow(predicted_imgs[i], cmap='magam')
				plt.title('prediction' + str(i + 1))

			ax = plt.subplot(1, num_display + 3, num_display + 2)
			ax.get_xaxis().set_visible(False)
			ax.get_yaxis().set_visible(False)
			plt.imshow(mmse, cmap='magam')
			plt.title('mmse')

			ax = plt.subplot(1, num_display + 3, num_display + 3)
			ax.get_xaxis().set_visible(False)
			ax.get_yaxis().set_visible(False)
			plt.imshow(ground_truth)
			plt.title('ground truth')

			plt.show()

		psnr_gt_mmse = compute_psnr(ground_truth, mmse)
		psnr_gt_mmse_list.append(psnr_gt_mmse)
		print("PSNR between ground truth and mmse:", psnr_gt_mmse)

		psnr_noisy_mmse = compute_psnr(img, mmse)
		psnr_noisy_mmse_list.append(psnr_noisy_mmse)
		print("PSNR between noisy image and mmse:", psnr_noisy_mmse)

		ssim = compute_ssim(ground_truth, mmse)
		ssim_list.append(ssim)
		print("SSIM between ground truth and mmse:", ssim)

		mmse_list.append(mmse)
	print("Average PSNR between ground truth and mmse:", np.mean(psnr_gt_mmse_list), "with Standard Deviation", np.std(psnr_gt_mmse_list))
	print("Average PSNR between gnoisy image and mmse:", np.mean(psnr_noisy_mmse_list), "with Standard Deviation", np.std(psnr_noisy_mmse_list))
	print("Average PSNR increase:", np.mean(np.array(psnr_gt_mmse_list) - np.array(psnr_noisy_mmse_list)))
	print("Average SSIM between ground truth and mmse:", np.mean(ssim_list), "with Standard Deviation", np.std(ssim_list))

