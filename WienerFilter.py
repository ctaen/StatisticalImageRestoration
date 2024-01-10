import numpy as np
from scipy.fftpack import fft2, ifft2, fftshift, ifftshift
from skimage import data, util
from scipy import ndimage
import matplotlib.pyplot as plt


def create_gaussian_kernel(size, sigma=1):
    # Create a Gaussian blur kernel
    ax = np.linspace(-(size - 1) / 2., (size - 1) / 2., size)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-0.5 * (np.square(xx) + np.square(yy)) / np.square(sigma))
    return kernel / np.sum(kernel)


def wiener_filter(blurry_noisy_image, kernel, beta, noise_var):
    # Size of the image and kernel
    image_shape = blurry_noisy_image.shape
    kernel_shape = kernel.shape

    # Zero padding of the kernel to match the image size
    padded_kernel = np.zeros(image_shape)
    pad_x = (image_shape[0] - kernel_shape[0]) // 2
    pad_y = (image_shape[1] - kernel_shape[1]) // 2
    padded_kernel[pad_x:pad_x + kernel_shape[0], pad_y:pad_y + kernel_shape[1]] = kernel

    # Fourier transform of the image and the padded kernel
    blurry_noisy_fft = fft2(blurry_noisy_image)
    kernel_fft = fft2(fftshift(padded_kernel))

    # Calculation of the Wiener filter in the frequency domain
    kernel_fft_conj = np.conj(kernel_fft)
    wiener_filter_fft = kernel_fft_conj / (np.abs(kernel_fft) ** 2 + beta * noise_var)

    # Application of the Wiener filter
    restored_image_fft = wiener_filter_fft * blurry_noisy_fft
    restored_image = np.real(ifft2(restored_image_fft))

    return restored_image


# Generate test image and kernel
test_image_original = data.camera()
scale_factor = [256 / test_image_original.shape[0], 256 / test_image_original.shape[1]]
test_image = ndimage.zoom(test_image_original, scale_factor, order=3)

kernel = create_gaussian_kernel(32, sigma=1)

# Create blurred and noisy image
blurry_image = ndimage.convolve(test_image, kernel)
blurry_noisy_image = util.random_noise(blurry_image, mode="gaussian", var=0.005)

# Parameters for the Wiener filter
beta = 1000  # Regularization parameter
noise_var = 0.005  # Variance of white Gaussian noise

# Application of the Wiener filter
restored_image = wiener_filter(blurry_noisy_image, kernel, beta, noise_var)

# Plotting the results
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(test_image, cmap='gray')  # Original Scene
axes[0].set_title('Original Scene')
axes[0].set_xticks([])

axes[1].imshow(blurry_noisy_image, cmap='gray')  # Blur + Noise
axes[1].set_title('Blur + Noise')
axes[1].set_xticks([])

axes[2].imshow(restored_image, cmap='gray')  # Restored with Wiener Filter
axes[2].set_title('Restored with Wiener Filter')
axes[2].set_xticks([])

plt.show()

exit(0)
