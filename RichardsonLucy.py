import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
from skimage import data, util


def create_transformation_matrix():
    # Create a Gaussian 3x3 blur kernel
    gaussian_kernel = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 16

    # Create the 128x128 transformation matrix A
    A = np.zeros((128 * 128, 128 * 128))
    for i in range(128):
        for j in range(128):
            # Position of the current pixel in the vectorized image
            index = i * 128 + j

            # Iterate over the blur kernel
            for ki in range(-1, 2):
                for kj in range(-1, 2):
                    # Calculate the position in the vectorized image after applying the kernel
                    row = (i + ki) % 128  # Modulo for edge handling
                    col = (j + kj) % 128
                    A[index, row * 128 + col] = gaussian_kernel[ki + 1, kj + 1]

    return A


def richardson_lucy_iteration(x_n, A, y):
    # Calculate A*x^(n)
    Ax_n = A.dot(x_n)

    # Use broadcasting to vectorize the calculation of the adjusted observation and the column sum normalizer
    adjusted_observation = (A * y).sum(axis=0)
    column_sum_normalizer = A.sum(axis=0)

    # Update rule for x^(n+1) with vectorized operations
    x_n_plus_1 = (x_n / column_sum_normalizer) * (adjusted_observation / Ax_n)

    return x_n_plus_1


def estimate(iterations, A, image):
    x_n_plus = image
    for _ in range(iterations):
        x_n_plus = richardson_lucy_iteration(x_n_plus, A, image)
    return x_n_plus


# Create sample data
test_image_original = data.camera()
scale_factor = [128 / test_image_original.shape[0], 128 / test_image_original.shape[1]]
test_image = ndimage.zoom(test_image_original, scale_factor, order=3)

# Create transformation matrix and apply it to the test image
A = create_transformation_matrix()
blurry_image_vector = A.dot(test_image.flatten())
blurry_image = blurry_image_vector.reshape(128, 128)

# Add poisson noise
blurry_image_uint8 = np.uint8(blurry_image)
blurry_noisy_image = util.random_noise(blurry_image_uint8, mode="poisson")

# Richardson Lucy
estimated_image = estimate(2, A, blurry_noisy_image.flatten())
estimated_image = estimated_image.reshape((128, 128))

# Plotting the results
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

axes[0].imshow(test_image, cmap='gray')  # Original Scene
axes[0].set_title('Original Scene')
axes[0].set_xticks([])

axes[1].imshow(blurry_noisy_image, cmap='gray')  # Blur + Noise
axes[1].set_title('Blur + Noise')
axes[1].set_xticks([])

axes[2].imshow(estimated_image, cmap='gray')  # Restored with Richardson-Lucy
axes[2].set_title('Richardson-Lucy')
axes[2].set_xticks([])

plt.show()

exit(0)
