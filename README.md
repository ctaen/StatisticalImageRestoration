## Overview

This GitHub repository contains Python scripts focused on advanced image processing techniques, particularly for image restoration. The repository demonstrates the implementation of three distinct image restoration algorithms: Richardson-Lucy Iteration, Wiener Filter, and Quadratic Penalized Weighted Least Squares (QPWLS) Estimator. These scripts provide a practical approach to mitigating issues like blur and noise in digital images, showcasing the power of computational image processing.

## Contents

1. `richardson_lucy.py` - Implements the Richardson-Lucy iterative method for image deconvolution, particularly useful for dealing with blur caused by a known Gaussian kernel.

2. `wiener_filter.py` - Features the application of the Wiener Filter, an optimal restoration filter for images corrupted by linear motion or out-of-focus blur in conjunction with additive Gaussian noise.

3. `qpwls_estimator.py` - Demonstrates the Quadratic Penalized Weighted Least Squares method for image restoration, a more advanced technique that takes into account the regularization aspect of the restoration process.

Each script includes a complete pipeline from image preparation, application of the algorithm, to the visualization of the results.

## Requirements

To run these scripts, you need to install the following Python packages:
- numpy~=1.26.2
- scipy~=1.11.4
- matplotlib~=3.8.2
- scikit-image~=0.22.0

You can install all required packages using the provided `requirements.txt` file with the following command:

```
pip install -r requirements.txt
```

## Usage

To use these scripts, clone the repository and navigate to the directory containing the scripts. You can run each script individually using Python. For example:

```
python richardson_lucy.py
```

Each script will display a series of images: the original image, the image with applied blur and noise, and the restored image using the respective algorithm.

## Contributing

Contributions to this project are welcome. Please feel free to fork the repository, make changes, and submit pull requests. If you encounter any issues or have suggestions for improvement, open an issue in the repository.

## License

This project is licensed under the MIT License - see the LICENSE file in the repository for details.

## Contact

If you have any questions or want to reach out for collaboration, please contact ge75sow@mytum.de.

