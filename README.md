**pyDeepDelta

This package 'pyDeepDelta' contains the following components:

pydeepdelta.py: A Python/TensorFlow module for predictive uncertainty quantification in deep learning models, as described in the paper "Epistemic Uncertainty Quantification in Deep Learning Classification by the Delta Method" found at: https://arxiv.org/abs/1912.00832

pydeepdelta_demo.ipynb: A demo Jupyter notebook showing how to apply the pyDeepDelta module on a LetNet-based convolutional neural network MNIST classifier.

pydeepdelta_sampler_demo.ipynb: A demo jupyter notebook showing how to combine the standard Laplace approximation with the key ideas from the paper. Hence, resulting in an efficient Laplace Approximation based Monte Carlo sampling algorithm.

utils.py: Helper functions

data/Lambda_G_rseed_0_layers_2wb-3wb-4wb-5wb-6wb_K_600_num_steps_90000_reg_lambda_0.01.npy: Pre-calculated OPG eigenvalues for K=600. 

data/Q_G_rseed_0_layers_2wb-3wb-4wb-5wb-6wb_K_600_num_steps_90000_reg_lambda_0.01.npy: Pre-calculated OPG eigenvectors for K=600. Note: As this file is too large for github (220MB), it can be downloaded here: https://drive.google.com/file/d/1-fKgvDei3Ba7rhO2YlCKcSdQWl6EfFcX/view?usp=sharing

data/Lambda_H_rseed_0_layers_2wb-3wb-4wb-5wb-6wb_K_600_num_steps_90000_reg_lambda_0.01.npy: Pre-calculated Hessian eigenvalues for K=600. 

data/Q_H_rseed_0_layers_2wb-3wb-4wb-5wb-6wb_K_600_num_steps_90000_reg_lambda_0.01.npy: Pre-calculated Hessian eigenvectors for K=600. Note: As this file is too large for github (220MB), it can be downloaded here: https://drive.google.com/file/d/1YMrob46M38bHV_Lq_pfFYBhBYdcsT8IE/view?usp=sharing

data/model_rseed_0_num_steps_90000_reg_lambda_0.01*: Pre-trained TensorFlow model for the demo in pydeepdelta_demo.ipynb.

pyDeepDelta is written in Python3 and depends on the following libraries

TensorFlow 2.1.0 (https://www.tensorflow.org)
NumPy (http://www.numpy.org)
SciPy (http://www.scipy.org)
tqdm (https://tqdm.github.io/docs/tqdm/)

**What is pyDeepDelta?

pyDeepDelta is a module for predictive uncertainty quantification in deep learning models. The underlying technology is described in the paper "Epistemic Uncertainty Quantification in Deep Learning Classification by the Delta Method" found at: https://arxiv.org/abs/1912.00832

**Where to get?

By GIT (development): https://github.com/gknilsen/pydeepdelta

**Author

Author: Geir K. Nilsen geir.kjetil.nilsen@gmail.com -- My blog: http://octovoid.com

Other people have contributed in various ways, see the THANKS file.

**License

Copyright (c) 2018-2021 by Geir K. Nilsen (geir.kjetil.nilsen@gmail.com) and the University of Bergen.

pyDeepDelta is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with pydeepdelta. If not, see http://www.gnu.org/licenses/.
