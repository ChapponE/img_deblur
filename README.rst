Deblur and Denoise Images using Unfolded Forward Backward
============
Introduction
------------
This project aims to deblur and denoise images, with a focus on the MNIST dataset. The methodology is based on the "Unfolded Forward Backward" approach, developed as part of the master's thesis project at ENS Lyon.

Project Background
------------
For detailed insights and background information, please refer to the publication ["Studying circumstellar environments with deep learning for high-contrast imagery reconstruction"](https://chappone.github.io/ws/publications/).

Installation
------------
detailed in requierements.txt file

Generating Data
------------
To generate datasets and apply blur (with customizable parameters including size of blur kernel, and the sigma's of blur and noise):
--> python data/generate/generate_dataset.py

Train models: 
------------
To train models (with customizable parameters including the scales variable which is the size of the UNets used in the model, and depth which is the number of iterations of the iterative method Forward Backward)
--> python src/train/train_unfolded_fb.py

- Test trained models:
------------
--> python src/utils/test.py