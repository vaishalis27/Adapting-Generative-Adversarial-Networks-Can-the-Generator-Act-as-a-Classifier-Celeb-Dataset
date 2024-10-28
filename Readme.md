# Adapting GANs: Can the Generator Act as a Classifier?

## Project Overview

This project investigates the potential of Generative Adversarial Networks (GANs), specifically using the StarGAN model, in performing classification tasks. The primary objective is to determine whether the generator within a GAN can be adapted to act as a classifier. The project builds on the official StarGAN codebase and applies it to datasets such as MNIST and CelebA to test this hypothesis.

## Requirements

Before running the project, ensure that the following dependencies are installed:

- Python 3.x
- PyTorch
- NumPy
- OpenCV
- Matplotlib
- tqdm

You can install the required packages with the following command:

```bash
pip install torch numpy opencv-python matplotlib tqdm
```

## Running the Project

The entire project is contained within the `master.py` file. This script includes all necessary steps, from dataset preparation to model training and evaluation.

### To run the project:

1. Ensure all dependencies are installed.
2. Simply run `master.py` file :

This will execute all necessary operations, including downloading datasets, training the model, and evaluating the results.

## Acknowledgments

I would like to express my gratitude to the developers of the official StarGAN code, whose work provided the foundation for this project. Special thanks to the team at [Clova AI Research, NAVER](https://clova.ai/en/research/research-area-detail.html?id=0), particularly Donghyun Kwak, for their insightful discussions and contributions to the open-source community.

