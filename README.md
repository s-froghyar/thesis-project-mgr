# Robust Deep Learning for Music Genre Recognition

This repo contains the implementation to replicate the experiments in the MSci project: Robust Deep Learning for Music Genre Regignition using Augmentation techniques by Soma Froghyar

For the experiments the [GTZAN](https://www.kaggle.com/andradaolteanu/gtzan-dataset-music-genre-classification) dataset was used and the baseline architecture is from Anguiar et al.

## Introduction

In this paper 2 techniques are compared to data augmentation in order to make a model invariant to adversarial perturbations. The 2 techniques are: Tangent Propagation and Augerino.

Tangent Propagation is a regularisation technique introduced by [Simard et al.](https://ieeexplore.ieee.org/document/1227801), which encourages local invariance to transformations to the input vector. It functions by penalizing the model for changes in the model's prediction when the signal is transformed. 

Augerino is a method by Benton et al. to learn invariances from training data alone. It learns the distribution of the augmentations by applying random distributions to the inputs and averaging the outputs. To learn more about Augerino: https://github.com/g-benton/learning-invariances

## Transformations

Two audio transformations are applied seperately in these experiments, namely: Gaussian Noise Injection and Pitch Shifting.

