# DSCAE
DSCAE: A Denoising Sparse Convolutional Autoencoder Defense Against Adversarial Examples

# abstract

Deep neural networks are a state-of-the-art method used to computer vision. Imperceptible perturbations added to benign images can induce the deep learning network to make incorrect predictions, though the perturbation is imperceptible to human eyes. Those adversarial examples threaten the safety of deep learning model in many real-world applications. In this work, we proposed a method called denoising sparse convolutional autoencoder (DSCAE) to defense against the adversarial perturbations. This is a preprocessing module works before the classification model, which can remove substantial amounts of the adversarial noise. The DSCAE defense has been evaluated against FGSM, DeepFool, C$\&$W, JSMA attacks on the MNIST and CIFAR-10 datasets. The experimental results show that DSCAE defends against state-of-art attacks effectively.

The adversarial examples are built on Craft Image Adversarial Samples(https://github.com/gongzhitaao/tensorflow-adversarial/tree/v0.2.0) with TensorFlow. 

# citation
Please, cite this paper if you use the code in this repository as part of a published research project:

@article{article,  
author = {Ye, Hongwei and Liu, Xiaozhang and Li, Chunlai},  
year = {2020},  
month = {11},  
pages = {1-11},  
title = {DSCAE: a denoising sparse convolutional autoencoder defense against adversarial examples},  
journal = {Journal of Ambient Intelligence and Humanized Computing},  
doi = {10.1007/s12652-020-02642-3}  
}
