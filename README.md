# Federated Learning framework based on FedAvg

PyTorch-based Federated Learning framework based on FederatedAveraging (FedAvg) algorithm.
This is an unofficial translation of the framework proposed by Caldas et a. in **LEAF** (written in TensorFlow). References follow. 

## Resources

  * **Homepage:** [leaf.cmu.edu](https://leaf.cmu.edu)
  * **Paper:** ["LEAF: A Benchmark for Federated Settings"](https://arxiv.org/abs/1812.01097)
  * **Original repository:** [LEAF: A Benchmark for Federated Settings GitHub Repository](https://github.com/TalwalkarLab/leaf)

## Datasets

1. FEMNIST

  * **Overview:** Image Dataset
  * **Details:** 62 different classes (10 digits, 26 lowercase, 26 uppercase), images are 28 by 28 pixels (with option to make them all 128 by 128 pixels), 3500 users
  * **Task:** Image Classification

[comment]: <> (2. Sentiment140)

[comment]: <> (  * **Overview:** Text Dataset of Tweets)

[comment]: <> (  * **Details** 660120 users)

[comment]: <> (  * **Task:** Sentiment Analysis)

[comment]: <> (3. Shakespeare)

[comment]: <> (  * **Overview:** Text Dataset of Shakespeare Dialogues)

[comment]: <> (  * **Details:** 1129 users &#40;reduced to 660 with our choice of sequence length. See [bug]&#40;https://github.com/TalwalkarLab/leaf/issues/19&#41;.&#41;)

[comment]: <> (  * **Task:** Next-Character Prediction)

2. Celeba

  * **Overview:** Image Dataset based on the [Large-scale CelebFaces Attributes Dataset](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)
  * **Details:** 9343 users (we exclude celebrities with less than 5 images)
  * **Task:** Image Classification (Smiling vs. Not smiling)

3. CIFAR-100
  * **Overview**: Image Dataset based on [CIFAR-100](https://www.cs.toronto.edu/~kriz/cifar.html) and [Federated Vision Datasets](https://github.com/google-research/google-research/tree/master/federated_vision_datasets)
  * **Details**: 100 users with 500 images each. Different combinations are possible, following Dirichlet's distribution
  * **Task**: Image Classification over 100 classes

4. CIFAR-10
  * **Overview**: Image Dataset based on [CIFAR10](https://www.cs.toronto.edu/~kriz/cifar.html) and [Federated Vision Datasets](https://github.com/google-research/google-research/tree/master/federated_vision_datasets)
  * **Details**: 100 users with 500 images each. Different combinations are possible, following Dirichlet's distribution
  * **Task**: Image Classification over 10 classes

5. iNaturalist
  * **Overview**: Image Dataset based on [iNaturalist-2017](https://github.com/visipedia/inat_comp/tree/master/2017) dataset and iNaturalist-User-120k from [Federated Vision Datasets](https://github.com/google-research/google-research/tree/master/federated_vision_datasets)
  * **Details**: 9,275 users. Non-i.i.d. and unbalanced setting.
  * **Task**: Image Classification over 1,203 classes

[comment]: <> (5. Synthetic Dataset)

[comment]: <> (  * **Overview:** We propose a process to generate synthetic, challenging federated datasets. The high-level goal is to create devices whose true models are device-dependant. To see a description of the whole generative process, please refer to the paper)

[comment]: <> (  * **Details:** The user can customize the number of devices, the number of classes and the number of dimensions, among others)

[comment]: <> (  * **Task:** Classification)

[comment]: <> (6. Reddit)

[comment]: <> (  * **Overview:** We preprocess the Reddit data released by [pushshift.io]&#40;https://files.pushshift.io/reddit/&#41; corresponding to December 2017.)

[comment]: <> (  * **Details:** 1,660,820 users with a total of 56,587,343 comments. )

[comment]: <> (  * **Task:** Next-word Prediction.)

## Notes

- Install the libraries listed in ```requirements.txt```
    - I.e. with pip: run ```pip3 install -r requirements.txt```
- Go to directory of respective dataset for instructions on generating data
- ```models``` directory contains instructions on running baseline reference implementations