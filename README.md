## BernNet: Learning Arbitrary Graph Spectral Filters via Bernstein Approximation

This repository contains a PyTorch implementation of "BernNet: Learning Arbitrary Graph Spectral Filters via Bernstein Approximation".(https://arxiv.org/abs/2106.10994)


## Requirements     
- pytorch 1.8.1
- numpy 1.18.1
- torch-geometric 1.6.3 
- tqdm 4.59.0
- scipy 1.6.2
- seaborn 0.11.1
- scikit-learn 0.24.1

## Datasets

For learning filters from the signal(Go to './LearningFilters').

We provide the processed dataset and you can run the code directly. We also provide the original images in the folder './image and the Matlab code 'preprocessing.m' for preprocessing.

For Node classification on real-world datasets(Go to'./NodeClassification').

We provide the datasets and you can run the code directly.


## Running the code

For learning filters from the signal(Go to './LearningFilters').

```sh
sh bernnet.sh
```

For Node classification on real-world datasets(Go to'./NodeClassification').

```sh
sh bernnet.sh
```
