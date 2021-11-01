## BernNet: Learning Arbitrary Graph Spectral Filters via Bernstein Approximation

This repository contains a PyTorch implementation of "BernNet: Learning Arbitrary Graph Spectral Filters via Bernstein Approximation".(https://arxiv.org/abs/2106.10994)


## Environment Settings    
- pytorch 1.8.1
- numpy 1.18.1
- torch-geometric 1.6.3 
- tqdm 4.59.0
- scipy 1.6.2
- seaborn 0.11.1
- scikit-learn 0.24.1


## Learning filters from the signal(./LearningFilters)
We conduct an empirical analysis on 50 real images with the resolution of 100×100 from the Image Processing Toolbox in Matlab.

### Datasets
We provide the processed dataset and you can run the code directly. We also provide the original images in the folder './LearningFilters/image‘ and the Matlab code './LearningFilters/preprocessing.m' for preprocessing.

### Running the code
Input Parameters
+ filter_type: the type of the filter applied to the spectral domain
+ net: the GNN models, default='BernNet'

You can run the following script in the folder './LearningFilters' directly
```sh
sh bernnet.sh
```
or run the following Command 
+ The band-pass filter for BernNet
```sh
python training.py --filter_type band --net BernNet
```
+ The high-pass filter for BernNet
```sh
python training.py --filter_type high --net BernNet
```
## Node classification on real-world datasets (./NodeClassification)
We evaluate the performance of BernNet against the competitors on 10 real-world datasets.

### Datasets
We provide the datasets in the folder './NodeClassification/data' and you can run the code directly, or you can choose not to download the datasets('./NodeClassification/data') here. The code will automatically build the datasets through the data loader of Pytorch Geometric.

### Running the code

You can run the following script in the folder './NodeClassification' directly and this script describes the hyperparameters settings of BernNet on each dataset.
```sh
sh bernnet.sh
```
or run the following Command 
+ Pubmed
```sh
python training.py  --dataset Pubmed --Bern_lr 0.01 --dprate 0.0 --weight_decay 0.0  --net BernNet
```
+ Texas
```sh
python training.py --dataset Texas --lr 0.05 --Bern_lr 0.002 --dprate 0.5 --net BernNet
```

## Citation
```sh
@inproceedings{he2021bernnet,
  title={BernNet: Learning Arbitrary Graph Spectral Filters via Bernstein Approximation},
  author={He, Mingguo and Wei, Zhewei and Huang, Zengfeng and Xu, Hongteng},
  booktitle={NeurIPS},
  year={2021}
}
```

## Contact

If you have any questions, please feel free to contact me with mingguo@ruc.edu.cn


