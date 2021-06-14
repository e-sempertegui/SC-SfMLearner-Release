# Uncertainty Prediction for Monocular Depth Estimation

This code is based on and combines implementation details coming from the following work:

* Unsupervised Scale-consistent Depth Learning from Video. Open source implementation [here](https://github.com/JiawangBian/SC-SfMLearner-Release).

* On the uncertainty of self-supervised monocular depth estimation. Open source implementation [here](https://github.com/mattpoggi/mono-uncertainty). Note, however, that training scripts are not available.

## Preamble
This code was developed and tested with python 3.8.5, Pytorch 1.6, and CUDA 10.1.


## Training

Training with NYU dataset can be reproduced by running

```bash
bash scripts/train_nyu.sh
```
where the training parameters are given as arguments. For more info on the available argumenets please refer to the file `train.py`. 
Note that the dataset path is defined in `scripts/train_nyu.sh`.
 
