# Weight-sharing-and-auxiliary-losses-for-classification
The objective of this project is to test different architectures to compare two digits visible in a two-channel image. It aims at showing in particular the impact of weight sharing, and of the use of an auxiliary loss to help the training of the main objective.

## Models description
Models' description and comparison is available in the [report](CLASSIFICATION__WEIGHT_SHARING__AUXILIARY_LOSSES.pdf).

## Requirements:
* Torch

## Run
To train Siamese model with auxiliary loss version 5 and test accuracy for class and digit prediction use command:
```bash
python main.py
```
Use argument
```
--model_version N
```
with N from 1 to 6 to define version of Siamese model with auxiliary loss.

Use argument
```
--n_samples N
```
to define number of samples to use as N.
    
Use argument
```
--n_epoch N
```
to define number of epochs for training as N.

Use argument
```
--n_rounds N
```
to define number of times to perform training as N. Used to measure accuracy characteristics among several runs (default 1).
    
Use argument
```
--grid_search
```
to perform grid search for learning rate and regularization term parameters.

Use argument
```
--lr N
```
to define learning rate for training as N (default 0.01).

Use argument
```
--reg N
```
to define regularization term for training as N (default 0.01).

Use argument
```
--plot_curves
```
to plot accuracy and loss curves after training.
