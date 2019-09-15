# pytorch-DCRNN

The original tensorflow implementation: [liyaguang/DCRNN](https://github.com/liyaguang/DCRNN),

This repo is still under development.

PyTorch implementation of Diffusion Convolutional Recurrent Neural Network in the following paper: \
Yaguang Li, Rose Yu, Cyrus Shahabi, Yan Liu, [Diffusion Convolutional Recurrent Neural Network: Data-Driven Traffic Forecasting](https://arxiv.org/abs/1707.01926), ICLR 2018.


## Requirements
- scipy=1.2.1
- numpy=1.16.2
- pandas=0.24.2
- torch>=1.1.0
- tqdm
- pytable


## Data Preparation
For data preparation, check the original repo:[liyaguang/DCRNN](https://github.com/liyaguang/DCRNN)


## Model Training
For now, training is only supported for METR-LA dataset due to data availability.
```bash
# METR-LA
python train.py --config config.json
```
Each epoch takes about 5-6min(~ 340 seconds) on a single RTX 2080 Ti for METR-LA. 

There is a chance that the training loss will explode, the temporary workaround is to restart from the last saved model before the explosion, or to decrease the learning rate earlier in the learning rate schedule. 


## Log and Model Savings
Log information will be saved at `saved/log/.../info.log` 
The best validated model will be saved at `saved/model/.../model_best.pth`

The best results that I obtained so far is shown in `test_results.log`
