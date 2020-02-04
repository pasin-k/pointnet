# Pointnet-keras
Use transfer learning based on Charlesq34's tensorflow implementation: https://github.com/charlesq34/pointnet
Please refer to [1d-tooth](https://github.com/jobpasin/tooth-1d) for details of the application.

Package requirement: Python3.6, keras, tensorflow, numpy, matplotlib, h5py

> # Classification:
> Download the aligned dataset from https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip
> Put all traning h5 files under Prepdata folder, all testing h5 files under Prepdata_test folder, then run train_cls.py. Accuracy rate will be 82.5%, which is slightly lower than the original implementation. 


# Transfer Learning
Train model with ModelNet40 using `train_cls.py` 
Then, run `transfer_train.py` on our dataset. We freeze all but last two layers during our training.


