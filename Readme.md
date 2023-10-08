# Readme

The code structure of the Capsule Network was inspired by **Gram.AI**'s code, and we would like to express our gratitude to them for their contribution.

This is their code repository: https://github.com/gram-ai/capsule-networks.git

##  Requirement

* Python 3

- PyTorch
- TorchVision
- TorchNet
- TQDM
- Visdom
- Seaborn
- Pandas
- Matplotlib
- Numpy
- Xlrd
- Xlwt
- Scikit-learn

## Dataset

The KTH dataset used in this project is sourced from https://www.csc.kth.se/cvap/actions/.

The human joint spatial coordinates were extracted using AlphaPose, and the code repository for AlphaPose can be found at https://github.com/MVIG-SJTU/AlphaPose.git.

The preprocessed dataset used by this code is located in the `dataset` folder.

## Usage

The `MCCN.py` file  contains the implementation of the MCCN network.

```
sudo python3 -m visdom.server & python3 MCCN.py
```

The `ResNet18.py` file  contains the implementation of the ResNet18 network.

```
sudo python3 ResNet18.py
```

The `model_load.py` file can be used to import a pre-trained model file and generate COE data.

```
sudo python3 model_load.py
```

The `coe_paint.py` file can visualize COE data.

```
sudo python3 coe_paint.py
```

The `CapsNet.py` file contains the implementation of the Capsule Network and can be used to implement mCapsNet, X-CapsNet and Y-CapsNet.

```
sudo python3 CapsNet.py
```