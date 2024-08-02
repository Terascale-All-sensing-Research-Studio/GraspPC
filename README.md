# [RO-MAN2024] GraspPC: Generating Diverse Hand Grasp Point Clouds on Objects

## Installation:
Code is tested using python 3.8.10, CUDA 12.0 and PyTorch version 1.9.1+cu111
  
We recommend using a virtual environment.
Here is a snippit of code to create a new virtual environment, activate and install requirements:

```bash
python3.8 -m venv myenv && \
source myenv/bin/activate && \
pip install -r requirements.txt
```

To install Pytorch extensions:
```bash
bash install.sh
```
If there is an error installing, you have to go into each folder in the extension folder and separately install each:
```bash
cd extensions/{folder}
python setup.py install
```

## Subfolder Contents:
1. GraspPC/data/GraspPC contains the different datasets we used, each folder holds the different test and train files used.
2. GraspPC/datasets contains the dataloaders used for each dataset, the $ROOTDIR is the location of where the data is stored.
3. GraspPC/experiments/GraspPC contains pretrained models for the different datasets.
4. GraspPC/models contains the different models we used to train/test with. There is a Baseline model, GraspPC model and the GraspPC_noint model.
	- The Baseline model is PoinTr, and GraspPC and GraspPC_noint model are derived from PoinTr: https://github.com/yuxumin/PoinTr
	- For each of the different models you want to use, you must first change the model in the __init__.py 
	- To use the Baseline code you must change the model to: import models.Baseline
	- To use GraspPc you must change the model to: import models.GraspPC
	- To use GraspPC_noint you must change the model to: import models.GraspPC_noint
5. GraspPC/tools contains the different runners used for running the different datasets
	- In order to run the different models you must change the __init__.py to have the runner you want to use.
	- For example this is what goes in the __init__.py:	
		- {model} import run_net
		- {model} import test_net

## Running Test Samples
Test with GraspPC model: 
```bash
bash ./scripts/test.sh 0 --ckpts ./experiments/GraspPC/{dataset_name}/{exp_name}/ckpt-last.pth(or ckpt-epoch#.pth) --config ./cfgs/{dataset}/{dataset_name}.yaml --exp_name exp_name
```
For instance to run HOH_3dm test: 
```bash
bash ./scripts/test.sh 0 --ckpts ./experiments/GraspPC/HOH_3dm/GraspPC_3dm/ckpt-epoch-300.pth --config ./cfgs/HOH_3dm/HOH_3dm.yaml  --exp_name exp_name
```
To train with GraspPC model: 
```bash
bash ./scripts/train.sh 0,1 --config ./cfgs/{dataset}/{dataset}.yaml  --exp_name exp_name
```

