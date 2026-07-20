# TAEPPO
Implementation of the paper "Neural Resource Encoding for Distributed Flexible Job-Shop Scheduling with Transportation-Time Constraints"

## Get Started

### Installation

* python $\ge$ 3.6.13
* pytorch $\ge$ 1.8.1
* gym $\ge$ 0.18.0
* numpy $\ge$ 1.19.5
* pandas $\ge$ 1.1.5
* visdom $\ge$ 0.1.8.9

Note that pynvml is used in ```test.py``` to avoid excessive memory usage of GPU, please modify the code when using CPU.

### Introduction

* ```data_dev``` and ```data_test``` are the validation sets and test sets, respectively.
* ```env``` contains code for the FJSP environment
* ```save``` is the folder where the experimental results are saved
* ```utils``` contains some helper functions
* ```config.json``` is the configuration file
* ```DRL.py``` is the implementation of NRE-DRL
* ```embedding.py``` is Topology-Aware Relational Attention Networks
* ```mlp.py``` is the MLP code (referenced from L2D)
* ```test.py``` for testing
* ```train.py``` for training
* ```validate.py``` is used for validation

## Reproduce result in paper

There are various experiments in this article, which are difficult to be covered in a single run. Therefore, please change ```config.json``` before running.

Note that disabling the ```validate_gantt()``` function in ```schedule()``` can improve the efficiency of the program, which is used to check whether the solution is feasible.

### train

```
python train.py
```

Note that there should be a validation set of the corresponding size in ```./data_dev```.

### test

```
python test.py
```
Note that there should be model files (```*.pt```) in ```./model```.

## Reference

* https://github.com/zcaicaros/L2D
* https://github.com/yd-kwon/MatNet
* https://github.com/songwenas12/fjsp-drl
* https://github.com/dmlc/dgl/tree/master/examples/pytorch/han

