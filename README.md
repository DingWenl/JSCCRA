# JSCCRA
JSCCRA data augmentation

## The related version information
1. Python == 3.9.13
2. Keras-gpu == 2.6.0
3. tensorflow-gpu == 2.6.0
4. scipy == 1.9.3
5. numpy == 1.19.3

## Training CNN-Former model with JSCCRA for the benchmark dataset
1. Download the code.
2. Download the [benchmark dataset](http://bci.med.tsinghua.edu.cn/download.html) and its [paper](https://ieeexplore.ieee.org/abstract/document/7740878).
3. Create a model folder to save the model.
4. Change the data and model folder paths in train and test files to your data and model folder paths.

## Subject-independent classification 
1. Run the `train_independent.py` file to get the trained model;
2. Run the `test_independent.py` file to conduct subject-independent test.
