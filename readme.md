# Incorporating Label Uncertainty in Understanding Adversarial Robustness
*A repository for reproducing the methods and experiments, presented in [this paper](https://arxiv.org/abs/2107.03250), for understanding adversarial robustness based on a notion of label uncertainty. Created by [Xiao Zhang](xiao-zhang.net).*
<br /><br />

# Installation
The code was developed using Python3 on [Anaconda](https://www.anaconda.com/download/#linux)
* Install Pytorch 1.6.0: 
    ```
    conda update -n base conda && conda install pytorch=1.6.0 torchvision -c pytorch -y
    ```

* Install other dependencies:
    ```
    pip install waitGPU && conda install -c conda-forge cleanlab imageio 
    ```
<br />

# What is in this respository & How to use
* ```00_data``` folder containes the [CIFAR-10H dataset](https://github.com/jcpeterson/cifar-10h)

* ```01_label_uncertainty``` folder containes the codes for visualizing label uncertainty (Figures 2 and Figure 6) and error region label uncertainty of classification models(Figure 3)
    1. visualize label uncertainty on CIFAR-10
        ```
        python visualize.py
        ```
    2. pretrain CIFAR-10 classifiers
        ```
        python train_cifar10.py --attack none && python train_cifar10.py --attack pgd
        ```
    3. compute error region statistics
        ```
        python err_stats.py
        ```

* ```02_concentration_estimation``` folder containes the codes for obtaining our intrinsic robustness estimates (Figure 4 and Table 1) 
    ```
    python concentration_lu_ball.py
    ```

* ```03_abstaining_classifier``` folder containes the codes for the experiments on abstaining classifier (Figures 5)
    ```
    python eval.py && python plot.py
    ```


* ```04_confident_learning``` folder containes the codes (adapted from [cleanlab](https://github.com/cleanlab/cleanlab)) for the experiments on estimating label error sets using confident learning (Figures 7 and Figure 8)
    1. prepare training and testing datasets
        ```
        cd data && bash prepare_dataset.bash
        ```
    2. pretrain CIFAR-10 classifier
        ```
        python cifar10_train_crossval.py
        ```
    3. Estimate label error sets using confident learning and visualize the difference with CIFAR-10H
        ```
        python estimate_label_errors_test.py
        ```
