# bo_framework


This repository contains the code I used for tuning hyper-parameters in two models.
Epilepsy Prediction and Natural Language Processing


## Usage:

### Default Settings:

In language_model, the default command will run the Penn TreeBank model to tune learning rate (0.0001, 1) with 2 hidden layers, 50 hidden units, 20 epochs of training, 20 evaluaations of functions (bo epochs). More details of default settings is in language_model/utils/pennflags.py

python bo_penn_tree_bank.py

In epilepsy_prediction model, the following command will run the lstm model to tune learning rate (0.0001, 0.1) with 2 hidden layers, 200 hidden units, 20 epochs of training, 20 evaluations of functions (bo epochs). More details of default settings is in epilepsy_prediciton/utils/lstmflags.py. 

python gpyoptflaglstm.py 

### Running Single Experiments:
* python bo_penn_tree_bank.py --run_single
* python gpyoptflaglstm.py --run_single


## Required
* numpy==1.14.5
* matplotlib==2.1.2
* pandas==0.22.0
* tensorflow==1.8.0
* Keras==2.1.5
* GPy==1.9.5
* GPyOpt==1.2.5
* scikit_learn==0.19.2
* scipy==1.0.0

Run the following command to install these packages:
* pip install -r requirements.txt


## References:
Bayesian Optimisation Library:
https://github.com/SheffieldML/GPyOpt

Epilepsy Predicition Model: 
https://github.com/rodrigomfw/framework.

Penn TreeBank model: 
https://github.com/tensorflow/models/blob/master/tutorials/rnn/ptb/ptb_word_lm.py.




