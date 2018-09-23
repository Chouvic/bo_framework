# bo_framework


This is a framework to test BO performance with two LSTM models. 
The BO library is adopt from GPyOpt.
The lstm framework is adopt from https://github.com/rodrigomfw/framework

Install Requirements Packages:
pip install -r requirements.txt



Usage:

In language_model:

The default command will run the Penn TreeBank model to tune learning rate (0.0001, 1) with 2 hidden layers, 50 hidden units, 20 epochs of training, 20 evaluaations of functions (bo epochs). More details of default settings is in language_model/utils/pennflags.py

python bo_penn_tree_bank.py


For epilepsy_prediction model:

The following command will run the lstm model to tune learning rate (0.0001, 0.1) with 2 hidden layers, 200 hidden units, 20 epochs of training, 20 evaluations of functions (bo epochs). More details of default settings is in epilepsy_prediciton/utils/lstmflags.py. 

python gpyoptflaglstm.py 




