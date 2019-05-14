""" Relevant imports and global variables """
import torch
torch.manual_seed(1234)
from generic_helpers import *
import sys

mini_batch_size = 250
nb_runs = 10
lr = 0.01

######### _1CHANNEL2IMAGES #############################################################
sys.path.insert(0, "channelimagesModels")
from BaseNet import *
from ConvNet1 import *
from _1channel2images import *
print("Working with 1channel2images framework, nb_classes = ", nb_classes)

model1C_list = [BaseNet1C(), ConvNet1_1C()]
nb_epochs = 30
for model_1C in model1C_list:
    test_results_1C = multiple_training_runs_1C(model_1C, nb_runs, lr, mini_batch_size, nb_epochs, verbose=False)
    write_to_csv_1C('1channel2images.csv', model_1C, test_results_1C, lr, nb_epochs)
    print("For model {}, mean test accuracy = {}".format(model_1C.name, test_results_1C[6]))
    
    
######### _2CHANNELS1IMAGE #############################################################
from _2channels1image import *
print("Working with 2channels1image framework, nb_classes = ", nb_classes)
model2C_list = [BaseNet2C(), ConvNet1_2C(), ConvNet2_2C(), ConvNet4_2C()]
nb_epochs = 75
for model_2C in model2C_list:
    test_results_2C = multiple_training_runs_fn(model_2C, train_model_2C, test_model_2C, title_2C, nb_runs, lr, mini_batch_size, nb_epochs, verbose=False)
    write_to_csv('2channels1image.csv', model_2C, test_results_2C, lr, nb_epochs)
    print("For model {}, mean test accuracy = {}".format(model_2C.name, test_results_2C[4]))
    
    
######### WEIGHT_SHARING ###############################################################
sys.path.insert(0, "weightssharingModels")
from NetSharing import *
from weight_sharing import *
print("Working with weight_sharing framework")
modelws_list = [NetSharing1(), NetSharing2(), NetSharing3(), NetSharing4()]
nb_epochs = 75
for model_ws in modelws_list:
    test_results_ws = multiple_training_runs_fn(model_ws, train_model_ws, test_model_ws, title_ws, nb_runs, lr, mini_batch_size, nb_epochs, verbose=False)
    write_to_csv('weightsharing.csv', model_ws, test_results_ws, lr, nb_epochs)
    print("For model {}, mean test accuracy = {}".format(model_ws.name, test_results_ws[4]))

    
######### AUXILIARY_LOSSES #############################################################
sys.path.insert(0, "auxiliarylossesModels")  
from Incept import *
from auxiliary_losses import *
print("Working with auxiliary_losses framework")
modelaux_list = [Incept1a(), Incept1b(), Incept2(), Incept3(), Incept4()]
nb_epochs = 60
for model_aux in modelaux_list:
    test_results_aux = multiple_training_runs_fn(model_aux, train_model_aux, test_model_aux, title_aux, nb_runs, lr, mini_batch_size, nb_epochs, verbose=False)
    write_to_csv('auxiliary_losses.csv', model_aux, test_results_aux, lr, nb_epochs)
    print("For model {}, mean test accuracy = {}".format(model_aux.name, test_results_aux[4]))
    
    
    
    
    
    
    
    
    
    