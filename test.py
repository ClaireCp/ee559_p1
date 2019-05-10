import torch
from torch import nn, optim
import sys
sys.path.insert(0, "../")
from generic_helpers import *
from torchvision import datasets
import dlc_practical_prologue as prologue

def load_random_datasets():
    """ Function to generate new random training and test sets for various runs """
    train_input, train_target, train_classes, test_input, test_target, test_classes = \
    prologue.generate_pair_sets(nb=1000)
    
    """ Feature Normalization: We normalize the train and test datasets with the mean and variance of the training set 
    (we don't want to introduce information of the test set into the training by normalizing over the whole dataset)"""
    train_input, mean_input, std_input = standardize(train_input)
    test_input = (test_input - mean_input) / std_input
    
    return train_input, train_target, train_classes, test_input, test_target, test_classes


######################################################################
""" Generic function for multiple training runs for _2channels1image, weight_sharing and  auxiliary_losses.
Arguments:
    train_fn: reference to the train_model() function of the current framework 
    test_fn: reference to the test_model() function of the current framework 
    title: title for the figure plots """
def multiple_training_runs_fn(model_ref, train_fn, test_fn, title,  nb_runs, lr, mini_batch_size=1000, nb_epochs=300, verbose=True, plot=True):
    list_time = []
    list_best_acc = [] # (best) accuracy on training set
    list_acc_test = [] # accuracy on test set
    list_val_acc_history = []
    list_test_acc_history = []
            
    for i in range(nb_runs):
        # We load new random training and test sets
        train_input, train_target, train_classes, test_input, test_target, test_classes = load_random_datasets()
        
        # and create a new instance of our model to have new random initial weights
        model = model_ref.return_new()       
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.95)
        
        model, time_elapsed, best_acc, val_acc_history, test_acc_history = train_fn(model, train_input, train_target, train_classes, test_input, test_target, test_classes, optimizer, mini_batch_size, nb_epochs, verbose)
        list_time.append(time_elapsed)
        list_best_acc.append(best_acc)
        list_val_acc_history.append(val_acc_history)
        list_test_acc_history.append(test_acc_history)
  
        acc_test = test_fn(model, test_input, test_target)
        list_acc_test.append(acc_test)
        
    if plot == True:
        title = title.format(model.name)
        
        mean_val_acc_history, std_val_acc_history = mean_per_epoch_list(list_val_acc_history)
        mean_test_acc_history, std_test_acc_history = mean_per_epoch_list(list_test_acc_history)
        plot_history(mean_val_acc_history, std_val_acc_history, mean_test_acc_history, std_test_acc_history, title)
        
    mean_time, std_time = compute_properties(list_time)
    mean_best_acc, std_best_acc = compute_properties(list_best_acc)
    mean_acc_test, std_acc_test = compute_properties(list_acc_test)
    
    return mean_time, std_time, mean_best_acc, std_best_acc, mean_acc_test, std_acc_test
