import torch
from torch import nn, optim
from torch.nn import functional as F

import sys
sys.path.insert(0, "../")
from generic_helpers import *

import time
import copy

""" Generic function for multiple training runs for _2channels1image, weight_sharing and  auxiliary_losses.
Arguments:
    train_fn: reference to the train_model() function of the current framework 
    test_fn: reference to the test_model() function of the current framework 
    title: title for the figure plots """
def multiple_training_runs_aux(model_ref, title,  nb_runs, lr, alpha, mini_batch_size=1000, nb_epochs=300, verbose=True, plot=True):
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
        
        model, time_elapsed, best_acc, val_acc_history, test_acc_history = train_model_aux(model, train_input, train_target, train_classes, test_input, test_target, test_classes, optimizer, alpha, mini_batch_size, nb_epochs, verbose)
        list_time.append(time_elapsed)
        list_best_acc.append(best_acc)
        list_val_acc_history.append(val_acc_history)
        list_test_acc_history.append(test_acc_history)
  
        acc_test = test_model_aux(model, test_input, test_target, test_classes, 10)
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


""" FRAMEWORK FOR USING AUXILIARY LOSSES """
""" Although the network is still a binary classifier, the forward pass also outputs two auxiliary variables of size=10 that are used for the CrossEntropyLoss; in short the forward pass offers an "early escape" which acts as an early classifier to predict the correct digit for each channel. The final output (after the full processing of the forward pass) however is still of size=1, and is a binary classifier for the main task "predicting if the first digit is less or equal to the second", and uses the BCEWithLogitsLoss. To train the network, we use a weighted sum of both losses. """

nb_classes_digits = 10
nb_classes_pairs = 2

def train_model_aux(model, train_input, train_target, train_classes, test_input, test_target, test_classes, optimizer, alpha, mini_batch_size=1000, nb_epochs=300, verbose=True):
    train_target = train_target.type(torch.FloatTensor).view(-1, 1) # from torch.Size([1000]) to torch.Size([1000, 1]) for BCEWithLogits
    nb_samples = len(train_input)
    criterion1 = torch.nn.CrossEntropyLoss()
    criterion2 = torch.nn.BCEWithLogitsLoss()  
    alpha = alpha # hyperparameter
    
    val_acc_history = []
    test_acc_history = []
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    since = time.time()
    for e in range(0, nb_epochs):
        
        # We need to loop over both train and eval mode to account for behaviors that might be different in train and eval mode such as BatchNorm or Dropout. We train in train mode and compute the accuracy in evaluation mode (more relevant).
        for phase in ['train', 'val']:
            if phase == 'train': model.train()
            else: model.eval()
                
            running_loss = 0.0
            running_corrects = 0
            running_correct_digits = 0
            
            for b in range(0, train_input.size(0), mini_batch_size):
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    x0_aux, x1_aux, x = model(train_input.narrow(0, b, mini_batch_size))
                    target = train_target.narrow(0, b, mini_batch_size)
                    classes = train_classes.narrow(0, b, mini_batch_size)
                    
                    loss0 = criterion1(x0_aux, classes[:,0])
                    loss1 = criterion1(x1_aux, classes[:,1])
                    loss2 = criterion2(x, target)
                    loss = loss2 + alpha*(loss0+loss1)
                    
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                nb_correct_digits = (torch.max(x0_aux, 1)[1] == classes[:,0]).sum().item() + (torch.max(x1_aux, 1)[1] == classes[:,1]).sum()
                               
                running_loss += loss.item() * train_input.size(0)
                output_to_prediction = torch.ge(torch.sigmoid(x), 0.5)
                running_corrects += torch.sum(output_to_prediction == target.type(torch.ByteTensor)) 
                running_correct_digits += nb_correct_digits

            epoch_loss = running_loss / nb_samples
            epoch_acc = running_corrects.double() / nb_samples
            epoch_acc_digits = running_correct_digits.double() / (2*nb_samples)
            
            if verbose and (e % 10 == 9):
                print('phase: %s, epoch: %d, loss: %.5f, acc: %.4f, acc_digits: %.4f' %
                      (phase, e+1, epoch_loss, epoch_acc, epoch_acc_digits))
                print()
                
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)
                
        test_acc = test_model_aux(model, test_input, test_target, test_classes, e)
        test_acc_history.append(test_acc)
                
    time_elapsed = time.time() - since
    print('Training complete in %.0f min %.0f s' % (time_elapsed // 60, time_elapsed % 60))
    print('Best val acc: %.4f' % (best_acc))
    
    model.load_state_dict(best_model_wts)
    return model, time_elapsed, best_acc.item(), val_acc_history, test_acc_history

######################################################################

def test_model_aux(model, test_input, test_target, test_classes, e):
    model.eval()
    class0, class1, test_output = model(test_input) 
    
    nb_correct_digits = (torch.max(class0, 1)[1] == test_classes[:,0]).sum().item() + (torch.max(class1, 1)[1] == test_classes[:,1]).sum().item()
    acc_digits = nb_correct_digits / (2*len(test_input))
                               
    output_to_prediction = torch.ge(torch.sigmoid(test_output), 0.5).flatten()
    nb_correct = (output_to_prediction == test_target.type(torch.ByteTensor)).sum().item()
    acc_pairs = nb_correct / len(test_input)
    
    if (e % 10 == 0): 
        print("e = {}, nb_correct_digits = {}, nb_correct_pairs = {}".format(e, nb_correct_digits, nb_correct))
        
    return acc_pairs

######################################################################

title_aux = 'Plot for 10 runs with auxiliary_losses framework (binary cross-entropy + auxiliary cross-entropy losses) and model = {}, \n accuracy obtained during training (model in eval mode) on the training set, and on the test set'

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
    
    print("mean and std train = {}, {} and mean and std test = {}, {}".format(train_input.mean(), train_input.std(), test_input.mean(), test_input.std()))
    
    return train_input, train_target, train_classes, test_input, test_target, test_classes
