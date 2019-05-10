import torch
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F

import sys
sys.path.insert(0, "../")
from generic_helpers import plot_history
from generic_helpers import compute_properties
from generic_helpers import mean_per_epoch_list
from test import load_random_datasets
from generic_helpers import count_parameters

import time
import copy


""" FRAMEWORK FOR INPUT AS TWO SINGLE CHANNEL IMAGES """
""" In this framework, the network is first trained to recognize the digits of each image from each pair and with the help of the class labels. To do so, we use the class labels provided and use a CrossEntropyLoss to maximize the response of the correct digit. Once the network can predict the digits, we compare the digits and define if they are a pair or not """

nb_classes = 10
nb_input_channels = 1

def train_model_1C(model, train_input, train_classes, test_input, test_target, test_classes, optimizer, mini_batch_size=1000, nb_epochs=300, verbose=True):
    criterion = torch.nn.CrossEntropyLoss()
    train_input = train_input.view(-1,1,14,14) # 2000x1x14x14 from 1-image-2-channels to 2-images-1-channel
    train_classes = train_classes.flatten() # the target are the (flattened) corresponding digit class labels 
    nb_samples = len(train_input)
    
    val_acc_history = []
    test_acc_history = []
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    since = time.time()
    for e in range(0, nb_epochs):
        
        for phase in ['train', 'val']:
            if phase == 'train': model.train()
            else: model.eval()
                
            running_loss = 0.0
            running_corrects = 0
            
            for b in range(0, train_input.size(0), mini_batch_size):
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    output = model(train_input.narrow(0, b, mini_batch_size))
                    target = train_classes.narrow(0, b, mini_batch_size)
                    
                    loss = criterion(output, target)
                    
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        
                running_loss += loss.item() * train_input.size(0) # mutliply by batch_size
                running_corrects += torch.sum(torch.max(output, 1)[1] == target) # for the batch   

            # then compute the average loss and the accuracy over the full batch
            epoch_loss = running_loss / nb_samples 
            epoch_acc = running_corrects.double() / nb_samples
            
            if verbose and (e % 100 == 99):
                print('phase: %s, epoch: %d, loss: %.5f, acc: %.4f' %
                      (phase, e+1, epoch_loss, epoch_acc))
                
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)
        
        test_acc_digits, _ = test_model_1C(model, test_input, test_target, test_classes)
        test_acc_history.append(test_acc_digits)
                
    time_elapsed = time.time() - since
    print('Training complete in %.0f min %.0f s' % (time_elapsed // 60, time_elapsed % 60))
    print('Best val acc: %.4f' % (best_acc))      
    
    model.load_state_dict(best_model_wts)
    return model, time_elapsed, best_acc.item(), val_acc_history, test_acc_history

######################################################################

def compare_pairs(model, input_):    
    tensor_a = torch.max(model(input_[:,0,:,:].view(-1,1,14,14)), 1)[1]
    tensor_b = torch.max(model(input_[:,1,:,:].view(-1,1,14,14)),1)[1]
    return torch.le(tensor_a, tensor_b)

def test_model_1C(model, test_input, test_target, test_classes):
    model.eval()
    test_input_flat = test_input.view(-1,1,14,14)
    test_classes = test_classes.flatten()
    
    output = model(test_input_flat)
    predicted_digits = torch.max(output, 1)[1]
    nb_correct_digits = (predicted_digits == test_classes).sum().item()
    acc_digits = nb_correct_digits / len(test_input_flat) # Test accuracy on task = predicting digits
    
    predicted_pairs = compare_pairs(model, test_input).type(torch.LongTensor)
    nb_correct_pairs = (predicted_pairs == test_target).sum().item()
    acc_pairs = nb_correct_pairs / len(test_input) # Test accuracy on task = comparison of pairs
    
    return acc_digits, acc_pairs

######################################################################

def multiple_training_runs_1C(model_ref, nb_runs, lr, mini_batch_size=1000, nb_epochs=300, verbose=True, plot=True):
    list_time = []
    list_best_acc = [] # (best) accuracy on training set
    list_acc_digits = [] # predicting digits accuracy on test set
    list_acc_pairs = [] # comparing pairs accuracy on test set
    list_val_acc_history = []
    list_test_acc_history = []
            
    for i in range(nb_runs):
        # We load new random training and test sets
        train_input, _, train_classes, test_input, test_target, test_classes = load_random_datasets()
        
        # and create a new instance of our model to have new random initial weights
        model = model_ref.return_new()     
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.95)
        
        model, time_elapsed, best_acc, val_acc_history, test_acc_history = train_model_1C(model, train_input, train_classes, test_input, test_target, test_classes, optimizer, mini_batch_size, nb_epochs, verbose)
        list_time.append(time_elapsed)
        list_best_acc.append(best_acc)
        list_val_acc_history.append(val_acc_history)
        list_test_acc_history.append(test_acc_history)
        
        acc_digits, acc_pairs = test_model_1C(model, test_input, test_target, test_classes)
        list_acc_digits.append(acc_digits)
        list_acc_pairs.append(acc_pairs)
        
    if plot == True:
        title = 'Plot for 10 runs with _1channel2images framework (cross-entropy loss) and model = {}, \n accuracy obtained during training (model in eval mode) on the training set, and on the test set \n Note: here, the accuracy is for the task "predicting digits"'.format(model.name)
        mean_val_acc_history, std_val_acc_history = mean_per_epoch_list(list_val_acc_history)
        mean_test_acc_history, std_test_acc_history = mean_per_epoch_list(list_test_acc_history)
        plot_history(mean_val_acc_history, std_val_acc_history, mean_test_acc_history, std_test_acc_history, title)
        
    mean_time, std_time = compute_properties(list_time)
    mean_best_acc, std_best_acc = compute_properties(list_best_acc)
    mean_acc_digits, std_acc_digits = compute_properties(list_acc_digits)
    mean_acc_pairs, std_acc_pairs = compute_properties(list_acc_pairs)
    
    return mean_time, std_time, mean_best_acc, std_best_acc, mean_acc_digits, std_acc_digits, mean_acc_pairs, std_acc_pairs
                             
                             
######################################################################
import csv

def check_overwrite(filename, model, lr, nb_epochs, row_to_write):
    overwrite = False
    with open(filename, 'r') as readFile:
        reader = csv.reader(readFile)
        row_list = list(reader)
        for index, row in enumerate(row_list):
            if (row[0] == model.name) and (row[2] == str(lr)) and (row[3] == str(nb_epochs)): 
                row_list[index] = row_to_write
                overwrite = True           
                break
    with open(filename, 'w') as writeFile:
        print("Overwriting file")
        writer = csv.writer(writeFile)
        writer.writerows(row_list)
    writeFile.close()
    readFile.close()
    return overwrite
    
def write_to_csv_1C(filename, model, test_results, lr, nb_epochs):
    nb_params = count_parameters(model)
    row = [model.name, nb_params, lr, nb_epochs, round(test_results[0], 2), 
           round(test_results[2], 4), round(test_results[3], 4), 
           round(test_results[4], 4), round(test_results[5], 4)]
    
    try: file = open(filename, 'r')
    except FileNotFoundError:
        csvData = [['Model', 'Number of parameters', 'Learning rate', 'Number of epochs', 'Training time', 'Mean digits accuracy (test set)', 'Std digits accuracy', 'Mean accuracy (test set)', 'Std accuracy']]
        with open('1channel2images.csv', 'w') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerows(csvData)
        csvFile.close()
        return
        
    overwrite = check_overwrite(filename, model, lr, nb_epochs, row)
    if overwrite == False:    
        with open(filename, 'a') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow(row)
        csvFile.close()
