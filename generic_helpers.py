""" This source file contains generic functions that are valid and used in particular for _2channels1image, weight_sharing and  auxiliary_losses """
import torch
from torch import nn, optim
import dlc_practical_prologue as prologue

def load_random_datasets():
    """ Function to generate new random training and test sets for various runs """
    train_input, train_target, train_classes, test_input, test_target, test_classes = \
    prologue.generate_pair_sets(nb=1000)
    
    """ Feature Normalization: We normalize the train and test datasets with the mean and variance of the training set (we don't want to introduce information of the test set into the training set by normalizing over the whole dataset)"""
    train_input, mean_input, std_input = standardize(train_input)
    test_input = (test_input - mean_input) / std_input
    
    return train_input, train_target, train_classes, test_input, test_target, test_classes


######################################################################
""" Generic function for multiple training runs for _2channels1image, weight_sharing and  auxiliary_losses. Arguments:
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
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.95, weight_decay= 0.005)
        
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


######################################################################
""" Writing to csv function, valid for _2channels1image, weight_sharing and  auxiliary_losses """
import csv
import matplotlib.pyplot as plt
from operator import add, neg

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

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
        writer = csv.writer(writeFile)
        writer.writerows(row_list)
    readFile.close()
    writeFile.close()
    return overwrite
    
def write_to_csv(filename, model, test_results, lr, nb_epochs):
    nb_params = count_parameters(model)
    row = [model.name, nb_params, lr, nb_epochs, round(test_results[0], 2), 
           round(test_results[2], 4), round(test_results[3], 4),
           round(test_results[4], 4), round(test_results[5], 4)]
    
    try: file = open(filename, 'r')
    except FileNotFoundError:
        csvData = [['Model', 'Number of parameters', 'Learning rate', 'Number of epochs', 'Training time', 'Mean best training accuracy', 'Std best training accuracy',
                    'Mean test accuracy', 'Std test accuracy']]
        with open(filename, 'w') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerows(csvData)
        csvFile.close()
        
    overwrite = check_overwrite(filename, model, lr, nb_epochs, row)
    if overwrite == False:    
        with open(filename, 'a') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow(row)
        csvFile.close()

        
######################################################################
""" compute_properties function in order not to have to use the math library """
def compute_properties(lst):
    mean = sum(lst) / len(lst)
    variance = sum([(e-mean)**2 for e in lst]) / (len(lst)-1)
    return mean, variance ** (1/2)

def standardize(x):
    mean_x = x.mean()
    x = x - mean_x
    std_x = x.std()
    x = x / std_x
    return x, mean_x, std_x

def mean_per_epoch_list(lst_lst):
    """ compute the mean value across the multiple runs for each epoch """
    list_per_epoch = list(zip(*lst_lst))
    mean_per_epoch_list = []
    std_per_epoch_list = []
    for lst in list_per_epoch:
        mean, std = compute_properties(lst)
        mean_per_epoch_list.append(mean)
        std_per_epoch_list.append(std)
    return mean_per_epoch_list, std_per_epoch_list
        
    
######################################################################
def plot_history(mean_val_acc_history, std_val_acc_history, mean_test_acc_history, std_test_acc_history, title):
    fig, ax = plt.subplots()
    fig.set_size_inches(12, 8)
    fig.suptitle(title, fontsize=16)
    ax.set_xlabel('epoch')
    ax.set_ylabel('accuracy [% correct]')
    ax.plot(mean_val_acc_history, label='training accuracy')
    ax.plot(mean_test_acc_history, label='test accuracy')
    plot_std(ax, mean_val_acc_history, std_val_acc_history)
    plot_std(ax, mean_test_acc_history, std_test_acc_history)
    ax.legend(loc='upper left')
    fig.tight_layout()
    plt.subplots_adjust(top=0.89)
    return fig

def plot_std(ax, mean, std):
    x = list(range(len(mean)))
    y1 = list(map(add, mean, std))
    y2 = list(map(add, mean, map(neg, std)))
    ax.fill_between(x, y1, y2, alpha=0.4)
            