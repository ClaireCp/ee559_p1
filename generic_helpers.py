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
              
    
            