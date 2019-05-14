import torch
from torch import nn
from torch.nn import functional as F

class Incept1a(nn.Module):
    """ Base module - Uses auxiliary losses AND weight sharing - built upon ConvNet1_2C and NetSharing1 (same CNN) """
    def __init__(self, nb_classes_digits=10, nb_classes_pairs=1, nb_hidden=100):
        super(Incept1a, self).__init__()
        self.name = "Incept1a"
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.fc1 = nn.Linear(256, nb_hidden)
        self.fc2 = nn.Linear(nb_hidden, nb_classes_digits)       
        self.fc3 = nn.Linear(512, nb_hidden)
        self.fc4 = nn.Linear(nb_hidden, nb_classes_pairs)

    def forward(self, x):
        x0 = x[:,0,:,:].view(-1,1,14,14)
        x1 = x[:,1,:,:].view(-1,1,14,14)       
        x_list = [x0, x1]
        res = []
        res_aux = []       
        for x in x_list: # Shared CNN and MLP for auxiliary losses
            x = F.relu(F.max_pool2d(self.conv1(x), kernel_size=2, stride=2))
            x = F.relu(F.max_pool2d(self.conv2(x), kernel_size=2, stride=2))
            x = x.view(-1, 256)
            res.append(x)
            x_aux = F.relu(self.fc1(x))
            x_aux = self.fc2(x_aux)
            res_aux.append(x_aux)  
        x = torch.cat((res[0], res[1]),1)
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return res_aux[0], res_aux[1], x
    
    def return_new(self):
        return Incept1a()
    
    
class Incept1b(nn.Module):
    """ Base module - Uses auxiliary losses BUT NO weight sharing - built upon ConvNet1_2C """
    def __init__(self, nb_classes_digits=10, nb_classes_pairs=1, nb_hidden=100):
        super(Incept1b, self).__init__()
        self.name = "Incept1b"
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.conv3 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv4 = nn.Conv2d(32, 64, kernel_size=3)
        self.fc1 = nn.Linear(256, nb_hidden)
        self.fc2 = nn.Linear(nb_hidden, nb_classes_digits)   
        self.fc3 = nn.Linear(256, nb_hidden)
        self.fc4 = nn.Linear(nb_hidden, nb_classes_digits)   
        self.fc5 = nn.Linear(512, nb_hidden)
        self.fc6 = nn.Linear(nb_hidden, nb_classes_pairs)

    def forward(self, x):
        x0 = x[:,0,:,:].view(-1,1,14,14)
        x1 = x[:,1,:,:].view(-1,1,14,14)        
        # CNN and MLP for auxiliary loss for channel 0
        x0 = F.relu(F.max_pool2d(self.conv1(x0), kernel_size=2, stride=2))
        x0 = F.relu(F.max_pool2d(self.conv2(x0), kernel_size=2, stride=2))
        x0 = x0.view(-1, 256)
        x0_aux = F.relu(self.fc1(x0))
        x0_aux = self.fc2(x0_aux)
        # CNN and MLP for auxiliary loss for channel 1
        x1 = F.relu(F.max_pool2d(self.conv3(x1), kernel_size=2, stride=2))
        x1 = F.relu(F.max_pool2d(self.conv4(x1), kernel_size=2, stride=2))
        x1 = x1.view(-1, 256)
        x1_aux = F.relu(self.fc3(x1))
        x1_aux = self.fc4(x1_aux)
        # MLP for main loss
        x = torch.cat((x0, x1),1)
        x = F.relu(self.fc5(x))
        x = self.fc6(x)
        return x0_aux, x1_aux, x
    
    def return_new(self):
        return Incept1b()
    

class Incept2(nn.Module):
    """ Built upon ConvNet3_2C and NetSharing2 (same CNN and same number of hidden units for siamese MLP) """
    def __init__(self, nb_classes_digits=10, nb_classes_pairs=1, nb_hidden=40):
        super(Incept2, self).__init__()
        self.name = "Incept2"
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=5)
        self.fc1 = nn.Linear(144, nb_hidden)
        self.fc2 = nn.Linear(nb_hidden, nb_classes_pairs)
        self.fc3 = nn.Linear(nb_hidden, nb_classes_digits)
        self.drop = nn.Dropout(p=0.5)
 
    def forward(self, x):
        x0 = x[:,0,:,:].view(-1,1,14,14)
        x1 = x[:,1,:,:].view(-1,1,14,14)
        x_list = [x0, x1]
        res = []
        res_aux = []
        for x in x_list: # Shared (CNN + 1FC layer) and MLP for auxiliary losses
            x = F.relu(F.max_pool2d(self.conv1(x), kernel_size=2, stride=1))
            x = F.relu(F.max_pool2d(self.conv2(x), kernel_size=2, stride=2))
            x = x.view(-1, 144)
            x = self.drop(F.relu(self.fc1(x))) # Additional shared linear layer!
            res.append(x)
            x_aux = self.fc3(x)
            res_aux.append(x_aux)    
        x = res[1] - res[0]
        x = self.fc2(x)
        return res_aux[0], res_aux[1], x
    
    def return_new(self):
        return Incept2()
    
    
class Incept3(nn.Module): #0.83 with nb_hidden=100
    """ Built upon NetSharing3 (same CNN and same number of hidden units for siamese MLP) """
    def __init__(self, nb_classes_digits=10, nb_classes_pairs=1, nb_hidden=10):
        super(Incept3, self).__init__()
        self.name = "Incept3"
        self.conv1 = nn.Conv2d(1, 4, kernel_size=3)
        self.conv2 = nn.Conv2d(4, 8, kernel_size=4)
        self.conv3 = nn.Conv2d(8, 12, kernel_size=3)
        self.fc1 = nn.Linear(48, nb_hidden)
        self.fc2 = nn.Linear(48, nb_classes_pairs)
        self.fc3 = nn.Linear(nb_hidden, nb_classes_digits)
        self.drop = nn.Dropout(p=0)
 
    def forward(self, x):
        x0 = x[:,0,:,:].view(-1,1,14,14)
        x1 = x[:,1,:,:].view(-1,1,14,14)
        x_list = [x0, x1]
        res = []
        res_aux = []
        for x in x_list:
            x = F.relu(F.max_pool2d(self.conv1(x), kernel_size=2, stride=1))
            x = F.relu(F.max_pool2d(self.conv2(x), kernel_size=2, stride=1))
            x = F.relu(F.max_pool2d(self.conv3(x), kernel_size=2, stride=2))
            x = x.view(-1, 48)
            res.append(x)
            x_aux = self.drop(F.relu(self.fc1(x)))
            x_aux = self.fc3(x_aux)
            res_aux.append(x_aux)
        x = res[0] - res[1]
        x = self.fc2(x)
        return res_aux[0], res_aux[1], x
    
    def return_new(self):
        return Incept3()
    

class Incept4(nn.Module):
    """ Built upon NetSharing4 (same CNN and same number of hidden units for siamese MLP) """
    def __init__(self, nb_classes_digits=10, nb_classes_pairs=1, nb_hidden=10):
        super(Incept4, self).__init__()
        self.name = "Incept4"
        self.conv1 = nn.Conv2d(1, 4, kernel_size=3)
        self.conv2 = nn.Conv2d(4, 6, kernel_size=4)
        self.fc1 = nn.Linear(96, nb_hidden)
        self.fc2 = nn.Linear(nb_hidden, nb_classes_pairs)
        self.fc3 = nn.Linear(nb_hidden, nb_classes_digits)
        self.drop = nn.Dropout(p=0.15)
 
    def forward(self, x):
        x0 = x[:,0,:,:].view(-1,1,14,14)
        x1 = x[:,1,:,:].view(-1,1,14,14)
        x_list = [x0, x1]
        res = []
        res_aux = []
        for x in x_list: # Shared (CNN + 1FC layer) and MLP for auxialiary losses
            x = F.relu(F.max_pool2d(self.conv1(x), kernel_size=2, stride=1))
            x = F.relu(F.max_pool2d(self.conv2(x), kernel_size=2, stride=2))
            x = x.view(-1, 96)
            x = self.drop(F.relu(self.fc1(x)))
            res.append(x)
            x_aux = self.fc3(x)
            res_aux.append(x)
        x = res[0] - res[1]
        x = self.fc2(x)
        return res_aux[0], res_aux[1], x
    
    def return_new(self):
        return Incept4()
    