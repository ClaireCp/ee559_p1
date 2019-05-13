import torch
from torch import nn
from torch.nn import functional as F

class Incept1(nn.Module):
    """ Base module - Uses auxiliary losses AND weight sharing - Parallel of NetSharing1 (same CNN) """ #0.87 with nb_hidden = 500
    def __init__(self, nb_classes_digits=10, nb_classes_pairs=1, nb_hidden=50):
        super(Incept1, self).__init__()
        self.name = "Incept1"
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
        for x in x_list:
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
        return Incept1()
    
    
class Incept2(nn.Module): # 0.81 with nb_hidden=300
    """ Parallel of NetSharing3 (same CNN) """
    def __init__(self, nb_classes_digits=10, nb_classes_pairs=1, nb_hidden=100):
        super(Incept2, self).__init__()
        self.name = "Incept2"
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3)
        self.fc1 = nn.Linear(512, nb_hidden)
        self.fc2 = nn.Linear(nb_hidden, nb_classes_digits)
        self.fc4 = nn.Linear(512, nb_hidden)
        self.fc5 = nn.Linear(nb_hidden, nb_classes_pairs)
        
        self.conv2d_drop = nn.Dropout2d(p=0.1)
        self.drop1 = nn.Dropout(p=0.25)
        self.drop2 = nn.Dropout(p=0.15)
        
        self.conv2d_bn1 = nn.BatchNorm2d(64)
        self.conv2d_bn2 = nn.BatchNorm2d(128)
        self.bn = nn.BatchNorm1d(nb_hidden)

    def forward(self, x):
        x0 = x[:,0,:,:].view(-1,1,14,14)
        x1 = x[:,1,:,:].view(-1,1,14,14)    
        x_list = [x0, x1]
        res = []
        res_aux = []
        for x in x_list:
            x = self.conv2d_drop(self.conv2d_bn1(F.relu(F.max_pool2d(self.conv1(x), kernel_size=2, stride=2))))
            x = self.conv2d_drop(F.relu(F.max_pool2d(self.conv2d_bn2(self.conv2(x)), kernel_size=2, stride=2)))
            x = x.view(-1, 512)
            res.append(x)
            x_aux = F.relu(self.drop1(self.bn(self.fc1(x))))
            x_aux = F.relu(self.fc2(x_aux))
            res_aux.append(x_aux)
        x = res[0] - res[1]     
        x = self.drop1(self.bn(F.relu(self.fc4(x))))
        x = self.fc5(x)  
        return res_aux[0], res_aux[1], x
    
    def return_new(self):
        return Incept2()
   
    
class Incept3(nn.Module):
    """ Parallel to NetSharing4 (same CNN) """
    def __init__(self, nb_classes_digits=10, nb_classes_pairs=1, nb_hidden=40):
        super(Incept3, self).__init__()
        self.name = "Incept3"
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
        for x in x_list:
            x = F.relu(F.max_pool2d(self.conv1(x), kernel_size=2, stride=1))
            x = F.relu(F.max_pool2d(self.conv2(x), kernel_size=2, stride=2))
            x = x.view(-1, 144)
            x = self.drop(F.relu(self.fc1(x)))
            res.append(x)
            x_aux = self.fc3(x)
            res_aux.append(x_aux)    
        x = res[1] - res[0]
        x = self.fc2(x)
        return res_aux[0], res_aux[1], x
    
    def return_new(self):
        return Incept3()
    
    
class Incept4(nn.Module):
    """ Parallel to NetSharing5 (same CNN) """
    def __init__(self, nb_classes_digits=10, nb_classes_pairs=1, nb_hidden=10):
        super(Incept4, self).__init__()
        self.name = "Incept4"
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
        return Incept4()
    
class Incept5(nn.Module):
    """ Parallel to NetSharing6 """
    def __init__(self, nb_classes_digits=10, nb_classes_pairs=1, nb_hidden=6):
        super(Incept5, self).__init__()
        self.name = "Incept5"
        self.conv1 = nn.Conv2d(1, 3, kernel_size=3)
        self.conv2 = nn.Conv2d(3, 6, kernel_size=3)
        self.conv3 = nn.Conv2d(6, 12, kernel_size=4)
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
            res_aux.append(x)
        x = res[0] - res[1]
        x = self.fc2(x)
        return res_aux[0], res_aux[1], x
    
    def return_new(self):
        return Incept5()
    
class Incept6(nn.Module):
    """ Parallel to NetSharing9 """
    def __init__(self, nb_classes_digits=10, nb_classes_pairs=1, nb_hidden=10):
        super(Incept6, self).__init__()
        self.name = "Incept6"
        self.conv1 = nn.Conv2d(1, 4, kernel_size=3)
        self.conv2 = nn.Conv2d(4, 6, kernel_size=4)
        self.fc1 = nn.Linear(2*96, nb_hidden)
        self.fc2 = nn.Linear(nb_hidden, 1)
        self.fc3 = nn.Linear(96, nb_classes_digits)
        self.drop = nn.Dropout(p=0.15)
 
    def forward(self, x):
        x0 = x[:,0,:,:].view(-1,1,14,14)
        x1 = x[:,1,:,:].view(-1,1,14,14)
        x_list = [x0, x1]
        res = []
        res_aux = []
        for x in x_list:
            x = F.relu(F.max_pool2d(self.conv1(x), kernel_size=2, stride=1))
            x = F.relu(F.max_pool2d(self.conv2(x), kernel_size=2, stride=2))
            x = x.view(-1, 96)
            res.append(x)
            x_aux = self.fc3(x)
            res_aux.append(x_aux)
 
        x = torch.cat((res[0], res[1]),1)
        x = self.drop(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return res_aux[0], res_aux[1], x
    
    def return_new(self):
        return Incept6()