import torch
from torch import nn
from torch.nn import functional as F

class NetSharing1(nn.Module):
    """ Base module, built upon ConvNet1_2C and foundation for Incept1a """
    def __init__(self, nb_hidden=100):
        super(NetSharing1, self).__init__()
        self.name = "NetSharing1"
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.fc1 = nn.Linear(512, nb_hidden)
        self.fc2 = nn.Linear(nb_hidden, 1)
 
    def forward(self, x):
        x0 = x[:,0,:,:].view(-1,1,14,14)
        x1 = x[:,1,:,:].view(-1,1,14,14)        
        x_list = [x0, x1]
        res = []        
        for x in x_list:
            x = F.relu(F.max_pool2d(self.conv1(x), kernel_size=2, stride=2))
            x = F.relu(F.max_pool2d(self.conv2(x), kernel_size=2, stride=2))
            x = x.view(-1, 256)
            res.append(x)
        x = torch.cat((res[0], res[1]),1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
    def return_new(self):
        return NetSharing1()

    
class NetSharing2(nn.Module):
    """ Let's decrease the number of parameters (a lot) - built upon ConvNet2_2C """
    def __init__(self, nb_hidden=40):
        super(NetSharing2, self).__init__()
        self.name = "NetSharing2"
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=5)
        self.fc1 = nn.Linear(144, nb_hidden)
        self.fc2 = nn.Linear(nb_hidden, 1)
        self.drop = nn.Dropout(p=0.5)
 
    def forward(self, x):
        x0 = x[:,0,:,:].view(-1,1,14,14)
        x1 = x[:,1,:,:].view(-1,1,14,14)
        x_list = [x0, x1]
        res = []
        for x in x_list:
            x = F.relu(F.max_pool2d(self.conv1(x), kernel_size=2, stride=1))
            x = F.relu(F.max_pool2d(self.conv2(x), kernel_size=2, stride=2))
            x = x.view(-1, 144)
            x = self.drop(F.relu(self.fc1(x)))
            res.append(x)
        x = res[1] - res[0]
        x = self.fc2(x)
        return x
    
    def return_new(self):
        return NetSharing2()
    
    
class NetSharing3(nn.Module):
    def __init__(self, nb_hidden=10):
        super(NetSharing3, self).__init__()
        self.name = "NetSharing3"
        self.conv1 = nn.Conv2d(1, 4, kernel_size=3)
        self.conv2 = nn.Conv2d(4, 8, kernel_size=4)
        self.conv3 = nn.Conv2d(8, 12, kernel_size=3)
        self.fc1 = nn.Linear(48, nb_hidden)
        self.fc2 = nn.Linear(nb_hidden, 1)
        self.drop = nn.Dropout(p=0.1)
 
    def forward(self, x):
        x0 = x[:,0,:,:].view(-1,1,14,14)
        x1 = x[:,1,:,:].view(-1,1,14,14)
        x_list = [x0, x1]
        res = []
        for x in x_list:
            x = F.relu(F.max_pool2d(self.conv1(x), kernel_size=2, stride=1))
            x = F.relu(F.max_pool2d(self.conv2(x), kernel_size=2, stride=1))
            x = F.relu(F.max_pool2d(self.conv3(x), kernel_size=2, stride=2))
            x = x.view(-1, 48)
            x = self.drop(F.relu(self.fc1(x)))
            res.append(x)
        x = res[0] - res[1]
        x = self.fc2(x)
        return x
    
    def return_new(self):
        return NetSharing3()
    
    
class NetSharing4(nn.Module):
    def __init__(self, nb_hidden=10):
        super(NetSharing4, self).__init__()
        self.name = "NetSharing4"
        self.conv1 = nn.Conv2d(1, 4, kernel_size=3)
        self.conv2 = nn.Conv2d(4, 6, kernel_size=4)
        self.fc1 = nn.Linear(96, nb_hidden)
        self.fc2 = nn.Linear(nb_hidden, 1)
        self.drop = nn.Dropout(p=0.15)
        
    def forward(self, x):
        x0 = x[:,0,:,:].view(-1,1,14,14)
        x1 = x[:,1,:,:].view(-1,1,14,14)
        x_list = [x0, x1]
        res = []
        for x in x_list:
            x = F.relu(F.max_pool2d(self.conv1(x), kernel_size=2, stride=1))
            x = F.relu(F.max_pool2d(self.conv2(x), kernel_size=2, stride=2))
            x = x.view(-1, 96)
            x = self.drop(F.relu(self.fc1(x)))
            res.append(x)
        x = res[0] - res[1]
        x = self.fc2(x)
        return x
    
    def return_new(self):
        return NetSharing4()
   