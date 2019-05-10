import torch
from torch import nn
from torch.nn import functional as F

class NetSharing1(nn.Module):
    """ Base (most simple) module """
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
        
        x0 = F.relu(F.max_pool2d(self.conv1(x0), kernel_size=2, stride=2))
        x0 = F.relu(F.max_pool2d(self.conv2(x0), kernel_size=2, stride=2))
        x0 = x0.view(-1, 256)
 
        x1 = F.relu(F.max_pool2d(self.conv1(x1), kernel_size=2, stride=2))
        x1 = F.relu(F.max_pool2d(self.conv2(x1), kernel_size=2, stride=2))
        x1 = x1.view(-1, 256)
 
        x = torch.cat((x0,x1),1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
    def return_new(self):
        return NetSharing1()

class NetSharing2(nn.Module):
    """ Increase the number of channels produced by the convolutions """
    def __init__(self, nb_hidden=100):
        super(NetSharing2, self).__init__()
        self.name = "NetSharing2"
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3)
        self.fc1 = nn.Linear(1024, nb_hidden)
        self.fc2 = nn.Linear(nb_hidden, 1)
 
    def forward(self, x):
        x0 = x[:,0,:,:].view(-1,1,14,14)
        x1 = x[:,1,:,:].view(-1,1,14,14)
        
        x0 = F.relu(F.max_pool2d(self.conv1(x0), kernel_size=2, stride=2))
        x0 = F.relu(F.max_pool2d(self.conv2(x0), kernel_size=2, stride=2))
        x0 = x0.view(-1, 512)
 
        x1 = F.relu(F.max_pool2d(self.conv1(x1), kernel_size=2, stride=2))
        x1 = F.relu(F.max_pool2d(self.conv2(x1), kernel_size=2, stride=2))
        x1 = x1.view(-1, 512)
 
        x = torch.cat((x0,x1),1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
    def return_new(self):
        return NetSharing2()
    
class NetSharing3(nn.Module):
    """ Add regularization using Dropout (zero out units) and Dropout2d (zero out entire channels) """
    def __init__(self, nb_hidden=100):
        super(NetSharing3, self).__init__()
        self.name = "NetSharing3"
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3)
        self.fc1 = nn.Linear(1024, nb_hidden)
        self.fc2 = nn.Linear(nb_hidden, 1)       
        #self.conv2d_drop = nn.Dropout2d(p=0.2)
        #self.drop = nn.Dropout(p=0.15)
        self.conv2d_drop = nn.Dropout2d(p=0.35)
        self.drop = nn.Dropout(p=0.3)
 
    def forward(self, x):
        x0 = x[:,0,:,:].view(-1,1,14,14)
        x1 = x[:,1,:,:].view(-1,1,14,14)
        
        x0 = F.relu(F.max_pool2d(self.conv1(x0), kernel_size=2, stride=2))
        x0 = F.relu(F.max_pool2d(self.conv2d_drop(self.conv2(x0)), kernel_size=2, stride=2))
        x0 = x0.view(-1, 512)
        x0 = self.drop(x0)
 
        x1 = F.relu(F.max_pool2d(self.conv1(x1), kernel_size=2, stride=2))
        x1 = F.relu(F.max_pool2d(self.conv2d_drop(self.conv2(x1)), kernel_size=2, stride=2))
        x1 = x1.view(-1, 512)
        x1 = self.drop(x1)
 
        x = torch.cat((x0,x1),1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
    def return_new(self):
        return NetSharing3()