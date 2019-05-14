from torch import nn
from torch.nn import functional as F

class ConvNet1_1C(nn.Module):
    """ Base (most simple) module for _1channel2images framework (1C) """
    def __init__(self, nb_classes=10, nb_hidden=50):
        super(ConvNet1_1C, self).__init__()
        self.name = "ConvNet1_1C"
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.fc1 = nn.Linear(256, nb_hidden)
        self.fc2 = nn.Linear(nb_hidden, nb_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), kernel_size=2, stride=2))
        x = F.relu(F.max_pool2d(self.conv2(x), kernel_size=2, stride=2))
        x = F.relu(self.fc1(x.view(-1, 256)))
        x = self.fc2(x)
        return x
    
    def return_new(self):
        return ConvNet1_1C()
    
######################################################################
""" Neural Network modules for _2channels1image framework (2C) """
   
class ConvNet1_2C(nn.Module):
    """ Base (most simple) module - Serves as foundation network for NetSharing1 and Incept1 """
    def __init__(self, nb_classes=1, nb_hidden=100):
        super(ConvNet1_2C, self).__init__()
        self.name = "ConvNet1_2C"
        self.conv1 = nn.Conv2d(2, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.fc1 = nn.Linear(256, nb_hidden)
        self.fc2 = nn.Linear(nb_hidden, nb_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), kernel_size=2, stride=2))
        x = F.relu(F.max_pool2d(self.conv2(x), kernel_size=2, stride=2))
        x = F.relu(self.fc1(x.view(-1, 256)))
        x = self.fc2(x)
        return x
    
    def return_new(self):
        return ConvNet1_2C()

    
class ConvNet2_2C(nn.Module):
    """ Serves as foundation network for NetSharing2 and Incept2"""
    def __init__(self, nb_classes=1, nb_hidden=40):
        super(ConvNet2_2C, self).__init__()
        self.name = "ConvNet2_2C"
        self.conv1 = nn.Conv2d(2, 8, kernel_size=3)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=5)
        self.fc1 = nn.Linear(144, nb_hidden)
        self.fc2 = nn.Linear(nb_hidden, nb_classes)
        
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), kernel_size=2, stride=1))
        x = F.relu(F.max_pool2d(self.conv2(x), kernel_size=2, stride=2))
        x = F.relu(self.fc1(x.view(-1, 144)))
        x = self.fc2(x)
        return x
    
    def return_new(self):
        return ConvNet2_2C()
        
        
class ConvNet4_2C(nn.Module):
    """ Serves as foundation network for NetSharing4 and Incept4"""
    def __init__(self, nb_classes=1, nb_hidden=10):
        super(ConvNet4_2C, self).__init__()
        self.name = "ConvNet4_2C"
        self.conv1 = nn.Conv2d(2, 4, kernel_size=3)
        self.conv2 = nn.Conv2d(4, 6, kernel_size=4)
        self.fc1 = nn.Linear(96, nb_hidden)
        self.fc2 = nn.Linear(nb_hidden, 1)
        self.drop = nn.Dropout(p=0.15)
        
    def forward(self, x):
        x0 = x[:,0,:,:].view(-1,1,14,14)
        x1 = x[:,1,:,:].view(-1,1,14,14)
        x = F.relu(F.max_pool2d(self.conv1(x), kernel_size=2, stride=1))
        x = F.relu(F.max_pool2d(self.conv2(x), kernel_size=2, stride=2))
        x = self.drop(F.relu(self.fc1(x.view(-1, 96))))
        x = self.fc2(x)
        return x
    
    def return_new(self):
        return ConvNet4_2C()
        
        
        
        
    
