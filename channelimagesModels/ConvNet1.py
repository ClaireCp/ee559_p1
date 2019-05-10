from torch import nn
from torch.nn import functional as F

class ConvNet1_1C(nn.Module):
    """ Base (most simple) module for _1channel2images framework (1C) """
    def __init__(self, nb_classes=10, nb_hidden=200):
        super(ConvNet1_1C, self).__init__()
        self.name = "ConvNet1_1C"
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.fc1 = nn.Linear(256, nb_hidden)
        self.fc2 = nn.Linear(nb_hidden, nb_classes)

    def forward(self, x): # x.shape =  torch.Size([1000, 1, 14, 14])
        x = F.relu(F.max_pool2d(self.conv1(x), kernel_size=2, stride=2)) # x.shape =  torch.Size([1000, 32, 6, 6])
        x = F.relu(F.max_pool2d(self.conv2(x), kernel_size=2, stride=2)) # x.shape =  torch.Size([1000, 64, 2, 2])
        x = F.relu(self.fc1(x.view(-1, 256))) # self.fc1 applied to x with shape = torch.Size([1000, 256])
        x = self.fc2(x) # self.fc2 applied to x with shape = torch.Size([1000, nb_hidden])
        return x
    
    def return_new(self):
        return ConvNet1_1C()
    
class ConvNet1_2C(nn.Module):
    """ Base (most simple) module for _2channels1image framework (2C) """
    def __init__(self, nb_classes=1, nb_hidden=200):
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
    """ Let's go deeper and for more generated channels per convolutions! Also let's vary kernel size """
    def __init__(self, nb_classes=1, nb_hidden=200):
        super(ConvNet2_2C, self).__init__()
        self.name = "ConvNet2_2C"
        self.conv1 = nn.Conv2d(2, 64, kernel_size=3)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=5)
        self.fc1 = nn.Linear(128, nb_hidden)
        self.fc2 = nn.Linear(nb_hidden, nb_classes)

    def forward(self, x): # x.shape =  torch.Size([1000, 2, 14, 14])
        x = F.relu(F.max_pool2d(self.conv1(x), kernel_size=2, stride=2)) # x.shape =  torch.Size([1000, 64, 6, 6])
        x = F.relu(F.max_pool2d(self.conv2(x), kernel_size=2, stride=2)) # x.shape =  torch.Size([1000, 128, 1, 1])
        x = F.relu(self.fc1(x.view(-1, 128))) # self.fc1 applied to x with shape = torch.Size([1000, 128])
        x = self.fc2(x) # self.fc2 applied to x with shape = torch.Size([1000, nb_hidden])
        return x
    
    def return_new(self):
        return ConvNet2_2C()
