import torch
from torch import nn
from torch.nn import functional as F

class Incept1(nn.Module):
    """ Base (most simple) module - Uses auxiliary losses AND weight sharing """
    def __init__(self, nb_classes_digits=10, nb_classes_pairs=1, nb_hidden=500):
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
        x1 = x[:,0,:,:].view(-1,1,14,14)
        
        x0 = F.relu(F.max_pool2d(self.conv1(x0), kernel_size=2, stride=2))
        x0 = F.relu(F.max_pool2d(self.conv2(x0), kernel_size=2, stride=2))
        x0 = x0.view(-1, 256)
        x0_aux = F.relu(self.fc1(x0))
        x0_aux = self.fc2(x0_aux)
        
        x1 = F.relu(F.max_pool2d(self.conv1(x1), kernel_size=2, stride=2))
        x1 = F.relu(F.max_pool2d(self.conv2(x1), kernel_size=2, stride=2))
        x1 = x1.view(-1, 256)
        x1_aux = F.relu(self.fc1(x1))
        x1_aux = self.fc2(x1_aux)
        
        x = torch.cat((x0, x1),1)
        x = F.relu(self.fc3(x.view(-1,512)))
        x = self.fc4(x)
        
        return x0_aux, x1_aux, x
    
    def return_new(self):
        return Incept1()
    
class Incept11(nn.Module):
    def __init__(self, nb_classes_digits=10, nb_classes_pairs=1, nb_hidden=100):
        super(Incept11, self).__init__()
        self.name = "Incept11"
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.fc1 = nn.Linear(256, nb_hidden)
        self.fc2 = nn.Linear(nb_hidden, nb_classes_digits)       
        self.fc3 = nn.Linear(512, nb_hidden)
        self.fc4 = nn.Linear(nb_hidden, 40)
        self.fc5 = nn.Linear(40, nb_classes_pairs)
        
        self.drop1 = nn.Dropout(p=0.50)
        self.drop2 = nn.Dropout(p=0.15)

    def forward(self, x):
        x0 = x[:,0,:,:].view(-1,1,14,14)
        x1 = x[:,0,:,:].view(-1,1,14,14)
        
        x0 = F.relu(F.max_pool2d(self.conv1(x0), kernel_size=2, stride=2))
        x0 = F.relu(F.max_pool2d(self.conv2(x0), kernel_size=2, stride=2))
        x0 = x0.view(-1, 256)
        x0_aux = self.drop1(F.relu(self.fc1(x0)))
        x0_aux = self.fc2(x0_aux)
        
        x1 = F.relu(F.max_pool2d(self.conv1(x1), kernel_size=2, stride=2))
        x1 = F.relu(F.max_pool2d(self.conv2(x1), kernel_size=2, stride=2))
        x1 = x1.view(-1, 256)
        x1_aux = self.drop1(F.relu(self.fc1(x1)))
        x1_aux = self.fc2(x1_aux)
        
        x = torch.cat((x0, x1),1)
        x = self.drop1(F.relu(self.fc3(x.view(-1,512))))
        x = self.drop1(F.relu(self.fc4(x)))
        x = self.fc5(x)                   
        
        return x0_aux, x1_aux, x
    
    def return_new(self):
        return Incept11()

    
class Incept2(nn.Module):
    def __init__(self, nb_classes_digits=10, nb_classes_pairs=1, nb_hidden=300):
        super(Incept2, self).__init__()
        self.name = "Incept2"
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3)
        self.fc1 = nn.Linear(512, nb_hidden)
        self.fc2 = nn.Linear(nb_hidden, 40)
        self.fc3 = nn.Linear(40, nb_classes_digits)
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
        x1 = x[:,0,:,:].view(-1,1,14,14)
        
        x0 = self.conv2d_drop(self.conv2d_bn1(F.relu(F.max_pool2d(self.conv1(x0), kernel_size=2, stride=2))))
        x0 = self.conv2d_drop(F.relu(F.max_pool2d(self.conv2d_bn2(self.conv2(x0)), kernel_size=2, stride=2)))

        x0_aux = x0.view(-1, 512)
        # MLP after siamese convnet
        x0_aux = F.relu(self.drop1(self.bn(self.fc1(x0_aux))))
        x0_aux = F.relu(self.fc2(x0_aux))
        x0_aux = F.relu(self.fc3(x0_aux))
        
        x1 = self.conv2d_drop(self.conv2d_bn1(F.relu(F.max_pool2d(self.conv1(x1), kernel_size=2, stride=2))))
        x1 = self.conv2d_drop(F.relu(F.max_pool2d(self.conv2d_bn2(self.conv2(x1)), kernel_size=2, stride=2)))
        x1_aux = x1.view(-1, 512)
        # MLP after siamese convnet
        x1_aux = F.relu(self.drop1(self.bn(self.fc1(x1_aux))))
        x1_aux = F.relu(self.fc2(x1_aux))
        x1_aux = F.relu(self.fc3(x1_aux))
        
        x = (x0 - x1).view(-1, 512)
        x = self.drop1(self.bn(F.relu(self.fc4(x))))
        x = self.fc5(x)  
        return x0_aux, x1_aux, x
    
    def return_new(self):
        return Incept2()
   
    
class Incept22(nn.Module):
    """ Increase the number of channels produced by the convolutions. Add regularization using Dropout (zero out units)  and Dropout2d (zero out entire channels), and BatchNorm. """
    def __init__(self, nb_classes_digits=10, nb_classes_pairs=1, nb_hidden=40):
        super(Incept22, self).__init__()
        self.name = "Incept22"
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3)
        self.fc1 = nn.Linear(512, nb_hidden)
        self.fc2 = nn.Linear(nb_hidden, 40)
        self.fc3 = nn.Linear(40, nb_classes_digits)
        self.fc4 = nn.Linear(1024, nb_hidden)
        self.fc5 = nn.Linear(nb_hidden, nb_classes_pairs)
        
        self.conv2d_drop = nn.Dropout2d(p=0)
        self.drop1 = nn.Dropout(p=0.25)
        self.drop2 = nn.Dropout(p=0.15)
        
        self.conv2d_bn1 = nn.BatchNorm2d(64)
        self.conv2d_bn2 = nn.BatchNorm2d(128)
        self.bn = nn.BatchNorm1d(nb_hidden)

    def forward(self, x):
        x0 = x[:,0,:,:].view(-1,1,14,14)
        x1 = x[:,0,:,:].view(-1,1,14,14)
        
        x0 = self.conv2d_drop(self.conv2d_bn1(F.relu(F.max_pool2d(self.conv1(x0), kernel_size=2, stride=2))))
        x0 = self.conv2d_drop(F.relu(F.max_pool2d(self.conv2d_bn2(self.conv2(x0)), kernel_size=2, stride=2)))

        x0_aux = x0.view(-1, 512)
        # MLP after siamese convnet
        x0_aux = F.relu(self.drop1(self.bn(self.fc1(x0_aux))))
        x0_aux = F.relu(self.fc2(x0_aux))
        x0_aux = F.relu(self.fc3(x0_aux))
        
        x1 = self.conv2d_drop(self.conv2d_bn1(F.relu(F.max_pool2d(self.conv1(x1), kernel_size=2, stride=2))))
        x1 = self.conv2d_drop(F.relu(F.max_pool2d(self.conv2d_bn2(self.conv2(x1)), kernel_size=2, stride=2)))
        x1_aux = x1.view(-1, 512)
        # MLP after siamese convnet
        x1_aux = F.relu(self.drop1(self.bn(self.fc1(x1_aux))))
        x1_aux = F.relu(self.fc2(x1_aux))
        x1_aux = F.relu(self.fc3(x1_aux))
        
        x = torch.cat((x0.view(-1, 512), x1.view(-1, 512)), 1)
        x = self.drop1(self.bn(F.relu(self.fc4(x))))
        x = self.fc5(x)    
        
        return x0_aux, x1_aux, x
    
    def return_new(self):
        return Incept22()
   
    
class Incept4(nn.Module):
    """ Similar to base (most simple) module - Uses auxiliary losses ONLY (no weight sharing) """
    def __init__(self, nb_classes_digits=10, nb_classes_pairs=1, nb_hidden=200):
        super(Incept4, self).__init__()
        self.name = "Incept4"
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
        x1 = x[:,0,:,:].view(-1,1,14,14)
        
        x0 = F.relu(F.max_pool2d(self.conv1(x0), kernel_size=2, stride=2))
        x0 = F.relu(F.max_pool2d(self.conv2(x0), kernel_size=2, stride=2))
        x0 = x0.view(-1, 256)
        x0_aux = F.relu(self.fc1(x0))
        x0_aux = self.fc2(x0_aux)
        
        x1 = F.relu(F.max_pool2d(self.conv3(x1), kernel_size=2, stride=2))
        x1 = F.relu(F.max_pool2d(self.conv4(x1), kernel_size=2, stride=2))
        x1 = x1.view(-1, 256)
        x1_aux = F.relu(self.fc3(x1))
        x1_aux = self.fc4(x1_aux)
        
        x = torch.cat((x0, x1),1)
        x = F.relu(self.fc5(x.view(-1,512)))
        x = self.fc6(x)
        
        return x0_aux, x1_aux, x
    
    def return_new(self):
        return Incept4()
