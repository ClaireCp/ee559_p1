import torch
from torch import nn
from torch.nn import functional as F

class Incept1(nn.Module):
    """ Base (most simple) module - Uses auxiliary losses AND weight sharing """
    def __init__(self, nb_classes_digits=10, nb_classes_pairs=1, nb_hidden=200):
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
    
class Incept2(nn.Module):
    """ Increase the number of channels produced by the convolutions """
    def __init__(self, nb_classes_digits=10, nb_classes_pairs=1, nb_hidden=200):
        super(Incept2, self).__init__()
        self.name = "Incept2"
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3)
        self.fc1 = nn.Linear(512, nb_hidden)
        self.fc2 = nn.Linear(nb_hidden, nb_classes_digits)       
        self.fc3 = nn.Linear(1024, nb_hidden)
        self.fc4 = nn.Linear(nb_hidden, nb_classes_pairs)

    def forward(self, x):
        x0 = x[:,0,:,:].view(-1,1,14,14)
        x1 = x[:,0,:,:].view(-1,1,14,14)
        
        x0 = F.relu(F.max_pool2d(self.conv1(x0), kernel_size=2, stride=2))
        x0 = F.relu(F.max_pool2d(self.conv2(x0), kernel_size=2, stride=2))
        x0 = x0.view(-1, 512)
        x0_aux = F.relu(self.fc1(x0))
        x0_aux = self.fc2(x0_aux)
        
        x1 = F.relu(F.max_pool2d(self.conv1(x1), kernel_size=2, stride=2))
        x1 = F.relu(F.max_pool2d(self.conv2(x1), kernel_size=2, stride=2))
        x1 = x1.view(-1, 512)
        x1_aux = F.relu(self.fc1(x1))
        x1_aux = self.fc2(x1_aux)
        
        x = torch.cat((x0, x1),1)
        x = F.relu(self.fc3(x.view(-1,1024)))
        x = self.fc4(x)
        
        return x0_aux, x1_aux, x
    
    def return_new(self):
        return Incept2()
    
class Incept3(nn.Module):
    """ Add regularization using Dropout (zero out units)  and Dropout2d (zero out entire channels), also increase processing after early escape (early classifier) """
    def __init__(self, nb_classes_digits=10, nb_classes_pairs=1, nb_hidden=100):
        super(Incept3, self).__init__()
        self.name = "Incept3"
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.fc1 = nn.Linear(256, nb_hidden)
        self.fc2 = nn.Linear(nb_hidden, nb_classes_digits)
        self.conv3 = nn.Conv2d(1, 6, kernel_size=5, stride=4)
        self.fc3 = nn.Linear(186, nb_hidden)
        self.fc4 = nn.Linear(nb_hidden, nb_classes_pairs)
        
        self.conv2d_drop = nn.Dropout2d(p=0.3)
        self.drop1 = nn.Dropout(p=0.25)
        self.drop2 = nn.Dropout(p=0.25)

    def forward(self, x):
        x0 = x[:,0,:,:].view(-1,1,14,14) # torch.Size([1000, 1, 14, 14])
        x1 = x[:,0,:,:].view(-1,1,14,14)
        
        # self.conv1(x0).shape = torch.Size([1000, 32, 12, 12])
        x0 = F.relu(F.max_pool2d(self.conv2d_drop(self.conv1(x0)), kernel_size=2, stride=2)) # x0.shape = torch.Size([1000, 32, 6, 6])
        # self.conv2(x0).shape = torch.Size([1000, 64, 4, 4])
        x0 = F.relu(F.max_pool2d(self.conv2d_drop(self.conv2(x0)), kernel_size=2, stride=2)) 
        # x0.shape = torch.Size([1000, 64, 2, 2])
        x0 = x0.view(-1, 256) # x0.shape = torch.Size([1000, 256])
        x0_aux = F.relu(self.drop1(self.fc1(x0))) # x0_aux.shape = torch.Size([1000, 100])
        x0_aux = self.fc2(x0_aux) # x0_aux.shape = torch.Size([1000, 10])
        
        x1 = F.relu(F.max_pool2d(self.conv2d_drop(self.conv1(x1)), kernel_size=2, stride=2))
        x1 = F.relu(F.max_pool2d(self.conv2d_drop(self.conv2(x1)), kernel_size=2, stride=2))
        x1 = x1.view(-1, 256)
        x1_aux = F.relu(self.drop1(self.fc1(x1)))
        x1_aux = self.fc2(x1_aux)
        
        # add a channel dimension and concatenate across width
        x = torch.cat((x0.view(-1, 1, 1, 256), x1.view(-1, 1, 1, 256)),2) # x.shape = torch.Size([1000, 1, 2, 256])
        # self.conv3(x).shape = torch.Size([1000, 6, 1, 63])
        x = F.relu(F.max_pool2d(self.conv2d_drop(self.conv3(x)), kernel_size=2, stride=2)) # x.shape = torch.Size([1000, 6, 1, 31])
        x = x.view(-1,186)
        x = F.relu(self.drop2(self.fc3(x)))
        x = self.fc4(x)
        
        return x0_aux, x1_aux, x
    
    def return_new(self):
        return Incept3()
    
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

class AuxModel(nn.Module):

    def __init__(self, nb_hidden=100):
        super(AuxModel, self).__init__()
        self.name = "AuxModel"
        self.cl1 = nn.Conv2d(1, 64, kernel_size=3)
        self.cl2 = nn.Conv2d(64, 128, kernel_size=3)
        self.full1 = nn.Linear(512, nb_hidden)
        self.full2 = nn.Linear(nb_hidden,40)
        self.full3 = nn.Linear(40,10)
        self.full4 = nn.Linear(20, 1)
 
    def forward(self, x):
        a = x[:,0,:,:].view(-1,1,14,14)
        b = x[:,1,:,:].view(-1,1,14,14)

        a = F.relu(F.max_pool2d(self.cl1(a), kernel_size=2, stride=2))
        b = F.relu(F.max_pool2d(self.cl1(b), kernel_size=2, stride=2))
        a = F.relu(F.max_pool2d(self.cl2(a), kernel_size=2, stride=2))
        b = F.relu(F.max_pool2d(self.cl2(b), kernel_size=2, stride=2))
        a = F.relu(self.full1(a.view(-1, 512)))
        b = F.relu(self.full1(b.view(-1, 512)))
        a = F.relu(self.full2(a))
        b = F.relu(self.full2(b))
        channel1 = F.relu(self.full3(a))
        channel2 = F.relu(self.full3(b))
        
 
        output = torch.cat((channel1,channel2),1)
        output = self.full4(output)

        return channel1, channel2, output
    def return_new(self):
        return AuxModel()