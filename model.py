import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, num_classes, num_channel):
        super().__init__()
        self.conv1 = nn.Conv2d(num_channel, 8, 3)
        self.conv2 = nn.Conv2d(8, 16, 3)
        self.conv2_bn = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(16, 32, 3)
        self.conv3_bn = nn.BatchNorm2d(32)
        self.conv4 = nn.Conv2d(32, 64, 3)
        self.conv4_bn = nn.BatchNorm2d(64)
        self.conv5 = nn.Conv2d(64, 64, 3)
        self.conv5_bn = nn.BatchNorm2d(64)
        self.conv6 = nn.Conv2d(64, 128, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.linear = nn.Linear(128*4*4, num_classes)
        


    def forward(self, x):
        x = F.relu(self.conv1(x))
        #print('after conv1: ',x.size())
        
        '''
        x = self.pool(x)
        #print('after pool1: ',x.size())
        '''
        x = F.relu(self.conv2_bn(self.conv2(x)))
        #print('after conv2: ',x.size())

        x = self.pool(x)
        #print('after pool1: ',x.size())

        x = F.relu(self.conv3_bn(self.conv3(x)))
        #print('after conv3: ',x.size())
        

        x = F.relu(self.conv4_bn(self.conv4(x)))
        #print('after conv4: ',x.size())

        x = self.pool(x)
        #print('after pool2: ',x.size())

        x = F.relu(self.conv5_bn(self.conv5(x)))
        #print('after conv5: ',x.size())

        x = F.relu(self.conv6(x))
        #print('after conv6: ',x.size())

        x = F.avg_pool2d(x, (4,4))
        #print('after avgpool: ',x.size())

        x = x.view(x.size(0),-1)
        x = self.linear(x)

        #print('out: ',x.size())

        return x


if __name__ =="__main__":
    net = Net(3,1)
    y = net(torch.randn(1,1,96,96))