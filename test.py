# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 22:34:31 2018

@author: youngjoong
"""


# pytorch mnist cnn + lstm

from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.model_zoo as model_zoo
from torchvision import datasets, transforms
from torch.autograd import Variable
import numpy as np
from PIL import Image
from torch.utils.data.dataset import Dataset


ROOTPATH="D:/Workspace/1123/Test/"
imagesize = 256
training_batch_size = 16
validating_testing_batch_size = 1

class FabricTestingDataset(Dataset):
    def __init__(self):
        """
        Args:
            
        """
        
        # Path to video frames
        self.image_folder_path = ROOTPATH+"MIT Fabric Dataset Processed/"
        # Transforms
        self.to_tensor = transforms.ToTensor()
        self.to_square = transforms.Resize((imagesize,imagesize))
        # Read the (videoindex,label) file
        gt = []
        f = open(ROOTPATH+"MIT Fabric Dataset Index/testing_dataset.txt", 'r')
        lines = f.readlines()
        for line in lines:
            index, label = line.split(",")
            label=label[0:-1]
            gt.append((int(index), int(label)))
        f.close()
        self.videoindex_label = gt
        self.data_len = len(gt)
    

    def __getitem__(self, index):
        # Get video index and label
        videoindex, label = self.videoindex_label[index]
        
        # how to save rgb t image to np array ?
        # need to also make validation dataset class and testing dataset class
        # dataloader also
        
        # initialize tensor of timestep x channel x H x W
        data = torch.empty(15, 3, imagesize, imagesize, dtype=torch.double)
        
        for i in range(15):
            
            # Open image
            img_as_img = Image.open(self.image_folder_path+str(videoindex)+"_"+str(i)+".jpg")
            resized_img = self.to_square(img_as_img)
            img_as_nparr = np.array(resized_img)
            
            # Normalize image
            r_min = np.amin(img_as_nparr[0])
            r_max = np.amax(img_as_nparr[0])
            
            g_min = np.amin(img_as_nparr[1])
            g_max = np.amax(img_as_nparr[1])
            
            b_min = np.amin(img_as_nparr[2])
            b_max = np.amax(img_as_nparr[2])
            
            img_as_nparr[0]=((img_as_nparr[0]-r_min)/(r_max-r_min))*2-1
            img_as_nparr[1]=((img_as_nparr[1]-g_min)/(g_max-g_min))*2-1
            img_as_nparr[2]=((img_as_nparr[2]-b_min)/(b_max-b_min))*2-1
            
            # Transform image to tensor
            img_as_tensor = self.to_tensor(img_as_nparr)
            #print(img_as_tensor.size())
            data[i] = img_as_tensor
      

        return data, label

    def __len__(self):
        return self.data_len

testing_dataset = FabricTestingDataset()
testing_loader = torch.utils.data.DataLoader(testing_dataset, batch_size=validating_testing_batch_size, shuffle=True, num_workers=0)


# Training settings
# parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
# parser.add_argument('--batch-size', type=int, default=64, metavar='N',
#                     help='input batch size for training (default: 64)')
# parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
#                     help='input batch size for testing (default: 1000)')
# parser.add_argument('--epochs', type=int, default=10, metavar='N',
#                     help='number of epochs to train (default: 10)')
# parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
#                     help='learning rate (default: 0.01)')
# parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
#                     help='SGD momentum (default: 0.5)')
# parser.add_argument('--no-cuda', action='store_true', default=False,
#                     help='disables CUDA training')
# parser.add_argument('--seed', type=int, default=1, metavar='S',
#                     help='random seed (default: 1)')
# parser.add_argument('--log-interval', type=int, default=10, metavar='N',
#                     help='how many batches to wait before logging training status')
# args = parser.parse_args()

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        
        # preprocessing layers
        # try first with 1 channel input 
        #self.pre_conv1 = nn.Conv2d(1, 32, kernel_size=7, stride=1, padding=3)
        #self.pre_maxpool1 = nn.MaxPool2d(kernel_size=2, stride=4)
        #self.pre_conv2 = nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=2)
        #self.pre_maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # end of preprocesing
        
        
        # start of ResNet
        # modified input channel number from 3 (RGB)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        
        # preprocessing layers
        #x = self.pre_conv1(x)
        #x = self.pre_maxpool1(x)
        #x = self.pre_conv2(x)
        #x = self.pre_maxpool2(x)
        
        
        # end of preprocesing
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(-1, 512) # T x L, L = 512
        #x = x.view(x.size(0), -1)
        #x = self.fc(x)
        #x = x.view(-1, 1000)
        return x


def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model


def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    return model


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model


def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    return model


def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    return model

class Args:
    def __init__(self):
        self.cuda = True
        self.no_cuda = False
        self.seed = 1
        self.batch_size = training_batch_size
        self.test_batch_size = validating_testing_batch_size
        self.epochs = 20
        self.lr = 0.001
        self.momentum = 0.5
        self.log_interval = 10


args = Args()

args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

kwargs = {'num_workers': 0, 'pin_memory': True} if args.cuda else {}

#train_loader = torch.utils.data.DataLoader(
#    datasets.MNIST(
#        '../data',
#        train=True,
#        download=True,
#        transform=transforms.Compose([
#            transforms.ToTensor(),
#            transforms.Normalize((0.1307, ), (0.3081, ))
#        ])),
#    batch_size=args.batch_size,
#    shuffle=True,
#    **kwargs)

#test_loader = torch.utils.data.DataLoader(
#    datasets.MNIST(
#        '../data',
#        train=False,
#        transform=transforms.Compose([
#            transforms.ToTensor(),
#            transforms.Normalize((0.1307, ), (0.3081, ))
#        ])),
#    batch_size=args.test_batch_size,
#    shuffle=True,
#    **kwargs)



class Combine(nn.Module):
    def __init__(self):
        super(Combine, self).__init__()
        
        
        # modify to use resnet-18 for CNN
        #self.cnn = CNN()
        self.cnn = resnet18()
        self.rnn = nn.LSTM(
            input_size=512, 
            hidden_size=512, 
            num_layers=2,
            batch_first=True)
        # modify to classify 30 classes
        #self.linear = nn.Linear(64,10)
        self.linear = nn.Linear(512,30)
        
    def forward(self, x):
       
        batch_size, timesteps, C, H, W = x.size()
        c_in = x.view(batch_size * timesteps, C, H, W)
        c_out = self.cnn(c_in)
        #print(c_out.size())
        r_in = c_out.view(batch_size,timesteps, -1)
        #print(r_in.size())
        r_out, (h_n, h_c) = self.rnn(r_in)
        #print(r_out.size())
        #print(r_out[:, -1, :].size())
        
        r_out2 = self.linear(r_out[:, -1, :])
        
        return F.log_softmax(r_out2, dim=1)


model = Combine()
if args.cuda:
    model.cuda()

optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
def test():
    model.eval()
    test_loss = 0
    correct = 0
    #for data, target in test_loader:
    i=0
    for data, target in testing_loader:  
        print("processing data"+str(i+1))
        #data = np.expand_dims(data, axis=1)
        #data = torch.FloatTensor(data)
        data = data.float()
        #print(target.size)
        
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        test_loss += F.nll_loss(
            output, target, size_average=False).data[0]  # sum up batch loss
        pred = output.data.max(
            1, keepdim=True)[1]  # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).long().cpu().sum()
        i+=1
    test_loss /= len(testing_loader.dataset)
    print(
        '\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(testing_loader.dataset),
            100. * correct / len(testing_loader.dataset)))
    f = open(ROOTPATH+"MIT Fabric Dataset Testing Loss Accuracy/"+"performance.txt",'a')
    f.write('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(testing_loader.dataset),
            100. * correct / len(testing_loader.dataset)))
    f.close()

model.load_state_dict(torch.load(ROOTPATH+"MIT Fabric Models/test0_after_3.pt"))
test()
   
