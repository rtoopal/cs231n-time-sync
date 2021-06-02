import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from IPython.display import Image
import numpy as np
import torch
import torch.nn as nn
from torch.nn import init
import torchvision
import torchvision.transforms as T
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import sampler
import torchvision.datasets as dset
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cv2



class Synchronizer(nn.Module):
    def __init__(self, num_keypoints, frames):
        super().__init__()

        self.M = frames

        self.conv1_1 = nn.Conv2d(num_keypoints, 32, (3, 3), padding=1)
        nn.init.kaiming_normal_(self.conv1_1.weight)
        self.conv1_2 = nn.Conv2d(32, 48, (3, 3), padding=1)
        nn.init.kaiming_normal_(self.conv1_2.weight)
        self.conv1_3 = nn.Conv2d(48, 64, (3, 3), padding=1)
        nn.init.kaiming_normal_(self.conv1_3.weight)
        self.conv1_4 = nn.Conv2d(64, 128, (3, 3), padding=1)
        nn.init.kaiming_normal_(self.conv1_4.weight)
        self.conv1_5 = nn.Conv2d(128, 256, (3, 3), padding=1)
        nn.init.kaiming_normal_(self.conv1_4.weight)
        self.fc1_1 = nn.Linear(4096, 256)
        nn.init.kaiming_normal_(self.fc1_1.weight)

        self.hidden1 = torch.randn(1, 1, 256)
        self.c1 = torch.randn(1, 1, 256)
        self.hidden2 = torch.clone(self.hidden1)
        self.c2 = torch.clone(self.c1)


        self.embed_lstm = nn.LSTM(256, 256)

        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        self.softmax = nn.Softmax(dim=1)
        self.ce_loss = nn.CrossEntropyLoss()




    def forward(self, vid1, vid2):
        """
        Inputs:
            - vid1: heatmap volume for video 1 with dims (T, 18, H, W) where T is the video length
            - vid2: heatmap volume for video 2 with dims (T, 18, H, W)
        """
        T = vid1.shape[0]
        
        scores = None
        # We've now created 256-dimensional vector representations of each frame (T, 256)
        #print("embedding video 1")
        out1 = self.embed(vid1)
        #print("embedding video 2")
        out2 = self.embed(vid2)
        
        # refactor embedding with LSTM layer 
        lstm_out1, hidden1 = self.embed_lstm(out1, (self.hidden1, self.c1))
        lstm_out1 = lstm_out1.view(T, -1)
        
        lstm_out2 , hidden2 = self.embed_lstm(out2, (self.hidden2, self.c2))
        lstm_out2 = lstm_out2.view(T, -1)
        #print("finished LSTM layers")
        """
        # Create list of costs and min cost and corresponding offset
        costs, min_cost, min_offset = self.concatOffsets(lstm_out1, lstm_out2)
        #print(costs, min_cost, min_offset)
        
        # concatenate 2 videos and fully connected layers convert to single offset value
        scores = self.softmax(costs)
        print("scores: ", scores)
        if min_offset < 0:
          print("second vid starts ", -min_offset, " frames after the first")
        elif min_offset > 0:
          print("first vid starts ", min_offset, " frames after the second")
        else:
          print("videos are aligned")
        """

        return lstm_out1, lstm_out2
        #return scores


    def loss(self, scores, labels):
        true_offset = torch.argmax(labels) - self.M
        pred_offset = torch.argmax(scores) - self.M
        diff = torch.abs(true_offset - pred_offset)
        a = self.ce_loss(scores, labels)
        b = diff/(2*self.M+1)
        return a + b

    def embed(self, vid):
        #print("embedding input: ", vid.size())
        T = vid.size()[0]
        out = self.conv1_1(vid)

        out = nn.functional.relu(out)
        out = nn.MaxPool2d((2,2),stride = 2)(out)
        #print("conv2 out: ",out.size())

        out = self.conv1_2(out)
        out = nn.functional.relu(out)
        out = nn.MaxPool2d((2,2),stride = 2)(out)
        #print("conv3 out: ", out.size())

        out = self.conv1_3(out)
        out = nn.functional.relu(out)
        out = nn.MaxPool2d((2,2),stride = 2)(out)
        #print("conv3 out: ",out.size())


        out = self.conv1_4(out)
        out = nn.functional.relu(out)
        out = nn.MaxPool2d((2,2),stride = 2)(out)
        #print("conv4 out: ",out.size())

        out = self.conv1_5(out)
        out = nn.functional.relu(out)
        out = nn.MaxPool2d((2,2),stride = 2)(out)
        #print("conv5 out: ",out.size())

        out = out.view(T, 1, -1)
        #print("conv out: ",out.size())
        
        out = self.fc1_1(out)
        #print("embedding shape: ",out.size())
        return out
