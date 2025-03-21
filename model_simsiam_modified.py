# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torchvision.models as models


class Identity(torch.nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

class SimSiam(nn.Module):
    def __init__(self, base_encoder, dim=64, pred_dim=128):
        """
        dim: feature dimension (default: 2048)
        pred_dim: hidden dimension of the predictor (default: 512)
        """
        super(SimSiam, self).__init__()

        # create the encoder
        # num_classes is the output fc dimension, zero-initialize last BNs
        self.encoder = base_encoder(num_classes=dim, zero_init_residual=True)
        self.encoder.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.encoder.layer3 = Identity()
        self.encoder.layer4 = Identity()

        # build a 3-layer projector
        prev_dim = 128 #self.encoder.fc.weight.shape[1]
        #print("prev_dim: ", prev_dim)
        self.encoder.fc = nn.Sequential(nn.Linear(prev_dim, prev_dim, bias=False),
                                        nn.BatchNorm1d(prev_dim),
                                        nn.ReLU(inplace=True), # first layer
                                        nn.Linear(prev_dim, prev_dim, bias=False),
                                        nn.BatchNorm1d(prev_dim),
                                        nn.ReLU(inplace=True), # second layer
                                        nn.Linear(prev_dim, dim, bias=True),
                                        nn.BatchNorm1d(dim, affine=False)) # output layer
        self.encoder.fc[6].bias.requires_grad = False # hack: not use bias as it is followed by BN

        # build a 2-layer predictor
        self.predictor = nn.Sequential(nn.Linear(dim, pred_dim, bias=False),
                                        nn.BatchNorm1d(pred_dim),
                                        nn.ReLU(inplace=True), # hidden layer
                                        nn.Linear(pred_dim, dim)) # output layer

        """ # build a 2-layer predictor
        self.predictor_2 = nn.Sequential(nn.Linear(dim, pred_dim, bias=False),
                                        nn.BatchNorm1d(pred_dim),
                                        nn.ReLU(inplace=True), # hidden layer
                                        nn.Linear(pred_dim, dim)) # output layer """


    
        
    def forward(self, orig, x1, anom):
        # compute features for one view
        z_org = self.encoder(orig)
        print(z_org.shape)
        z1 = self.encoder(x1) # NxC
        #z2 = self.encoder(x2) # NxC
        z_anom = self.encoder(anom)

        p_org = self.predictor(z_org)
        print(p_org.shape)
        p1 = self.predictor(z1) # NxC
        #p2 = self.predictor(z2) # NxC
        p_anom = self.predictor(z_anom)
        return p_org, p1, p_anom, z_org.detach(), z1.detach(), z_anom.detach()

if __name__=="__main__":
    simSiam  = SimSiam(models.__dict__['resnet18'])
    print(simSiam)
    x = torch.rand((32,1,96,96))     #batch size = 1
    out = simSiam(x, x, x)
