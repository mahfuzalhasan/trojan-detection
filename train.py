import argparse
import os
import numpy as np
import math
from datetime import datetime
import time
import sys

from tensorboardX import SummaryWriter

import torchvision
import torchvision.transforms as transforms
import torchvision.models as models

import os
import argparse
from collections import OrderedDict

import torchvision.transforms as transforms
from torchvision.utils import save_image
from torchvision import datasets

from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch  

from dataParserCutPaste import DataParser
from datasetCutPaste import CutPasteDataset
import visualizer

import Config.configuration as cfg
import Config.parameters as params

#from resnet import ResNet18
#from model import Net
#from model_cutpaste import CutPasteNet
from model_simsiam_modified import SimSiam


device = torch.device('cuda:'+str(params.gpu))

# Training
def train(run_id, use_cuda, epoch, train_dataloader, model,
                                    optimizer, similarity_loss, writer):
    print('\n Training Epoch: %d' % epoch)
    global device
    model.train()
    correct = 0
    total = 0
    total_losses = []
    losses_1 = []
    losses_2 = []
    start_time = time.time()
    accuracy = []

    for param_group in optimizer.param_groups:
        print('Learning rate: ',param_group['lr'])

    for i, (img, aug_1, anom, class_labels) in enumerate(train_dataloader):

        if use_cuda:
            ########### While working with Single GPU
            #inputs, targets = inputs.to(device), targets.to(device)
            #similarity_loss.to(device)
            ################################

            ##############While working with Multiple GPU
            img, aug_1, anom = img.to(f'cuda:{model.device_ids[0]}'), aug_1.to(f'cuda:{model.device_ids[0]}'), anom.to(f'cuda:{model.device_ids[0]}')
            similarity_loss.to(f'cuda:{model.device_ids[0]}')
            ################

        optimizer.zero_grad()
        p_org, p1, p_anom, z_org, z1, z_anom = model(img, aug_1, anom)
        sim_1 = (similarity_loss(p_org, z1).mean() + similarity_loss(p1, z_org).mean()) * 0.5 # between different view
        #p3, p4, z3, z4 = model(img, anom)
        sim_2 = (similarity_loss(p_org, z_anom).mean() + similarity_loss(p_anom, z_org).mean()) * 0.5 # anomalous pair

        loss = -(sim_1 - 0.5 * sim_2)
        loss.backward()
        optimizer.step()

        """ predicted = torch.argmax(outputs, axis=1)
        acc_batch = torch.true_divide(torch.sum(predicted == targets), predicted.size(0)) """

        #accuracy.append(acc_batch.item())

        total_losses.append(loss.item())
        losses_1.append(sim_1.item())
        losses_2.append(sim_2.item())

        if i%10 == 0:
            #batches_done = epoch * len(dataloader) + i
            print(
                "[Epoch %d/%d] [Batch %d/%d] [loss: %f] [sim clean: %.3f] [sim anom: %.3f] "
                % (epoch, params.num_epoch, i, len(train_dataloader), np.mean(total_losses), np.mean(losses_1), np.mean(losses_2))
            )

            visuals = OrderedDict([('input_1', aug_1[0 : 64, :, :, :]),
                                   ('input_org', img[0 : 64, :, :, :]),
                                   ('anomaly', anom[0 : 64, :, :, :])])

            visualizer.write_img(visuals, run_id, epoch, i)

    save_dir = os.path.join(cfg.saved_models_dir, run_id)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if epoch % params.checkpoint == 0:
        save_file_path = os.path.join(save_dir, 'model_{}.pth'.format(epoch))
        states = {
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        torch.save(states, save_file_path)

    time_taken = time.time() - start_time
    print(f'Training Epoch {epoch}::: Total Loss:{np.mean(total_losses)} Clean Sample Sim:{np.mean(losses_1)} Anom Sample Sim:{np.mean(losses_2)} time taken:{time_taken}')
    sys.stdout.flush()

    writer.add_scalar('Training Total Loss', np.mean(total_losses), epoch)
    writer.add_scalar('Training Clean Sim', np.mean(losses_1), epoch)
    writer.add_scalar('Training Anomaly Sim', np.mean(losses_2), epoch)

    return model


def validation(run_id, use_cuda, epoch, valid_dataloader, model, similarity_loss, writer):
    print('\n Validation Epoch: %d' % epoch)
    global device
    model.eval()
    correct = 0
    total = 0
    total_losses = []
    losses_1 = []
    losses_2 = []
    start_time = time.time()
    accuracy = []

    """ for param_group in optimizer.param_groups:
        print('Learning rate: ',param_group['lr']) """
    with torch.no_grad():
        for i, (img, aug_1, anom, class_labels) in enumerate(valid_dataloader):

            if use_cuda:
                ########### While working with Single GPU
                #inputs, targets = inputs.to(device), targets.to(device)
                #similarity_loss.to(device)
                ################################

                ##############While working with Multiple GPU
                img, aug_1, anom = img.to(f'cuda:{model.device_ids[0]}'), aug_1.to(f'cuda:{model.device_ids[0]}'), anom.to(f'cuda:{model.device_ids[0]}')
                similarity_loss.to(f'cuda:{model.device_ids[0]}')
                ################

            
            p_org, p1, p_anom, z_org, z1, z_anom = model(img, aug_1, anom)
            sim_1 = (similarity_loss(p_org, z1).mean() + similarity_loss(p1, z_org).mean()) * 0.5 # between different view
            #p3, p4, z3, z4 = model(img, anom)
            sim_2 = (similarity_loss(p_org, z_anom).mean() + similarity_loss(p_anom, z_org).mean()) * 0.5 # anomalous pair

            loss = -(sim_1 - 0.5 * sim_2)


            total_losses.append(loss.item())
            losses_1.append(sim_1.item())
            losses_2.append(sim_2.item())

            if i%4 == 0:
                #batches_done = epoch * len(dataloader) + i
                print(
                    "[Epoch %d/%d] [Batch %d/%d] [loss: %f] [sim clean: %.3f] [sim anom: %.3f] "
                    % (epoch, params.num_epoch, i, len(valid_dataloader), np.mean(total_losses), np.mean(losses_1), np.mean(losses_2))
                )

                visuals = OrderedDict([('input_1', aug_1[0 : 64, :, :, :]),
                                    ('input_org', img[0 : 64, :, :, :]),
                                    ('anomaly', anom[0 : 64, :, :, :])])

                visualizer.write_img(visuals, run_id, epoch, i, val = True)


    time_taken = time.time() - start_time
    print(f'validation Epoch {epoch}::: Total Loss:{np.mean(total_losses)} Clean Sample Sim:{np.mean(losses_1)} Anom Sample Sim:{np.mean(losses_2)} time taken:{time_taken}')
    sys.stdout.flush()

    writer.add_scalar('Validation Total Loss', np.mean(total_losses), epoch)
    writer.add_scalar('Validation Clean Sim', np.mean(losses_1), epoch)
    writer.add_scalar('Validation Anomaly Sim', np.mean(losses_2), epoch)
    #return model

def load_model(model, optimizer):
    if cfg.saved_model_path is not None:
        state_dict_model = torch.load(cfg.saved_model_path)
        model.load_state_dict(state_dict_model['state_dict'])
        optimizer.load_state_dict(state_dict_model['optimizer'])
        print('model loaded')
        return model, optimizer


def train_task(run_id, use_cuda):
    global device
    writer = SummaryWriter(os.path.join(cfg.logs_dir, str(run_id)))

    #model = ResNet18( params.num_classes, params.num_channel)
    model = SimSiam(models.__dict__['resnet18'])
    if use_cuda:

        #######To load in Single GPU
        #model.to(device)
        #######

        #####To load in Multiple GPU. In FICS server we have 4 GPU. That's why 4 device ids here.
        model = nn.DataParallel(model, device_ids = [0, 2, 3])
        model.to(f'cuda:{model.device_ids[0]}')
        ######

    data_parser = DataParser(cfg.data_path, run_id)
    train_dataset = CutPasteDataset(data_parser.train_img_file, run_id)
    print("tr dataset: ",len(train_dataset))

    valid_dataset = CutPasteDataset(data_parser.valid_img_file, run_id, val=True)
    print("validation dataset: ",len(valid_dataset))
    #valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=params.batch_size, shuffle=False)
    #print("vl ld: ",len(valid_dataloader))

    similarity_loss = nn.CosineSimilarity(dim=1)
    optimizer = torch.optim.SGD(model.parameters(), lr=params.learning_rate, momentum=0.9, weight_decay=5e-4)

    #model, optimizer = load_model(model, optimizer)

    for epoch in range(params.num_epoch):
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=params.batch_size, shuffle=True)
        model = train(run_id, use_cuda, epoch, train_dataloader, model, optimizer, similarity_loss, writer)
        valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=params.batch_size, shuffle=False)
        validation(run_id, use_cuda, epoch, valid_dataloader, model, similarity_loss, writer)


if __name__=="__main__":
    run_started = datetime.today().strftime('%m-%d-%y_%H%M')
    print("run id: ",run_started)
    use_cuda = torch.cuda.is_available()
    print('use cuda: ', use_cuda)
    train_task(run_started, use_cuda)
