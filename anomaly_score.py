
# Torch Library -----------
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

# Torchvision Library ---------
import torchvision


# python library -------------
import os
import random
import numpy as np


# 3rd party library -------------
import cv2
import pickle

# Internal imports --------------
from utils import imgResize, convert_to_tensor
from model_simsiam_modified import SimSiam
import Config.configuration as cfg
import Config.parameters as params


class Resnet18(torch.nn.Module):
    def __init__(self, model, output_layer):
        super(Resnet18, self).__init__()
        # Get a resnet50 backbone
        # Extract 4 main layers (note: MaskRCNN needs this particular name
        # mapping for return nodes)
        
        self.output_layer = output_layer
        self.children_list = []
        for n,c in model.module.encoder.named_children():
            self.children_list.append(c)
            print('layer: ', n)
            if n == self.output_layer:
                break
        self.net = nn.Sequential(*self.children_list)
        self.net = nn.DataParallel(self.net, device_ids = [1, 2, 3])
        self.net.to(f'cuda:{self.net.device_ids[0]}')
        #self.net.load_state_dict()
        self.net.eval()
        #print(self.net)


    def forward(self, x):
        x = self.net(x)
        #x = self.fpn(x)
        return x



class TestDataset(Dataset):
    def __init__(self, img_files, train_img = False, tool_created_img = False):
        super(TestDataset,self).__init__()
        self.img_files = img_files
        self.train_img = train_img
        self.tool_created_img = tool_created_img
        random.shuffle(self.img_files)

    def __getitem__(self, index):
        img_path =  self.img_files[index%len(self.img_files)]
        #print('img path: ', img_path)
        #img_path = image_container['path']
        img = cv2.imread(img_path)
        if self.train_img or self.tool_created_img:
            img = imgResize(img)
        img = img[:,:,0]
        image = convert_to_tensor(img)
        return img_path, image

    def __len__(self):
        return len(self.img_files)


class TestParser(object):
    def __init__(self, data_path):
        super(TestParser, self).__init__()
        self.data_path = cfg.test_data_path

        self.orig_img_files = self._load_test_data("original")
        #self.augmented_img_files = self._load_test_data("augmented")
        self.anomalous_img_files = self._load_test_data("anomaly")


    def _load_test_data(self, img_type):
        img_files = []
        path = os.path.join(self.data_path, img_type)
        for img_name in os.listdir(path):
            img_path = os.path.join(path, img_name)
            img_files.append(img_path)
        return img_files
    
    """ def _load_test_data(self, img_type):
        img_files = []
        anom = False
        if img_type == "anomaly":
            anom = True

        for folder_id in os.listdir(self.data_path):
            for dwelling_time in os.listdir(os.path.join(self.data_path, folder_id)):
                image_dir_path = os.path.join(self.data_path, folder_id, dwelling_time)
                for img_name in os.listdir(image_dir_path):
                    if not anom:
                        if '_n' in img_name:
                            continue
                    else:
                        if '_n' not in img_name:
                            continue
                    img_path = os.path.join(image_dir_path, img_name)
                    img_files.append(img_path)
        return img_files """



class Test():
    def __init__(self, test_data_path, saved_model_path, use_cuda = True):
        super(Test, self).__init__()
        self.saved_model_path = saved_model_path
        self.use_cuda = use_cuda
        self.data_parser = TestParser(test_data_path)
        self.origImgDataset = TestDataset(self.data_parser.orig_img_files, tool_created_img=params.tool_created_img)
        self.anomImgDataset = TestDataset(self.data_parser.anomalous_img_files, tool_created_img=params.tool_created_img)

        self.model = self._init_model()
        print("loading from: ", self.saved_model_path[self.saved_model_path.rindex('/')-13 :])
        self.model.load_state_dict(torch.load(self.saved_model_path)['state_dict'])
        self.model.eval()
        self.featureExtractor = Resnet18(self.model, 'layer2')
        """ print(self.model.module.encoder.conv1.weight)
        print(self.model.module.encoder.conv1.bias)
        print("feature extractor weight")
        print(self.featureExtractor.net.module[0].weight)
        print(self.featureExtractor.net.module[0].bias) """
        #exit()
        
    def mean_train_feature(self):
        train_img_file  = pickle.load(open(cfg.train_data_path, 'rb'))
        img_files = [d['path'] for d in train_img_file]
        dataset = TestDataset(img_files, train_img=True)
        dataloader = DataLoader(dataset, batch_size=512, shuffle=True)
        print('dataloader size: ',len(dataloader))

        output_holder = []

        with torch.no_grad():
            for i, (img_path, images) in enumerate(dataloader):
                if self.use_cuda:
                    images = images.to(f'cuda:{self.model.device_ids[0]}')
                #print("input: ", image.size())
                outputs = self.featureExtractor(images)
                output_holder.append(outputs)
                #print("outputs: ", outputs.size())

        #print("length output holder: ",len(output_holder))
        mean_feature = torch.mean(torch.cat(output_holder, dim=0), dim=0)
        print('mean train feature : ',mean_feature.size())
        if not os.path.exists(cfg.mean_train_feature_path):
            os.makedirs(cfg.mean_train_feature_path)
        torch.save(mean_feature, os.path.join(cfg.mean_train_feature_path, 'mean_feature.pt'))
        #print('mean feature 1: ',mean_feature_1.size())
        #mean_feature_2 = torch.mean(torch.stack(output_holder), dim=0)
        #print('mean feature 2: ',mean_feature_1.size())
        #mean_feature = torch.mean(output_holder)
        



    def test_anomaly(self):
        types = ['original', 'anomaly']
        mean_feature = torch.load(os.path.join(cfg.mean_train_feature_path, 'mean_feature.pt'))
        print('mean train feature: ',mean_feature.size())
        output_mapper = {}
        similarity = nn.CosineSimilarity(dim=-1)
        for d_set in types:
            print('type: ',d_set)
            if d_set == 'original':
                dataset = self.origImgDataset
            else:
                print('select anomalous')
                dataset = self.anomImgDataset

            print("test dataset: ", len(dataset))
            test_dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
            error = []
            

            with torch.no_grad():
                for i, (img_path, image) in enumerate(test_dataloader):
                    if self.use_cuda:
                        image = image.to(f'cuda:{self.featureExtractor.net.device_ids[0]}')
                    #print("input: ", image.size()) 
                    output = self.featureExtractor(image)
                    output = torch.squeeze(output)
                    #print("output: ", output.size())
                    #distance = similarity(mean_feature, output)
                    distance = torch.nn.functional.mse_loss(mean_feature, output)
                    #print('distance: ', distance.size())
                    #exit()
                    output_mapper[img_path] = distance
                    error.append(distance)
            mean_error = torch.mean(torch.stack(error))
            print('mean distance for ', d_set,' ', mean_error)
        print('##output Mapper:##')
        print(output_mapper)



    def _init_model(self):
        model = SimSiam(torchvision.models.__dict__['resnet18'])
        if self.use_cuda:
            #######To load in Single GPU
            #model.to(device)
            #######
            #####To load in Multiple GPU. In FICS server we have 4 GPU. That's why 4 device ids here.
            model = nn.DataParallel(model, device_ids = [1, 2, 3])
            model.to(f'cuda:{model.device_ids[0]}')
            ######
        return model


if __name__=='__main__':
    use_cuda = torch.cuda.is_available()
    testTask = Test(cfg.test_data_path, cfg.saved_model_path, use_cuda)
    testTask.mean_train_feature()
    testTask.test_anomaly()