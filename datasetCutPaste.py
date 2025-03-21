import glob
import random
import os
import numpy as np
import cv2
import xml.etree.ElementTree as ET
from PIL import Image
from datetime import datetime
import copy

import torch
#Torch Library
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch  

#TorchVision Library
import torchvision
import torchvision.transforms as transforms
from torchvision.utils import save_image


#Python Libraries
from collections import defaultdict



#System Libraries
import sys
import copy


#Project Files
from dataParserCutPaste import DataParser
from cutpaste import CutPaste
from utils import imgResize, convert_to_tensor
from augmentation import Augmentation


#sys.path.insert(1,"/home/UFAD/mdmahfuzalhasan/Documents/Projects/HARTS/Classification/Config")
from Config import configuration as cfg
from Config import parameters as params




class CutPasteDataset(Dataset):
    def __init__(self, img_files, run_id, val=False, cutpaste_type = '3way'):
        super(CutPasteDataset,self).__init__()
        self.img_files = img_files
        self.run_id = run_id
        self.val = val 
        self.transform = CutPaste(type=cutpaste_type)
        self.global_augmentation = Augmentation()
        random.shuffle(self.img_files)


    def sizeFix(self,img):
        img = np.array(img)
        img_resize = imgResize(img)
        #img_resize = img_resize[:,:,0]
        return img_resize

    def convert_to_tensor(self, img):
        #img = np.array(img) 
        img = img[:, :, 0]
        img = np.expand_dims(img, axis=2)
        img = img/255.0
        img_torch = torch.from_numpy(img)
        img_torch = img_torch.type(torch.FloatTensor)
        img_torch = img_torch.permute(-1, 0, 1)
        return img_torch


    def __getitem__(self, index):
        image_container =  self.img_files[index%len(self.img_files)]
        img_path = image_container['path']
        label = image_container['label']

        img = cv2.imread(img_path)

        pil_image = Image.fromarray(img)
        image, cp = self.transform(pil_image)
        
        #pre-processing----------------
        image = self.sizeFix(image)
        cp = self.sizeFix(cp)

        aug_img_1 = self.global_augmentation.generation(copy.deepcopy(image), params.isReduced)
        #aug_img_2 = self.global_augmentation.generation(copy.deepcopy(image), params.isReduced)
        #------------------------------

        aug_img_1 = self.convert_to_tensor(aug_img_1)
        #aug_img_2 = self.convert_to_tensor(aug_img_2)
        cp = self.convert_to_tensor(cp)
        image = self.convert_to_tensor(image)

        """ save_path = os.path.join("./output_check", self.run_id)
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        input_img_path = '%s/input_%05d.jpg' % (save_path, index)
        cp_path = '%s/cp_%05d.jpg' % (save_path, index)
        #scar = '%s/scar_%05d.jpg' % (path, iteration)

        torchvision.utils.save_image(image, input_img_path, normalize=True, nrow=1, range=(0, 1))
        torchvision.utils.save_image(cp, cp_path, normalize=True, nrow=1, range=(0, 1)) """
        #-----------------------

        label = np.asarray(label)
        class_label = torch.from_numpy(label)
        #print('label: ',label)

        #exit()

        return image, aug_img_1, cp, class_label 

    def __len__(self):
        return len(self.img_files)

if __name__ == "__main__":
    run_started = datetime.today().strftime('%m-%d-%y_%H%M')
    print('run id: ',run_started)
    parser = DataParser(cfg.data_path, run_started)
    #exit()

    train_dataset = CutPasteDataset(parser.train_img_file, run_started)
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    print("Train dataset length: ",len(train_dataset))
    print('Train dataloader: ',len(train_dataloader))
    

    valid_dataset = CutPasteDataset(parser.valid_img_file, run_started)
    valid_dataloader = DataLoader(valid_dataset, batch_size=1, shuffle=True)
    print("Valid dataset length: ",len(valid_dataset))
    print('Valid dataloader: ',len(valid_dataloader))

    for i, (batch, class_labels) in enumerate(train_dataloader):
        x = torch.cat(batch, axis=0)
        y = torch.arange(len(batch))
        y = y.repeat_interleave(len(batch[0]))
        #print(x.size())
        #print(y)
        #print(cfg.output_check)
        #save_image(inputs, os.path.join(cfg.output_check,str(i)+'.png'),range=(0,1),nrow=8)
        print('class_labels: ',class_labels)
        if i==100:
            exit()
