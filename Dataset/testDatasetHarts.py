import glob
import random
import os
import numpy as np
import cv2
import xml.etree.ElementTree as ET
from PIL import Image
from datetime import datetime

import torch
#Torch Library
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch  

#TorchVision Library
import torchvision.transforms as transforms
from torchvision.utils import save_image


#Python Libraries
from collections import defaultdict


#System Libraries
import sys
import copy


#Project Files
#from dataParserHarts import HARTS_parser
sys.path.append('./Dataset')

#sys.path.insert(1,"/home/UFAD/mdmahfuzalhasan/Documents/Projects/HARTS/Classification/Config")
from Config import configuration as cfg
from Config import parameters as params





class HARTS_dataset(Dataset):
    def __init__(self, img_files):
        super(HARTS_dataset,self).__init__()
        self.img_files = img_files
        #random.shuffle(self.img_files)


    def imgResize(self,img):
        h = img.shape[0]
        w = img.shape[1]
        color = (0,0,0)
        new_h = 90
        new_w = 90

        '''
        if h<w:
            new_h = int((params.img_height/params.img_width) * w)
            new_w = w

        else:
            if w/h < params.img_width/params.img_height:
                new_w = int((params.img_width/params.img_height)*h)
                new_h = h

            elif w/h > params.img_width/params.img_height:
                new_h = int((params.img_height/params.img_width)*w)
                new_w = w
        '''
        result = np.full((new_h,new_w,3), color, dtype=np.uint8)
        # compute center offset
        if h > new_h or w > new_w:
            img = cv2.resize(img,(new_w,new_h))
            result[:,:] = img
            return result
        else:
            xx = (new_w - w) // 2
            yy = (new_h - h) // 2
            result[yy:yy+h, xx:xx+w] = img

            return result


    def __getitem__(self,index):
        img_path =  self.img_files[index%len(self.img_files)]
        #print(index,' ',img_path)
        img = cv2.imread(img_path)
        img_resize = self.imgResize(img)  
        # final image operations
        img_resize = img_resize[:,:,0]
        img_resize = np.expand_dims(img_resize,axis=2)
        img_resize = img_resize/255.0
        img_resize = np.transpose(img_resize, (2, 0, 1))  
        img_tensor = torch.from_numpy(img_resize).float()
        #img_tensor = img_tensor.type(torch.FloatTensor)
        return img_tensor 

    def __len__(self):
        return len(self.img_files)

if __name__ == "__main__":
    #run_started = datetime.today().strftime('%m-%d-%y_%H%M')
    print('run id: ',run_started)
    #parser = HARTS_parser(cfg.data_path, run_started)
    #exit()

    train_dataset = HARTS_dataset(parser.train_img_file)
    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    print("Train dataset length: ",len(train_dataset))
    print('Train dataloader: ',len(train_dataloader))
    

    valid_dataset = HARTS_dataset(parser.valid_img_file)
    valid_dataloader = DataLoader(valid_dataset, batch_size=16, shuffle=True)
    print("Valid dataset length: ",len(valid_dataset))
    print('Valid dataloader: ',len(valid_dataloader))

    for i, (inputs, targets) in enumerate(train_dataloader):
        print(inputs.size())
        #print(cfg.output_check)
        #save_image(inputs, os.path.join(cfg.output_check,str(i)+'.png'),range=(0,1),nrow=8)
        print('targets: ',targets)
        #exit()
