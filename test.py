
import torch
#Torch Library
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import cv2
from PIL import Image


import os
import random


from utils import imgResize, convert_to_tensor
from model_cutpaste import CutPasteNet
import Config.configuration as cfg
import Config.parameters as params


class TestDataset(Dataset):
    def __init__(self, img_files):
        super(TestDataset,self).__init__()
        self.img_files = img_files
        random.shuffle(self.img_files)

    def __getitem__(self, index):
        img_path =  self.img_files[index%len(self.img_files)]
        #img_path = image_container['path']
        img = cv2.imread(img_path)
        #cv2.imwrite(img)
        img_resize = imgResize(img)
        img_resize = img_resize[:,:,0]
        image = convert_to_tensor(img_resize)
        return img_path, image

    def __len__(self):
        return len(self.img_files)


class TestParser(object):
    def __init__(self, data_path):
        super(TestParser, self).__init__()
        self.data_path = data_path
        self.img_files = self._load_test_data()

    def _load_test_data(self):
        img_files = []
        for folder_id in os.listdir(self.data_path):
            for dwelling_time in os.listdir(os.path.join(self.data_path, folder_id)):
                image_dir_path = os.path.join(self.data_path, folder_id, dwelling_time)
                for img_name in os.listdir(image_dir_path):
                    img_path = os.path.join(image_dir_path, img_name)
                    img_files.append(img_path)
        return img_files


class Test():
    def __init__(self, test_data_path, saved_model_path, use_cuda = True):
        super(Test, self).__init__()
        self.saved_model_path = saved_model_path
        self.use_cuda = use_cuda
        self.data_parser = TestParser(test_data_path)
        self.testDataset = TestDataset(self.data_parser.img_files)
        self.model = self._init_model()
        

    def test_anomaly(self):
        self.model.load_state_dict(torch.load(self.saved_model_path)['state_dict'])
        self.model.eval()

        print("test dataset: ",len(self.testDataset))
        test_dataloader = torch.utils.data.DataLoader(self.testDataset, batch_size=1, shuffle=False)
        output_holder = {}
        prediction_holder = {}
        with torch.no_grad():
            for i, (img_path, image) in enumerate(test_dataloader):
                if self.use_cuda:
                    image = image.to(f'cuda:{self.model.device_ids[0]}')
                outputs = self.model(image)
                prediction = torch.argmax(outputs, axis=1)
                
                img_path = img_path[0]
                

                output_holder[img_path] = torch.nn.functional.softmax(outputs)
                prediction_holder[img_path] = prediction.item()
        print("######output######")
        print(output_holder)
        print("######prediction######")
        print(prediction_holder)




    def _init_model(self):
        model = CutPasteNet()
        if self.use_cuda:
            #######To load in Single GPU
            #model.to(device)
            #######

            #####To load in Multiple GPU. In FICS server we have 4 GPU. That's why 4 device ids here.
            model = nn.DataParallel(model, device_ids = [2, 3])
            model.to(f'cuda:{model.device_ids[0]}')
            ######
        return model


if __name__=='__main__':
    use_cuda = torch.cuda.is_available()
    testTask = Test(cfg.test_data_path, cfg.saved_model_path, use_cuda)
    testTask.test_anomaly()




    
    

    