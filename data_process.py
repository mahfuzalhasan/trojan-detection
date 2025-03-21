import numpy as np
import random

import os

import pandas as pd
import csv
import pickle
import cv2


import Config.configuration as cfg


class DataProcess(object):
    def __init__(self, data_folder):
        self.data_folder = data_folder 

    def _load_SEM_data(self, img_dir, valid=False):
        img_files = []
        if not valid:
            print("Train")
        else:
            print("Val")
        for img_folder in os.listdir(img_dir):
            store = {}
            for img_name in os.listdir(os.path.join(img_dir, img_folder)):
                store={}
                if '6' in img_name:
                    continue
                store['set'] = int(img_folder)
                store['input'] = os.path.join(img_dir, img_folder, img_name)
                target_name = img_name.replace(img_name[1], '6')
                store['target'] = os.path.join(img_dir, img_folder, target_name)
                img_files.append(store)
        return img_files

    def _split(self):
        random.shuffle(self.img_files)
        index = int(0.85 * len(self.img_files))
        train_img_file = self.img_files[0:index+1]
        valid_img_file = self.img_files[index+1:]
        return train_img_file, valid_img_file  


    def read_data_file(self):

        self.img_files = self._load_SEM_data(self.data_folder)
        self.train_files, self.valid_files = self._split()

        print("total: ",len(self.img_files))
        print("train: ",len(self.train_files))
        print("valid: ",len(self.valid_files))
        self._dump_file(self.train_files, 'train')
        self._dump_file(self.valid_files, 'val')


    def _dump_file(self, data_list, file_type):
        save_file_path = os.path.join(cfg.data_path, file_type+'.pkl')
        with open(save_file_path, 'wb') as f:
            pickle.dump(data_list, f)
        

if __name__ == '__main__':
    data_process = DataProcess(cfg.data_folder)
    data_process.read_data_file()