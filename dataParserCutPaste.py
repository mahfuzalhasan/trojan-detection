import os
import numpy as np
from multiprocessing import Pool
from functools import partial
import pickle
import random
import cv2
import pickle
import statistics
from datetime import datetime

import sys
#sys.path.insert(1,"/home/UFAD/mdmahfuzalhasan/Documents/Projects/HARTS/Classification/Config")

import Config.configuration as cfg
import Config.parameters as params



class DataParser(object):
    def __init__(self, data_path, run_id):
        super(DataParser,self).__init__()
        self.data_path = data_path  
        self.train_img_file = []
        self.valid_img_file = []
        self.img_files = []

        data_split_dir = cfg.data_split
        if not os.path.exists(data_split_dir):
            os.makedirs(data_split_dir)

        train_image_path = os.path.join(data_split_dir,"train_images.pkl")
        validation_image_path = os.path.join(data_split_dir, "valid_images.pkl")

        class_sample = {0:500,1:500,2:500,3:300,4:1000,5:200,6:1500}


        if os.path.exists(train_image_path):
            print('load from pre-saved distribution')
            self.train_img_file = pickle.load(open(train_image_path, 'rb'))
            self.valid_img_file = pickle.load(open(validation_image_path, 'rb'))
        else:
            print("Creating train and validation")
            img_dicts = self._load_harts_data(self.data_path)
            self.img_files = []
            for class_id in range(7):
                samples = [img for img in img_dicts if img['label']==class_id]
                print(f'samples from class {class_id}:{len(samples)}')
                random.shuffle(samples)
                # self.img_files.extend(samples[:class_sample[class_id]])
                self.img_files.extend(samples)
            # exit()
            print(f'total samples:{len(self.img_files)}')
            # exit()
            self.train_img_file, self.valid_img_file = self._split()
            #self.valid_img_file = self._load_harts_data(self.data_path, valid=True)
            pickle.dump(self.train_img_file, open(train_image_path, 'wb'))
            pickle.dump(self.valid_img_file, open(validation_image_path, 'wb'))

        random.shuffle(self.train_img_file)
        random.shuffle(self.valid_img_file)
        print("train data: ",len(self.train_img_file))
        print("validation data: ",len(self.valid_img_file))


    #def __len__(self):
    #    return len(self.train_img_file)

    def _split(self):
        random.shuffle(self.img_files)

        index = int(0.9 * len(self.img_files))
        train_img_file = self.img_files[0:index+1]
        valid_img_file = self.img_files[index+1:]
        return train_img_file, valid_img_file  

             
    def _load_harts_data(self, data_path, valid=False):
        img_files = []
        if not valid:
            print("Train")
        else:
            print("Val")
        for class_dir in os.listdir(data_path):
            count = 0
            class_files = []
            
            #print('##########class: ',class_dir)
            max_height = -1
            max_width = -1
            heights = []
            widths = []
            ratios = []
            
            for dwelling_time in os.listdir(os.path.join(data_path,class_dir)):
                dwelling_time_separation = os.path.join(data_path, class_dir, dwelling_time)
                for image_dir in os.listdir(dwelling_time_separation):
                    image_dir_path = os.path.join(dwelling_time_separation,image_dir)
                    #print('images dir: ',image_dir_path)
                    for img_name in os.listdir(image_dir_path):
                        img_path = os.path.join(image_dir_path, img_name)
                        img = cv2.imread(img_path)
                        
                        height = img.shape[0]
                        width = img.shape[1]
                        heights.append(height)
                        widths.append(width)
                        ratios.append(height/width)
                        if height > max_height:
                            max_height = height
                        if width > max_width:
                            max_width = width
                        

                        dictionary = {}
                        dictionary['path'] = img_path
                        dictionary['folder'] = image_dir_path
                        dictionary['label'] = int(class_dir)
                        img_files.append(dictionary)
                        count += 1
                    #print("max height & width: ",max_height," ",max_width)
            #print('class: ',int(class_dir),' total count: ', count)
            
            #print('min h: ',min(heights), 'max h:',max(heights), 'mean h: ',statistics.mean(heights), 'stdev h: ',statistics.pstdev(heights))
            #print('min w: ',min(widths), 'max w:',max(widths), 'mean w: ',statistics.mean(widths), 'stdev w: ',statistics.pstdev(widths))
            #print('min r: ',min(ratios),' max r: ',max(ratios), ' mean r:', statistics.mean(ratios))
            
        return img_files
    

if __name__=="__main__":
    run_started = datetime.today().strftime('%m-%d-%y_%H%M')
    parser = DataParser(cfg.data_path, run_id=run_started)