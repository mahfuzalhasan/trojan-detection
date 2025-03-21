#system library
import sys
sys.path.append('../')


#python library
import random

#torch library
import torch

#third-party library
import albumentations as A
import cv2


#other project files
import Config.configuration as cfg 
import Config.parameters as params


class Augmentation():
    def __init__(self):
        super(Augmentation, self).__init__()
        self.transform_structure = self._get_geometric_transformation()
        #self.transform_pixel = self._get_pixel_level_transformation()


    def  _get_geometric_transformation(self):
        sigma_limit = random.uniform(0.1, 2.0)
        window = random.randrange(3, 7, 2)
        transform = A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.GaussianBlur(blur_limit=(window, window), sigma_limit=sigma_limit, p=0.5),
                A.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1, p=0.8)
            ],
        )
        return transform


    def generation(self, img, isReduced):
        scale_factor = params.scale_factor
        if isReduced:
            width = int(img.shape[1] * scale_factor)
            height = int(img.shape[0] * scale_factor)
            dsize = (width, height)
            img = cv2.resize(img, dsize)
            target = cv2.resize(target, dsize)
        transformed = self.transform_structure(image=img)
        aug_img = transformed['image']
        #transformed_pixel = self.transform_pixel(image=aug_img)
        return aug_img

