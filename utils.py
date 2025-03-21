import Config.parameters as params

import numpy as np
import math

import torch

def imgResize(img):
    h = img.shape[0]
    w = img.shape[1]
    color = (0,0,0)
    new_h = params.resize_height
    new_w = params.resize_width
    result = np.full((new_h, new_w, 3), color, dtype=np.uint8)
    # compute center offset
    xx = (new_w - w) // 2
    yy = (new_h - h) // 2
    result[yy:yy+h, xx:xx+w] = img

    return result

def convert_to_tensor(img):
    img = np.array(img) 
    img = np.expand_dims(img, axis=2)
    img = img/255.0
    img_torch = torch.from_numpy(img)
    img_torch = img_torch.type(torch.FloatTensor)
    img_torch = img_torch.permute(-1, 0, 1)
    return img_torch

def adjust_learning_rate(optimizer, init_lr, epoch, args):
    """Decay the learning rate based on schedule"""
    cur_lr = init_lr * 0.5 * (1. + math.cos(math.pi * epoch / params.epochs))
    for param_group in optimizer.param_groups:
        if 'fix_lr' in param_group and param_group['fix_lr']:
            param_group['lr'] = init_lr
        else:
            param_group['lr'] = cur_lr