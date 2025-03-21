from tensorboardX import SummaryWriter
import os
import torchvision


import Config.configuration as cfg
import Config.parameters as params

def write_img(visuals, run_id, ep, iteration, val=False ):
    if not val:
        path = os.path.join(cfg.output_dir, run_id, 'train', str(ep))
    else:
        path = os.path.join(cfg.output_dir, run_id, 'val', str(ep))
    if not os.path.exists(path):
        os.makedirs(path)

    input_2 = '%s/input_aug_%05d.jpg' % (path, iteration)
    input_1 = '%s/input_org_%05d.jpg' % (path, iteration)
    anomaly = '%s/anomaly_%05d.jpg' % (path, iteration)

    torchvision.utils.save_image(visuals['input_1'], input_2, normalize=True, nrow=8, value_range=(0, 1))
    torchvision.utils.save_image(visuals['input_org'], input_1, normalize=True, nrow=8, value_range=(0, 1))
    torchvision.utils.save_image(visuals['anomaly'], anomaly, normalize=True, nrow=8, value_range=(0, 1))