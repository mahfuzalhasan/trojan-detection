#training
num_epoch = 500
batch_size = 256
#resume = None #"/home/UFAD/mdmahfuzalhasan/Documents/Results/output_harts/saved_models/02-10-21_1341_msgan/00074.pth"
resume = "/home/UFAD/mdmahfuzalhasan/Documents/Results/output_harts/saved_models/Classification/04-23-21_0814/model_7.pth"
gpu = 2
smoothing_value = 0.85

#test
result_dir = "/home/UFAD/mdmahfuzalhasan/Documents/Results/output_harts/test_output"
name = "02-11-21_0752_msgan"
num = 100


#optimizer
learning_rate = 1e-4

#data
num_classes = 3
channels = 1
img_size = 64
nz = 100#latent_dim = 100
resize_height = 96
resize_width = 96
img_save_freq = 3
num_channel = 1

#augmentation
height = 512
width = 512
scale = (0.2, 1)
ratio=(1, 1)
scale_factor = 0.5
isReduced = False


#model related
checkpoint = 1      #no of epoch to save model

#sem_image
paths = ["Set39_D6_Cropped.tif"]


#dataset
validation_set = "DT4_Mag12"
tool_created_img = False

#output
display_freq = 150