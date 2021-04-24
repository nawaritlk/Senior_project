import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets,transforms
import random
import numpy as np
import time

import model
import function


# Setting
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-content_image',help="Content image", default=r'Test\ContentImg.jpg')
parser.add_argument('-style_image',help="Style image", default=r'Test\StyleImage\Chakrabhan\0001.jpg')
parser.add_argument('-src_image',help="Source image for histogram matching", default=None)
parser.add_argument('-ref_image',help="Reference image for histogram matching", default=None)
params = parser.parse_args()

print(params.content_image)
print(params.style_image)

# SETTINGS
IMAGE_SIZE = 224
NUM_EPOCHS = 1
BATCH_SIZE = 4
CONTENT_WEIGHT = 17
STYLE_WEIGHT = 50
ADAM_LEARNING_RATE = 0.001

# SAVE OUTPUT
SAVE_EVERY = 500
SAVE_MODEL_PATH = "output/models/"
SAVE_IMAGE_PATH = "outputs/generated_img/"

DATASET2_PATH = "C:/Users/USER/Desktop/Senior_project/Model/Test/StyleImage/Chakrabhan"
DATASET_PATH = "C:/Users/USER/Desktop/Senior_project/Model/Test/StyleImage/Chalood"

SEED = 35
PLOT_LOSS = 1

# PATH IMAGE
STYLE_IMG_PATH = r'StyleImage/Chakrabhan/0001.jpg'
CONTENT_IMG_PATH = r'Test/ContentImg.jpg'

def train():
    # check device
    device = ("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    import os
    print(os.listdir())

    # Dataset
    preprocess = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])])
    
    train_dataset = datasets.ImageFolder(DATASET_PATH, transform=preprocess)
    train_loader = torch.utils.data.DataLoader(train-dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Model

    VGG = model.VGG19().to(device)
    print(VGG)

    # Style Features
    style_img = function.loadImg(STYLE_IMG_PATH)
    style_tensor = function.im_convertT(style_img).to(device)
    _,C,H,W = style_tensor.shape
    style_features = VGG(style_tensor.expand([BATCH_SIZE, C, H, W]))
    style_gram = {}
    for key, value in style_features.items():
        style_gram[key] = function.gram_matrix(value)

    # Optimizer
    optimizer = optim.Adam([target],lr=ADAM_LEARNING_RATE)

    content_loss = []
    style_loss = []
    total_loss = []

    batch_content_loss = 0
    batch_style_loss = 0
    batch_total_loss = 0


    # Training
    batch_count = 1
    start_time = time.time()
    for epoch in range(NUM_EPOCHS)
        print("Epoch : {}/{}".format(epoch+1,NUM_EPOCHS))
        for content_batch,_ in train_loader:
            curr_batch_size = content_batch.shape[0]
            
        







    stop_time = time.time()

    # Show Result
    print("Training time : {} seconds".format(stop_time-start_time))
    print("=> Content Loss ",content_loss)
    print("=> Style Loss ",style_loss)
    print("=> Total Loss ",total_loss)



    if(PLOT_LOSS):
        function.plotLoss(content_loss,style_loss,total_loss)

train()