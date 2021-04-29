import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
import time

import model
import function
import transformer
import os

# SETTINGS
IMAGE_SIZE = [224,224]
NUM_EPOCHS = 1
BATCH_SIZE = 4
CONTENT_WEIGHT = 17
STYLE_WEIGHT = 50
ADAM_LEARNING_RATE = 0.001

BASE_PATH = os.getcwd()
# SAVE OUTPUT
SAVE_EVERY = 1
SAVE_MODEL_PATH = BASE_PATH+r'\tt\outputs\models\models'
SAVE_IMAGE_PATH = BASE_PATH+r'\tt\outputs\models\generated_image'

DATASET_PATH = r'\ContentImage'

# SEED = 35
PLOT_LOSS = 1

# PATH IMAGE
STYLE_IMG_PATH = BASE_PATH+r'\StyleImage\Chakrabhan\0001.jpg'

def train():
    # check device
    device = ("cuda" if torch.cuda.is_available() else "cpu")
    print("using ",device)
    # if device=="cuda":
    #     print('__CUDNN VERSION:', torch.backends.cudnn.version())
    #     print('__Number CUDA Devices:', torch.cuda.device_count())
    #     print('__CUDA Device Name:',torch.cuda.get_device_name(0))
    #     print('__CUDA Device Total Memory [GB]:',torch.cuda.get_device_properties(0).total_memory/1e9)

    # Dataset
    preprocess = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])])
    # print('dataset_path : '+BASE_PATH+DATASET_PATH)
    # print('-> sum : '+str(sum([len(files) for r, d, files in os.walk(BASE_PATH+DATASET_PATH)])))
    train_dataset = datasets.ImageFolder(BASE_PATH+DATASET_PATH, transform=preprocess)
    print('# of train dataset = ',len(train_dataset))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # Model
    TransformerNetwork = transformer.TransformerNetwork().to(device)
    VGG = model.VGG19().to(device)
    # print(VGG)

    # Style Feature
    style_img = function.loadImg(STYLE_IMG_PATH)
    style_tensor = function.im_convertT(style_img).to(device)
    _,C,H,W = style_tensor.shape
    style_features = VGG(style_tensor.expand([BATCH_SIZE, C, H, W]))
    style_gram = {}
    for key, value in style_features.items():
        style_gram[key] = function.gram_matrix(value)

    # Optimizer
    num_iteration = NUM_EPOCHS
    optim_state = {
        'max_iter' : num_iteration,
        'tolerance_change' : -1,
        'tolerance_grad' : -1
    }
    lbfgs_num_correction = 100
    optim_state['history_size'] = lbfgs_num_correction
    optimizer = optim.LBFGS(TransformerNetwork.parameters(),**optim_state)
    # optimizer = optim.Adam(TransformerNetwork.parameters(),lr=ADAM_LEARNING_RATE)

    content_loss_history = []
    style_loss_history = []
    total_loss_history = []

    batch_content_loss = 0
    batch_style_loss = 0
    batch_total_loss = 0

    # Training
    batch_count = 1
    start_time = time.time()
    for epoch in range(NUM_EPOCHS):
        print("Epoch : {}/{}".format(epoch+1,NUM_EPOCHS))
        
        def closure(train_loader,batch_style_loss,batch_content_loss,batch_total_loss):
            for content_batch, _ in train_loader:
                curr_batch_size = content_batch.shape[0]

                torch.cuda.empty_cache()
                
                optimizer.zero_grad()

                content_batch = content_batch[:,[2,1,0]].to(device)
                generated_batch = TransformerNetwork(content_batch)
                content_features = VGG(content_batch)
                generated_features = VGG(generated_batch)

                # content loss
                MSELoss = nn.MSELoss().to(device)
                content_loss = CONTENT_WEIGHT * MSELoss(generated_features['relu2_2'],content_features['relu2_2'])
                batch_content_loss += content_loss
                
                # style loss
                style_loss = 0
                for key,value in generated_features.items():
                    s_loss = MSELoss(function.gram_matrix(value),style_gram[key][:curr_batch_size])
                    style_loss += s_loss
                style_loss *= STYLE_WEIGHT
                batch_style_loss += style_loss.item()

                # total loss
                total_loss = content_loss+style_loss
                batch_total_loss += total_loss.item()

                total_loss.backward()

            optimizer.step(closure)

            # save model
            if (((batch_count-1)%SAVE_EVERY==0) or (batch_count==NUM_EPOCHS*len(train_loader))):
                print("=== Iteration : {}/{} ===".format(batch_count,NUM_EPOCHS*len(train_loader)))
                print("\tContent Loss : {:2f}".format(batch_content_loss/batch_count))
                print("\tStyle Loss : {:2f}".format(batch_style_loss/batch_count))
                print("\tTotal Loss : {:2f}".format(batch_total_loss/batch_count))
                print("\t=> Time elapsed : {} seconds".format(time.time()-start_time))

                checkpoint_path = SAVE_MODEL_PATH+"checkpoint_"+str(batch_count-1)
                torch.save(TransformerNetwork.state_dict(),checkpoint_path)
                print("Save TransformerNetwork at {}".format(checkpoint_path))

                sample = generated_batch[0].clone().detach().unsqueeze(dim=0)
                sample_image = function.im_convert(sample)
                sample_path = SAVE_IMAGE_PATH+"sample_"+str(batch_count-1)+'.jpg'
                # function.saveImg(sample_image,sample_path)
                function.saveImg(sample,sample_path)
                function.showImg(sample_image)
                print("Save generated image at {}".format(sample_path))

                content_loss_history.append(batch_content_loss/batch_count)
                style_loss_history.append(batch_style_loss/batch_count)
                total_loss_history.append(batch_total_loss/batch_count)

            batch_count += 1    
        
    stop_time = time.time()

    # Show Result
    print("Training time : {} seconds".format(stop_time-start_time))
    print("=> Content Loss ",content_loss_history)
    print("=> Style Loss ",style_loss_history)
    print("=> Total Loss ",total_loss_history)

    TransformerNetwork.eval()
    TransformerNetwork.cpu()
    final_path = SAVE_MODEL_PATH+"_transformer_weight.pt"
    print("Save TransformerNetwork weight at {}".format(final_path))
    torch.save(TransformerNetwork.state_dict(),final_path)
    print("Save final model")

    if(PLOT_LOSS):
        function.plotLoss(content_loss_history,style_loss_history,total_loss_history)

train()