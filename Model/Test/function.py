from PIL import Image
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from skimage import exposure
import torch
from torchvision import transforms, datasets

def loadImg(path):
    img = cv.imread(path)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    # cv.imshow("Img",img)
    return img

def showImg(img,name='Image'):
    plt.title(name)
    plt.imshow(img)
    plt.axis('off')
    plt.show()

def saveImg(img,img_path):
    cv.imwrite(img_path, img)

# change tensor to image
def im_convert(tensor):  
    image=tensor.cpu().clone().detach().numpy()     
    image=image.squeeze()  
    image=image.transpose(1,2,0)  
    image=image*np.array((0.5,0.5,0.5))+np.array((0.5,0.5,0.5))  
    image=image.clip(0,1)  
    return image  

# change image to tensor
def im_convertT(img, max_size=None):
    # Rescale the image
    if (max_size==None):
        in_transform = transforms.Compose([
            # transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
            # transforms.Lambda(lambda x: x.mul(255))
        ])    
    else:
        H, W, C = img.shape
        image_size = tuple([int((float(max_size) / max([H,W]))*x) for x in [H, W]])
        in_transform = transforms.Compose([
            # transforms.ToPILImage(),
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
            # transforms.Lambda(lambda x: x.mul(255))
        ])
    # Convert image to tensor
    tensor = in_transform(img)
    tensor = tensor.unsqueeze(dim=0)
    return tensor

#Initializing gram_matrix function for our tensor image   
def gram_matrix(tensor):  
    #Unwrapping the tensor dimensions into respective variables i.e. batch size, distance, height and width   
    _,d,h,w=tensor.size()   
    #Reshaping data into a two dimensional of array or two dimensional of tensor  
    tensor=tensor.view(d,h*w)  
    #Multiplying the original tensor with its own transpose using torch.mm   
    #tensor.t() will return the transpose of original tensor  
    gram=torch.mm(tensor,tensor.t())  
    #Returning gram matrix   
    return gram  

def showStyleContentTarget(style,content,target,title='Show Style Content & Target Image'):
    fig,(ax1,ax2,ax3)=plt.subplots(1,3,figsize=(15,5))  
    fig.suptitle(title, fontsize=16)   
    #Plotting content image   
    ax1.set_title('Content Image')
    ax1.imshow(content)  
    ax1.axis('off')  
    #Plotting style image  
    ax2.set_title('Style Image')
    ax2.imshow(style)
    ax2.axis('off')  
    #Plotting target image  
    ax3.set_title('Generated Image')
    ax3.imshow(target) 
    ax3.axis('off') 
    plt.show()
    plt.close() 

def histogramMatching(src, ref):
    multi = True if src.shape[-1] > 1 else False
    matched = exposure.match_histograms(src, ref, multichannel = multi)
    return matched

# def histogramEqualizeGray(img):
#     gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
#     dst = cv.equalizeHist(gray)
#     return dst

def plotLoss(contentLoss,styleLoss,totalLoss,title='Loss'):
    x = [i for i in range(len(total_loss))]
    plt.figure(figsize=[10,6])
    plt.plot(x, c_loss, label="Content Loss")
    plt.plot(x, s_loss, label="Style Loss")
    plt.plot(x, total_loss, label="Total Loss")
    
    plt.legend()
    plt.xlabel('Every 500 iterations')
    plt.ylabel('Loss')
    plt.title(title)
    plt.show()


STYLE_IMG_PATH = r'StyleImage/Chakrabhan/0001.jpg'
CONTENT_IMG_PATH = r'Test/ContentImg.jpg'
style_img = loadImg(STYLE_IMG_PATH)
content_img = loadImg(CONTENT_IMG_PATH)
showImg(style_img,'Chakrabhan Style Image')
showStyleContentTarget(style_img,content_img,style_img)
preserved_img = histogramMatching(style_img,content_img)
showStyleContentTarget(style_img,content_img,preserved_img)

# # concatanate image Horizontally
# Hori = np.concatenate((style_img,preserve_img), axis=1)
 
# cv.imshow('HORIZONTAL', Hori)
  
# cv.waitKey(0)
# cv.destroyAllWindows()