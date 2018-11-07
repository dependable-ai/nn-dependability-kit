import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.autograd  import Variable
from torch.autograd.gradcheck import zero_gradients
from torch.utils.data import Dataset, DataLoader

import matplotlib.patches as patches
from PIL import Image



def displayMNIST (img):
    ''' Display MNIST image where pixels are without value 0-1
    '''

    if not (type(img) is np.ndarray) : 
        raise TypeError("Only numpy.array is supported")
    if not (img.ndim == 3 and img.shape[0] == 1):
        raise AttributeError("Expects (1, 28, 28) shape")
    
    img = np.squeeze(img, axis=0) * 255
    pixels = img.reshape((28, 28))
    
    # Create 2-inch by 2-inch image
    plt.figure(figsize=(1.5,1.5))
    plt.imshow(pixels, cmap='gray')
    plt.show()


def displayGTSRB (img):
    ''' Display image from German Traffic Sign Recognition Benchmark
    '''

    if not (type(img) is np.ndarray) : 
        raise TypeError("Only numpy.array is supported")
    if not (img.ndim == 3 and img.shape[0] == 3):
        raise AttributeError("Expects (3, length, width) shape")
        
    # show images, by np.moveaxis(img, 0, -1), i.e., moving the axis from (3, 32, 32) to (32, 32, 3)
    plt.figure(figsize=(2,2))
    image = plt.imshow(np.moveaxis(img, 0, -1))
    plt.grid(False)


def iterative_FGSM_attack (img, label, model):    
    # eps = 5 * 8 / 225.     
    eps = 1 * 8 / 225. 
    steps = 100
    norm = float('inf')
    # step_alpha = 0.05 
    step_alpha = 0.01 
    loss = nn.CrossEntropyLoss()
    
    x, y = Variable(img, requires_grad=True), Variable(label)
    result = img
    adv = torch.zeros(result.shape)
    for step in range(steps):
        zero_gradients(x)
        out = model(x)        
        # Returns the maximum value of each row of the input tensor in the given dimension dim
        _ , y.data = torch.max(out.data, 1)
        
        if ((not (y.data == label)) and step > 1) :
            return result.cpu(), adv.cpu(), True
            # print("perturbed input for "+ str(label[0])+  " is now identified to be: "+ str(y.data[0]))
            
        _loss = loss(out, y)
        _loss.backward()
        normed_grad = step_alpha * torch.sign(x.grad.data)
        step_adv = x.data + normed_grad
        adv = step_adv - img
        adv = torch.clamp(adv, -eps, eps)
        result = img + adv
        # torch.clamp(input, min, max, out=None) → Tensor
        # Clamp all elements in input into the range [min, max] and return a resulting Tensor.
        result = torch.clamp(result, 0.0, 1.0)
        x.data = result        
        
    return result.cpu(), adv.cpu(), False


def iterative_targeted_FGSM_attack (img, label, anotherLabel, model):    
    
    eps = 5 * 8 / 225. 
    #eps = 1 * 8 / 225. 
    steps = 100
    norm = float('inf')
    step_alpha = 0.05 
    #step_alpha = 0.01
    loss = nn.CrossEntropyLoss()
    
    x, y = Variable(img, requires_grad=True), Variable(anotherLabel)
    result = img
    adv = torch.zeros(result.shape)
    for step in range(steps):
        zero_gradients(x)
        out = model(x)         
        # Returns the maximum value of each row of the input tensor in the given dimension dim
        _ , y.data = torch.max(out.data, 1)
        if (step == 1 and y.data == anotherLabel):
            # It an image directly to the perturbed class. Do nothing
            result.cpu(), adv.cpu(), False
        
        if (y.data == anotherLabel and step > 1) :
            return result.cpu(), adv.cpu(), True
            # print("perturbed input for "+ str(label[0])+  " is now identified to be: "+ str(y.data[0]))
            
        _loss = loss(out, y)
        _loss.backward()
        normed_grad = step_alpha * torch.sign(x.grad.data)
        step_adv = x.data + normed_grad
        adv = step_adv - img
        adv = torch.clamp(adv, -eps, eps)
        result = img + adv
        # torch.clamp(input, min, max, out=None) → Tensor
        # Clamp all elements in input into the range [min, max] and return a resulting Tensor.
        result = torch.clamp(result, 0.0, 1.0)
        x.data = result        
        
    return result.cpu(), adv.cpu(), False
    


def drawFrontCar(imgFile, boundingBox = None):
    im = np.array(Image.open(imgFile), dtype=np.uint8)

    # Create figure and axes
    fig,ax = plt.subplots(1)

    # Display the image
    ax.imshow(im)

    if not (boundingBox == None):
        # Create a Rectangle patch
        rect = patches.Rectangle((int(boundingBox[0]-boundingBox[2]/2),int(boundingBox[1]-boundingBox[3]/2)),int(boundingBox[2]),int(boundingBox[3]),linewidth=1,edgecolor='r',facecolor='none')

        # Add the patch to the Axes
        ax.add_patch(rect)

    plt.show()

    
