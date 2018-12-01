import numpy as np 
import scipy
import math
import scipy.misc
import os
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd  import Variable
from torch.autograd.gradcheck import zero_gradients

import matplotlib 
import seaborn as sns
import matplotlib.pyplot as plt

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()    

    
def evaluateImageAndComputeSoftmax(net, image, label):
    """
    Evaluate the quality of the perturbation.

    
    Returns: the predicted class, together the quantity of probability by applying softmax
    Raises: error when image dimension is not OK. 
    
    """
    if not (image.ndim == 3 and image.shape[0] == 3):
        raise Exception("Dimension should be (3 x length x width)")
        
    inputs = np.expand_dims(image, axis=0)
    inputs = torch.from_numpy(np.float32(inputs))
    outputs = net(inputs)
    _, predicted = torch.max(outputs, 1)
    #print("Predicted: "+str(predicted[0].data))
    #print("Correct: "+str(label))
    #print("Prob: "+str(softmax(outputs[0].detach().numpy())[predicted[0]]))
    #print(softmax(outputs[0].detach().numpy())[predicted[0]])
    
    if (int(predicted[0].data) == int(label)):
        # If the prediction is correct, take the largest probability
        return int(predicted[0].data), softmax(outputs[0].detach().numpy())[predicted[0]]
    else:    
        # Otherwise, take the original label and check what is the probability now
        return int(predicted[0].data), softmax(outputs[0].detach().numpy())[int(label)]


def add_weather(image, weather_typ):
    if not (image.ndim == 3 and image.shape[2] == 3):
        raise Exception("Dimension should be (length x width x 3)")

    if weather_typ == "snow":
        imageInt = np.rint((image*255))
        tmp = imageInt.astype(int)
        image_HLS = cv2.cvtColor(imageInt,cv2.COLOR_RGB2HLS) ## Conversion to HLS
        image_HLS = np.array(image_HLS, dtype = np.float64) 
        brightness_coefficient =  2.5 
        snow_point= 150 ## increase this for more snow
        image_HLS[:,:,1][image_HLS[:,:,1]<snow_point] = image_HLS[:,:,1][image_HLS[:,:,1]<snow_point]*brightness_coefficient ## scale pixel values up for channel 1(Lightness)
        image_HLS[:,:,1][image_HLS[:,:,1]>255]  = 255 ##Sets all values above 255 to 255
        image_HLS = np.array(image_HLS, dtype = np.uint8)
        image_RGB = cv2.cvtColor(image_HLS, cv2.COLOR_HLS2RGB) ## Conversion to RGB
        return image_RGB

    elif weather_typ == "haze" or weather_typ == "fog": 
        # Create an overly simplified version out of the the open source SW from https://github.com/noahzn/FoHIS (Foggy and Hazy Images), 
        # where for simple object recognition, one does not need to have depth information associated.        
        height, width = image.shape[:2]
        # Change from 1.5 to 1 to create less haze
        distance = np.ones((height, width))*2

        distance_through_fog = np.zeros_like(distance)
        distance_through_haze = np.zeros_like(distance)
        distance_through_haze_free = np.zeros_like(distance)        
        
        I = np.empty_like(image)
        result = np.empty_like(image)

        I[:, :, 0] = image[:, :, 0] * np.exp(-1*distance)
        I[:, :, 1] = image[:, :, 1] * np.exp(-1*distance)
        I[:, :, 2] = image[:, :, 2] * np.exp(-1*distance)
        O = 1-np.exp(-1*distance)
        
        Ial = np.empty_like(image)  # color of the fog/haze
        if weather_typ == "haze":
            Ial[:, :, 0] = 225/255
            Ial[:, :, 1] = 225/255
            Ial[:, :, 2] = 201/255
        else:
            Ial[:, :, 0] = 240/255
            Ial[:, :, 1] = 240/255
            Ial[:, :, 2] = 240/255
        
        result[:, :, 0] = I[:, :, 0] + O * Ial[:, :, 0]
        result[:, :, 1] = I[:, :, 1] + O * Ial[:, :, 1]
        result[:, :, 2] = I[:, :, 2] + O * Ial[:, :, 2]
   
        return result

        
def add_noise(noise_typ, image):
    '''
    Based on modification of https://stackoverflow.com/questions/22937589/how-to-add-noise-gaussian-salt-and-pepper-etc-to-image-in-python-with-opencv
    '''
    if not (image.ndim == 3 and image.shape[2] == 3):
        raise AttributeError("Expects (length, width, 3) shape")
    
    # This function supposes that the numpy is of shape (row,col,channel)      
    noisy = None
    if noise_typ == "gauss":
        row,col,ch= image.shape
        mean = 0.5
        var = 0.001
        sigma = var**0.5
        gauss = np.random.normal(mean,sigma,(row,col,ch))
        gauss = gauss.reshape(row,col,ch)
        noisy = image + gauss

    elif noise_typ == "s&p":
        row,col,ch = image.shape
        s_vs_p = 0.5
        amount = 0.004
        out = np.copy(image)
        # Salt mode
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt))
              for i in image.shape]
        out[tuple(coords)] = 1

        # Pepper mode
        num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper))
              for i in image.shape]
        out[tuple(coords)] = 0
        return out
    elif noise_typ == "poisson":
        vals = len(np.unique(image))
        vals = 2 ** np.ceil(np.log2(vals))
        noisy = np.random.poisson(image * vals) / float(vals)
        
    #elif noise_typ =="speckle":
    #    row,col,ch = image.shape
    #    gauss = np.random.randn(row,col,ch)
    #    gauss = gauss.reshape(row,col,ch)        
    #    noisy = image + image * gauss    
    
    # Ensure that images are within 0 and 1
    noisy = torch.from_numpy(noisy)
    noisy = torch.clamp(noisy, 0.0, 1.0)
    return noisy.numpy()

def iterative_FGSM_attack (img, label, model):    
    
    eps = 1 * 8 / 225. 
    steps = 100
    norm = float('inf')
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
        

        
        
class Perturbation_Loss_Metric():
    """Computing (adversarial) perturbation loss metric.

    Details on how this metric is defined is in Sec II.D of the paper 
    "Towards Dependability Metrics for Neural Networks" (https://arxiv.org/abs/1806.02338) 
    It is essentially applying known perturbation and observe the confidence drop.

    Attributes:

    """


    def __init__(self):
        self.lossMatrices = []    
        self.perturbationKind = ["gauss", "poisson", "s&p", "fgsm", "snow", "haze", "fog"]
    

    
    def getMetricQuantity(self, criterion):
        
        npArray = np.array(self.lossMatrices)
        
        result = []
    
        if(criterion == "AVERAGE_LOSS"):
            result.extend(np.mean(npArray, axis=0).tolist())
        elif (criterion == "MAX_LOSS"):
            npArray = npArray[npArray[:,0].argsort()]
            result.append(npArray[-1][0])
            npArray = npArray[npArray[:,1].argsort()]
            result.append(npArray[-1][1])        
            npArray = npArray[npArray[:,2].argsort()]
            result.append(npArray[-1][2])          
            npArray = npArray[npArray[:,3].argsort()]
            result.append(npArray[-1][3])        
            npArray = npArray[npArray[:,4].argsort()]
            result.append(npArray[-1][4])   
            npArray = npArray[npArray[:,5].argsort()]
            result.append(npArray[-1][5])    
            npArray = npArray[npArray[:,6].argsort()]
            result.append(npArray[-1][6])                
        elif (criterion == "TOP_10%_LARGEST_LOSS"):
            npArray = npArray[npArray[:,0].argsort()]
            result.append(npArray[int(len(self.lossMatrices)*9/10)][0])  
            npArray = npArray[npArray[:,1].argsort()]
            result.append(npArray[int(len(self.lossMatrices)*9/10)][1])           
            npArray = npArray[npArray[:,2].argsort()]
            result.append(npArray[int(len(self.lossMatrices)*9/10)][2])               
            npArray = npArray[npArray[:,3].argsort()]
            result.append(npArray[int(len(self.lossMatrices)*9/10)][3])                
            npArray = npArray[npArray[:,4].argsort()]
            result.append(npArray[int(len(self.lossMatrices)*9/10)][4])     
            npArray = npArray[npArray[:,5].argsort()]
            result.append(npArray[int(len(self.lossMatrices)*9/10)][5])   
            npArray = npArray[npArray[:,6].argsort()]
            result.append(npArray[int(len(self.lossMatrices)*9/10)][6])                  
        else:
            raise AttributeError("Currently only AVERAGE_LOSS, MAX_LOSS, TOP_10%_LARGEST_LOSS are supported")
        
        return result         

        
    def printMetricQuantity(self, criterion):
    
        labels = np.array(self.perturbationKind)
        stats = np.array(self.getMetricQuantity(criterion))

        #print(labels)
        #print(stats)
        
        angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False)
        # close the plot
        stats=np.concatenate((stats,[stats[0]]))
        angles=np.concatenate((angles,[angles[0]]))
        sns.set()
        #fig=sns.plt.figure()
        fig= plt.figure()
        ax = fig.add_subplot(111, polar=True)
        ax.plot(angles, stats, 'o-', linewidth=2)
        ax.fill(angles, stats, alpha=0.25)
        ax.set_thetagrids(angles * 180/np.pi, labels)
        ax.set_title(criterion)
        ax.grid(True)
        
        
    def addInputs(self, net, image, label):
        """ 
            Modify a single image based on predefined perturbation directions. 
            
            Keyword arguments:
            
        """
        
        c0, prob0 = evaluateImageAndComputeSoftmax(net, image, label) 
        
        imageFor = np.moveaxis(image, 0, -1)
        
        # Gaussian noise
        result1 = np.moveaxis(add_noise("gauss", imageFor), -1, 0)
        # util.displayGTSRB(image)
        cl, prob1 = evaluateImageAndComputeSoftmax(net, result1, label)

        # poisson noise
        result2 = np.moveaxis(add_noise("poisson", imageFor), -1, 0)
        c2, prob2 = evaluateImageAndComputeSoftmax(net, result2, label)

        # salt-and-pepper
        result3 = np.moveaxis(add_noise("s&p", imageFor), -1, 0)
        c3, prob3 = evaluateImageAndComputeSoftmax(net, result3, label)
        
        # FGSM
        img = (torch.from_numpy(image)).unsqueeze(0)
        lab = (torch.from_numpy(label)).unsqueeze(0)
        adv_img, noise, attackSuccessful = iterative_FGSM_attack(img, lab, net)
        c4, prob4 = evaluateImageAndComputeSoftmax(net, adv_img[0].numpy(), label)
        
        # Snow
        imagep = add_weather(imageFor, "snow")
        result5 = np.moveaxis(imagep/255.0, -1, 0)
        c5, prob5 = evaluateImageAndComputeSoftmax(net, result5, label)

        # Haze
        imagep = add_weather(imageFor, "haze")
        result6 = np.moveaxis(imagep, -1, 0)
        c6, prob6 = evaluateImageAndComputeSoftmax(net, result6, label)
        
        # fog
        imagep = add_weather(imageFor, "fog")
        result7 = np.moveaxis(imagep, -1, 0)
        c7, prob7 = evaluateImageAndComputeSoftmax(net, result7, label)
        
        # Compute the performance drop of current image
        performanceDrop = [100*(prob0 - prob1), 100*(prob0 - prob2), 100*(prob0 - prob3), 100*(prob0 - prob4), 100*(prob0 - prob5), 100*(prob0 - prob6), 100*(prob0 - prob7)]
        
        # If after the noise, the identification rate increases, just set it to be 0.
        performanceDropClip = []
        for i in range(7):
            if performanceDrop[i] < 0:
                performanceDropClip.append(0.0)
            else:
                performanceDropClip.append(performanceDrop[i])
        
        self.lossMatrices.append(performanceDropClip)
        
        
    def dumpMetricState(self):
        return self.lossMatrices

