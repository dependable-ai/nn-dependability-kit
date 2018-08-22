import torch
import torch.nn as nn
import torch.tensor
import numpy as np
from torch.autograd  import Variable
from torch.autograd.gradcheck import zero_gradients

def generateTestCase (img, targetedNeuronIndex, desiredNAP, model):    
    """ Generate a test case to satisfy the specified neuronActivationPattern
    
    Args:
        img: an image to be perturbed towards the desired activation pattern
        targetedNeuronIndex:  Indices of neuron with the goal of satisfying the pattern
        desiredNAP: desired shape (>0 is set to 1; <= 0 is set to -1)
        model: neural network under analysis
    """
    

    steps = 20000
    loss = nn.L1Loss()
    alpha = 0.1 
    x = Variable(img, requires_grad=True)
    result = img
    delta = torch.zeros(result.shape)

    for step in range(steps):
        if (step == steps -1):
            print("Unable to find an image to satisfy the required pattern!")
        
        zero_gradients(x)
        out, intermediate = model.forwardWithIntermediate(x)
        

        # The output label is first set to values equal to the computed layer. 
        labels = torch.tensor(intermediate)    

        # Modify the label, such that if the sign of the specified neuron is different, use the desired value. By doing so, 
        # (1) the loss of other un-specified neuron will be 0, 
        # (2) for specified neuron with correct sign, the loss will be 0
        # (3) for specified neuron without correct sign, there will be a loss
        satisfied = True
        for i in range(len(targetedNeuronIndex)):
            if not ((intermediate.detach().numpy().squeeze(0)[targetedNeuronIndex[i]]>0) == (desiredNAP[i]>0)):
                satisfied = False
                labels[0][targetedNeuronIndex[i]] = desiredNAP[i]

        if satisfied:
            if step == 0:
                print("The original image already satisfy the required pattern!")
                return result.cpu(), True    
            else :
                print("Found an image to successfully create the required pattern:")
                return result.cpu(), True                

        if(step == 0):
            print(intermediate)
            print(labels)

        y = Variable(labels)
        _loss = loss(intermediate, y)
        _loss.backward()
                    
        result = x.data - (alpha * x.grad.data)
        # TODO: Here we are not clamping the input
        result = torch.clamp(result, 0.0, 1.0)
        x.data = result
        # x.data = x.data - (alpha * x.grad.data)        
        
        if(step % 500 == 0):
            print(str(step)+": "+str(intermediate.detach().numpy().squeeze(0)[targetedNeuronIndex[0]]) + ", "+ str(intermediate.detach().numpy().squeeze(0)[targetedNeuronIndex[1]]))
        
        
    return result.cpu(), False
    


