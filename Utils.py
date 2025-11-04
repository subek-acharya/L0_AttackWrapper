import torch
import torch.nn as nn
import numpy as np

#Convert a dataloader into x and y tensors 
def DataLoaderToTensor(dataLoader):
    #First check how many samples in the dataset
    numSamples = len(dataLoader.dataset) 
    sampleShape = GetOutputShape(dataLoader) #Get the output shape from the dataloader
    sampleIndex = 0
    #xData = torch.zeros(numSamples, sampleShape[0], sampleShape[1], sampleShape[2])
    xData = torch.zeros((numSamples,) + sampleShape) #Make it generic shape for non-image datasets
    yData = torch.zeros(numSamples)
    #Go through and process the data in batches 
    for i, (input, target) in enumerate(dataLoader):
        batchSize = input.shape[0] #Get the number of samples used in each batch
        #Save the samples from the batch in a separate tensor 
        for batchIndex in range(0, batchSize):
            xData[sampleIndex] = input[batchIndex]
            yData[sampleIndex] = target[batchIndex]
            sampleIndex = sampleIndex + 1 #increment the sample index 
    return xData, yData

def TensorToNumpy(x_tensor, y_tensor):
    x_numpy = x_tensor.cpu().numpy()
    x_numpy = x_numpy.transpose(0, 2, 3, 1)  # NCHW -> NHWC
    
    y_numpy = y_tensor.cpu().numpy()
    y_numpy = y_numpy.astype(np.int64)
    
    return x_numpy, y_numpy

def get_predictions(model, x_nat, y_nat, device):
    x = torch.from_numpy(x_nat).permute(0, 3, 1, 2).float().to(device)
    y = torch.from_numpy(y_nat).to(device)
    with torch.no_grad():
        output = model(x)
    
    return (output.max(dim=-1)[1] == y).cpu().numpy()

def get_predictions_and_gradients(model, x_nat, y_nat, device):
    x = torch.from_numpy(x_nat).permute(0, 3, 1, 2).float().to(device)
    x.requires_grad_()
    y = torch.from_numpy(y_nat).to(device)

    with torch.enable_grad():
        output = model(x)
        loss = nn.CrossEntropyLoss()(output, y)

    grad = torch.autograd.grad(loss, x)[0]
    grad = grad.detach().permute(0, 2, 3, 1).cpu().numpy()

    pred = (output.detach().max(dim=-1)[1] == y).detach().cpu().numpy()

    return pred, grad

