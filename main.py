import torch
from torch.utils import data
from torchvision import datasets, transforms
from torch.utils.data import TensorDataset, DataLoader, ConcatDataset
import numpy as np

import DataManagerPytorch as DMP
from resnet import ResNet18

import L0_PGD_AttackWrapper as L0PGD
import L0_Linf_PGD_AttackWrapper as L0LinfPGD
import L0_Sigma_PGD_AttackWrapper as L0SPGD

def main():

    modelDir="./models/model_test.pt"
    
    # Parameters for the dataset
    batchSize = 64
    numClasses = 10
    inputImageSize = [1, 3, 32, 32]

    # Define GPU device
    device = torch.device("cuda")

    #load model
    model = ResNet18().to(device)
    checkpoint = torch.load(modelDir)
    model.load_state_dict(checkpoint)
    model.eval()
    
    # Load CIFA10 Data
    valLoader = DMP.GetCIFAR10Validation(inputImageSize[2], batchSize)
    num_samples = len(valLoader.dataset)
    print(f"Total samples in valLoader: {num_samples}")
    
    # Check the clean accuracy of the model
    cleanAcc = DMP.validateD(valLoader, model, device)
    print("CIFAR-10 Clean Val Loader Acc:", cleanAcc)

    ###### Get correctly classified, classwise balanced samples to do the attack
    totalSamplesRequired = 500
    correctLoader = DMP.GetCorrectlyIdentifiedSamplesBalanced(model, totalSamplesRequired, valLoader, numClasses)

    ###### Check to make sure the accuracy is 100% on the correct loader
    correctAcc = DMP.validateD(correctLoader, model, device)
    print("Clean Accuracy for Correct Loader:", correctAcc)

    # Check the number of samples in the correctLoader
    print("Number of samples in correctLoader:", len(correctLoader.dataset))
  
    # Attack Paramaters
    n_restarts = 10
    num_steps = 20
    step_size=120000.0/255.0
    sparsity=10
    epsilon = 0.2
    kappa = 0.4
    
    # print("New_L0_PGD_Attack:\n ")
    # L0PGD.L0_PGD_AttackWrapper(model, device, correctLoader, n_restarts,  num_steps, step_size, sparsity)

    print("L0_Linf_PGD_AttackWrapper:\n ")
    L0LinfPGD.L0_Linf_PGD_AttackWrapper(model, device, correctLoader, n_restarts, num_steps, step_size, sparsity, epsilon)

    # print("L0_Sigma_PGD_AttackWrapper:\n ")
    # L0SPGD.L0_Sigma_PGD_AttackWrapper(model, device, correctLoader, n_restarts, num_steps, step_size, sparsity, kappa)

if __name__ == '__main__':
    main()