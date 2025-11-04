import numpy as np
import torch
import torch.nn as nn
from Utils import get_predictions, get_predictions_and_gradients, TensorToNumpy

def L0_Sigma_PGD_AttackWrapper(model, device, dataLoader, n_restarts, num_steps, step_size, sparsity, kappa):
    
    model.eval()
    
    all_adv_examples = []
    all_original_examples = []
    all_labels = []
    all_robust_acc = []
    
    total_batches = len(dataLoader)
    total_samples = len(dataLoader.dataset)
    
    # Process each batch from dataloader
    for batch_idx, (x_batch, y_batch) in enumerate(dataLoader):
        batch_num = batch_idx + 1
        current_batch_size = x_batch.size(0)
 
        # Convert batch tensors to numpy format (NHWC)
        x_numpy, y_numpy = TensorToNumpy(x_batch, y_batch)
        
        # Perform attack on this batch
        adv_batch, pgd_adv_acc_batch = perturb(model, device, x_numpy, y_numpy, sparsity, num_steps, step_size, n_restarts, kappa)
        
        # Store results
        all_adv_examples.append(adv_batch)
        all_original_examples.append(x_numpy)
        all_labels.append(y_numpy)
        all_robust_acc.append(pgd_adv_acc_batch)
    
    # Concatenate all results
    all_adv_examples = np.concatenate(all_adv_examples, axis=0)
    all_original_examples = np.concatenate(all_original_examples, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    all_robust_acc = np.concatenate(all_robust_acc, axis=0)
    
    # Calculate overall statistics
    overall_robust_acc = np.sum(all_robust_acc) / total_samples * 100.0
    
    # Calculate pixels changed statistics
    pixels_changed = np.sum(np.amax(np.abs(all_adv_examples - all_original_examples) > 1e-10, axis=-1), axis=(1,2))
    
    # Calculate maximum perturbation across all samples
    max_perturbation = np.amax(np.abs(all_adv_examples - all_original_examples))
    
    print(f"{'='*70}")
    print(f"Total samples processed: {total_samples}")
    print(f"Overall Robust Accuracy at {sparsity} pixels: {overall_robust_acc:.2f}%")
    print(f"Samples correctly classified after attack: {np.sum(all_robust_acc)}")
    print(f"Maximum perturbation size: {max_perturbation:.5f}")
    print(f"{'='*70}\n")

def perturb(model, device, x_nat, y_nat, sparsity, num_steps, step_size, n_restarts, kappa):
    adv = np.copy(x_nat)
    sigma = sigma_map(x_nat)
    pgd_adv_acc = None

    for counter in range(n_restarts):
        if counter == 0:
            corr_pred = get_predictions(model, x_nat, y_nat, device)
            pgd_adv_acc = np.copy(corr_pred)

        x_batch_adv, curr_pgd_adv_acc = perturb_L0_sigma(model, x_nat, y_nat, sparsity, num_steps, step_size, device, sigma, kappa)

        pgd_adv_acc = np.minimum(pgd_adv_acc, curr_pgd_adv_acc)
        adv[np.logical_not(curr_pgd_adv_acc)] = x_batch_adv[np.logical_not(curr_pgd_adv_acc)]
        
    return adv, pgd_adv_acc

def sigma_map(x):
    ''' creates the sigma-map for the batch x '''
    sh = [4]
    sh.extend(x.shape)
    t = np.zeros(sh)
    t[0,:,:-1] = x[:,1:]
    t[0,:,-1] = x[:,-1]
    t[1,:,1:] = x[:,:-1]
    t[1,:,0] = x[:,0]
    t[2,:,:,:-1] = x[:,:,1:]
    t[2,:,:,-1] = x[:,:,-1]
    t[3,:,:,1:] = x[:,:,:-1]
    t[3,:,:,0] = x[:,:,0]

    mean1 = (t[0] + x + t[1]) / 3
    sd1 = np.sqrt(((t[0] - mean1) ** 2 + (x - mean1) ** 2 + (t[1] - mean1) ** 2) / 3)

    mean2 = (t[2] + x + t[3]) / 3
    sd2 = np.sqrt(((t[2] - mean2) ** 2 + (x - mean2) ** 2 + (t[3] - mean2) ** 2) / 3)

    sd = np.minimum(sd1, sd2)
    sd = np.sqrt(sd)

    return sd

def perturb_L0_sigma(model, x_nat, y_nat, sparsity, num_steps, step_size, device, sigma, kappa):
    rs = True
    # np.random.seed(42)
    if rs == True:     # This is for the random start
        x2 = x_nat + np.random.uniform(-kappa, kappa, x_nat.shape)
        x2 = np.clip(x2, 0, 1)
    else:
        x2 = np.copy(x_nat)
    adv_not_found = np.ones(y_nat.shape)
    adv = np.zeros(x_nat.shape)

    for i in range(num_steps):
        if i > 0:
            pred, grad = get_predictions_and_gradients(model, x2, y_nat, device)
            adv_not_found = np.minimum(adv_not_found, pred.astype(int))
            adv[np.logical_not(pred)] = np.copy(x2[np.logical_not(pred)])

            grad /= (1e-10 + np.sum(np.abs(grad), axis=(1,2,3), keepdims=True))
            x2 = np.add(x2, (np.random.random_sample(grad.shape) - 0.5) * 1e-12 + step_size * grad, casting='unsafe')

        x2 = project_L0_sigma(x2, sparsity, sigma, kappa, x_nat)

    return adv, adv_not_found

def project_L0_sigma(y, k, sigma, kappa, x_nat):
    ''' projection of the batch y to a batch x such that:
        - 0 <= x <= 1
        - each image of the batch x differs from the corresponding one of
          x_nat in at most k pixels
        - (1 - kappa*sigma)*x_nat <= x <= (1 + kappa*sigma)*x_nat '''
    
    
    x = np.copy(y)
    p1 = 1.0 / np.maximum(1e-12, sigma) * (x_nat > 0).astype(float) + 1e12 * (x_nat == 0).astype(float)
    p2 = 1.0 / np.maximum(1e-12, sigma) * (1.0 / np.maximum(1e-12, x_nat) - 1) * (x_nat > 0).astype(float) + \
         1e12 * (x_nat == 0).astype(float) + 1e12 * (sigma == 0).astype(float)
    lmbd_l = np.maximum(-kappa, np.amax(-p1, axis=-1, keepdims=True))
    lmbd_u = np.minimum(kappa, np.amin(p2, axis=-1, keepdims=True))
    
    lmbd_unconstr = np.sum((y - x_nat) * sigma * x_nat, axis=-1, keepdims=True) / \
                    np.maximum(1e-12, np.sum((sigma * x_nat) ** 2, axis=-1, keepdims=True))
    lmbd = np.maximum(lmbd_l, np.minimum(lmbd_unconstr, lmbd_u))
    
    p12 = np.sum((y - x_nat) ** 2, axis=-1, keepdims=True)
    p22 = np.sum((y - (1 + lmbd * sigma) * x_nat) ** 2, axis=-1, keepdims=True)
    p3 = np.sort(np.reshape(p12 - p22, [x.shape[0], -1]))[:, -k]
    
    x = x_nat + lmbd * sigma * x_nat * ((p12 - p22) >= p3.reshape([-1, 1, 1, 1]))
    
    return x