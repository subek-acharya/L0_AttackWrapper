import numpy as np
import torch
import torch.nn as nn
from Utils import get_predictions, get_predictions_and_gradients, TensorToNumpy
from L0_Utils import perturb_L0_box, project_L0_box

def L0_Linf_PGD_AttackWrapper(model, device, dataLoader, n_restarts, num_steps, step_size, sparsity, epsilon):
    
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
        adv_batch, pgd_adv_acc_batch = perturb(model, device, x_numpy, y_numpy, sparsity, num_steps, step_size, n_restarts, epsilon)
        
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

def perturb(model, device, x_nat, y_nat, sparsity, num_steps, step_size, n_restarts, epsilon):
    adv = np.copy(x_nat)
    pgd_adv_acc = None 
      
    for counter in range(n_restarts):
        if counter == 0:
            corr_pred = get_predictions(model, x_nat, y_nat, device)
            pgd_adv_acc = np.copy(corr_pred)

        x_batch_adv, curr_pgd_adv_acc = perturb_L0_box(model, x_nat, y_nat, np.maximum(-epsilon, -x_nat), np.minimum(epsilon, 1.0 - x_nat), sparsity, num_steps, step_size, device)
        pgd_adv_acc = np.minimum(pgd_adv_acc, curr_pgd_adv_acc)
        adv[np.logical_not(curr_pgd_adv_acc)] = x_batch_adv[np.logical_not(curr_pgd_adv_acc)]

    return adv, pgd_adv_acc