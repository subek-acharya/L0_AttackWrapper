import numpy as np
import torch
from Utils import get_predictions_and_gradients

def perturb_L0_box(model, x_nat, y_nat, lb, ub, sparsity, num_steps, step_size, device):
  rs = True
  # np.random.seed(42)
  if rs == True:   
    x2 = x_nat + np.random.uniform(lb, ub, x_nat.shape)
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
      x2 = np.add(x2, (np.random.random_sample(grad.shape)-0.5)*1e-12 + step_size * grad, casting='unsafe')
    x2 = x_nat + project_L0_box(x2 - x_nat, sparsity, lb, ub)
    
  return adv, adv_not_found

def project_L0_box(y, k, lb, ub):
  x = np.copy(y)
  p1 = np.sum(x**2, axis=-1)
  # print("p1.shape: ", p1.shape) 
  p2 = np.minimum(np.minimum(ub - x, x - lb), 0)
  p2 = np.sum(p2**2, axis=-1)
  # print("p2 shape: ", p2.shape)
  p3 = np.sort(np.reshape(p1-p2, [p2.shape[0],-1]))[:,-k]
  x = x*(np.logical_and(lb <=x, x <= ub)) + lb*(lb > x) + ub*(x > ub)
  x *= np.expand_dims((p1 - p2) >= p3.reshape([-1, 1, 1]), -1)
  return x
