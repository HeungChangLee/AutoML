import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable


class Architect(object):

  def __init__(self, model, args):
    self.network_momentum = args.momentum
    self.network_weight_decay = args.weight_decay
    self.model = model
    self.optimizer = torch.optim.Adam(self.model.arch_parameters(),
        lr=args.arch_learning_rate, betas=(0.5, 0.999), weight_decay=args.arch_weight_decay)
    self.lr = args.arch_learning_rate    

  def step(self, input_valid, target_valid, is_opt_step):
    if is_opt_step:
        self.optimizer.zero_grad()
    
    self._backward_step_policy_rl(input_valid, target_valid)
    
    if is_opt_step:
        self.optimizer.step()

  def _backward_step_policy_rl(self, input_valid, target_valid):
    logits = self.model(input_valid)    
    batch_size = target_valid.size(0)
  
    _, pred = logits.topk(1, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target_valid.view(1, -1).expand_as(pred))
    correct_k = correct[0].view(-1).float().sum(0)
    reward = correct_k.mul_(1.0/batch_size)
    #print (reward)
    sample_log_probs = self.model.sample_log_probs
    
    self.model.baseline = self.model.baseline - (1 - self.model.bl_dec) * (self.model.baseline - reward)
    loss = sample_log_probs * (reward - self.model.baseline)
    loss.backward()
  
  def reset_optimizer(self):
    self.optimizer = torch.optim.Adam(self.model.arch_parameters(),
        lr=self.lr, betas=(0.5, 0.999), weight_decay=self.network_weight_decay)    

