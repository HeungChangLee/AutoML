import os
import sys
import time
import glob
import numpy as np
import torch
import utils
import logging
import argparse
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
import pickle
import time
import pandas as pd
import random

from torch.autograd import Variable
from model_search import Network
from architect import Architect

from genotypes import PRIMITIVES
from genotypes import Genotype

parser = argparse.ArgumentParser("cifar")
parser.add_argument('--data', type=str, default='../data', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=128, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.05, help='init learning rate')
parser.add_argument('--learning_rate_min', type=float, default=0.001, help='min learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=300, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=16, help='num of init channels')
parser.add_argument('--layers', type=int, default=8, help='total number of layers')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.3, help='drop path probability')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--seed', type=int, default=1, help='random seed')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--train_portion', type=float, default=0.8, help='portion of training data')
parser.add_argument('--unrolled', action='store_true', default=False, help='use one-step unrolled validation loss')
parser.add_argument('--arch_learning_rate', type=float, default=3.5e-4, help='learning rate for arch encoding')
parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')
#Added
parser.add_argument('--max_hidden_node_num', type=int, default=1, help='max number of hidden nodes in a cell')
args = parser.parse_args()

args.save = 'search-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)


CIFAR_CLASSES = 10
model_path = 'search-RL-100iter/weights.pt'
node_num = 4
search_epoch = 1 #35*100/2
search_arch_num = 1000

def w_parse(weights):
  gene = []
  n = 2
  start = 0
  for i in range(node_num):      
    end = start + n
    W = weights[start:end].copy()
    edges = sorted(range(i + 2), key=lambda x: -max(W[x][k] for k in range(len(W[x]))))[:2]
    for j in edges:
      k_best = None
      for k in range(len(W[j])):        
        if k_best is None or W[j][k] > W[j][k_best]:
          k_best = k
      gene.append((PRIMITIVES[k_best], j))
    start = end
    n += 1
  return gene

def main(seed):
  if not torch.cuda.is_available():
    logging.info('no gpu device available')
    sys.exit(1)

  np.random.seed(seed)
  torch.cuda.set_device(args.gpu)
  cudnn.benchmark = True
  torch.manual_seed(seed)
  cudnn.enabled=True
  torch.cuda.manual_seed(seed)
  logging.info('gpu device = %d' % args.gpu)
  logging.info("args = %s", args)

  criterion = nn.CrossEntropyLoss()
  criterion = criterion.cuda()
  model = Network(args.init_channels, CIFAR_CLASSES, args.layers, criterion)
  model = model.cuda()
  logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

  optimizer = torch.optim.SGD(
      model.parameters(),
      args.learning_rate,
      momentum=args.momentum,
      weight_decay=args.weight_decay,
      nesterov=True)

  train_transform, valid_transform = utils._data_transforms_cifar10(args)
  train_data = dset.CIFAR10(root=args.data, train=True, download=True, transform=train_transform)

  num_train = len(train_data)
  indices = list(range(num_train))
  split = int(np.floor(args.train_portion * num_train))
  
  batch_size = args.batch_size
  train_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices[:split])
  valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train])
  train_queue = torch.utils.data.DataLoader(
      train_data, batch_size=batch_size,
      sampler=train_sampler,
      pin_memory=False, num_workers=0)

  valid_queue = torch.utils.data.DataLoader(
      train_data, batch_size=batch_size,
      sampler=valid_sampler,
      pin_memory=False, num_workers=0)

  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, float(args.epochs), eta_min=args.learning_rate_min)

  architect = Architect(model, args)
  model.load_state_dict(torch.load(model_path))
  
  start_time = time.time()
  best_arch_search(valid_queue, model)
  print(model)
'''
  for epoch in range(args.epochs):
    scheduler.step()
    lr = scheduler.get_lr()[0]
    logging.info('epoch %d lr %e', epoch, lr)

    #genotype = model.genotype()
    #logging.info('genotype = %s', genotype)

    #print(model.alphas_normal)
    #print(model.alphas_reduce)
    #normal_weights, reduce_weights = model._compute_weight_from_alphas()
    #normal_weights = torch.from_numpy(normal_weights)
    #reduce_weights = torch.from_numpy(reduce_weights)
    #print(normal_weights)
    #print(reduce_weights)

    # training
    train_acc, train_obj = train(train_queue, valid_queue, model, criterion, optimizer, lr)
    logging.info('train_acc %f', train_acc)

    valid_acc = train_arch(train_queue, valid_queue, model, architect, criterion, optimizer, lr)
    logging.info('valid_acc %f', valid_acc)
    
    # validation
    #if (epoch+1) % 1 == 0:
      #valid_acc, valid_obj = infer(valid_queue, model, criterion)
      #logging.info('valid_acc %f', valid_acc)

    utils.save(model, os.path.join(args.save, 'weights.pt'))
'''

  #best_arch_search(valid_queue, model)
    
'''
  k = sum(1 for i in range(node_num) for n in range(2+i))
  num_ops = len(PRIMITIVES)
  acc_mat_normal = np.zeros((k, num_ops))
  cnt_mat_normal = np.zeros((k, num_ops))
  acc_mat_reduce = np.zeros((k, num_ops))
  cnt_mat_reduce = np.zeros((k, num_ops))
  np.set_printoptions(precision=3)
  #for epoch in range(search_epoch):
  nop = True
  if nop:
    print ('Search epoch: ', epoch)
    #model.generate_random_alphas()
    model.generate_fixed_alphas()
    valid_acc = infer_for_search(valid_queue, model)
      
    #valid_acc = valid_acc.data.cpu().numpy()
    
    normal_gene = model.alphas_normal.data.cpu().numpy()
    cnt_mat_normal = cnt_mat_normal + normal_gene
    acc_mat_normal = acc_mat_normal + normal_gene * valid_acc
          
    reduce_gene = model.alphas_reduce.data.cpu().numpy()
    cnt_mat_reduce = cnt_mat_reduce + reduce_gene
    acc_mat_reduce = acc_mat_reduce + reduce_gene * valid_acc
    
    avg_acc_normal = np.divide(acc_mat_normal, cnt_mat_normal)
    avg_acc_normal[np.isnan(avg_acc_normal)] = 0
    avg_acc_normal[np.isinf(avg_acc_normal)] = 0
    
    avg_acc_reduce = np.divide(acc_mat_reduce, cnt_mat_reduce)
    avg_acc_reduce[np.isnan(avg_acc_reduce)] = 0
    avg_acc_reduce[np.isinf(avg_acc_reduce)] = 0
    
    if (epoch+1) % 1 == 0:
        print ('Avg accuracies')
        print (avg_acc_normal)
        print (avg_acc_reduce, '\n')

        gene_normal = w_parse(avg_acc_normal)
        gene_reduce = w_parse(avg_acc_reduce)
        print ('Normal genotype: ', gene_normal)
        print ('Reduce genotype: ', gene_reduce)
        elapsed_time = time.time() - start_time
        elapsed_time = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
        print('Elapsed time: ', elapsed_time, '\n')
    
  #with open(os.path.join(args.save, 'search_results.pkl'), 'wb') as f:    
      #pickle.dump((acc_mat_normal, cnt_mat_normal, acc_mat_reduce, cnt_mat_reduce), f)  '''

def train(train_queue, valid_queue, model, criterion, optimizer, lr):
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  top5 = utils.AvgrageMeter()

  for step, (input, target) in enumerate(train_queue):
    model.train()
    n = input.size(0)

    input = Variable(input, requires_grad=False).cuda()
    target = Variable(target, requires_grad=False).cuda(async=True)

    # get a random minibatch from the search queue with replacement
    #input_search, target_search = next(iter(valid_queue))
    #input_search = Variable(input_search, requires_grad=False).cuda()
    #target_search = Variable(target_search, requires_grad=False).cuda(async=True)

    #architect.step(input, target, input_search, target_search, lr, optimizer, unrolled=args.unrolled)

    optimizer.zero_grad()
    #model.generate_droppath_alphas()
    model.sample_new_architecture()
    #print(model.sampled_weight_normal)
    #print(model.sampled_weight_reduce)    
    logits = model(input)
    loss = criterion(logits, target)

    loss.backward()
    nn.utils.clip_grad_norm(model.parameters(), args.grad_clip)
    optimizer.step()

    prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
    objs.update(loss.data[0], n)
    top1.update(prec1.data[0], n)
    top5.update(prec5.data[0], n)

    if step % args.report_freq == 0:
      logging.info('train %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)
    #if step >= 50:
      #break

  return top1.avg, objs.avg

def train_arch(train_queue, valid_queue, model, architect, criterion, optimizer, lr):    
  top1 = utils.AvgrageMeter()  
  for step, (input, target) in enumerate(valid_queue):
    model.eval()
    n = input.size(0)

    input = Variable(input, requires_grad=False).cuda()
    target = Variable(target, requires_grad=False).cuda(async=True)

    # get a random minibatch from the search queue with replacement    
    model.sample_new_architecture() 
    if step == 0:
      for e in model.edge_alphas_normal:
        print(F.softmax(e, dim=-1))
      for e in model.edge_alphas_reduce:
        print(F.softmax(e, dim=-1))
      print(F.softmax(model.alphas_normal, dim=-1))
      print(F.softmax(model.alphas_reduce, dim=-1))
      print(architect.baseline)
    logits = model(input)
    architect.step(input, target, lr, optimizer, unrolled=args.unrolled) 
    prec1, _ = utils.accuracy(logits, target, topk=(1, 5))
    n = input.size(0)    
    top1.update(prec1.data[0], n)
    
  return top1.avg
    

def infer(valid_queue, model, criterion):
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  top5 = utils.AvgrageMeter()
  model.eval()    
  
  #with torch.no_grad():
  for step, (input, target) in enumerate(valid_queue):      
    input = Variable(input).cuda()
    target = Variable(target).cuda(async=True)

    #model.generate_random_alphas()
    #model.generate_droppath_alphas()
    model.sample_new_architecture()
    logits = model(input)
    loss = criterion(logits, target)

    prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
    n = input.size(0)
    objs.update(loss.data[0], n)
    top1.update(prec1.data[0], n)
    top5.update(prec5.data[0], n)    

    if step % args.report_freq == 0:
      logging.info('valid %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)
      #if step >= 50:
        #break    

  return top1.avg, objs.avg

def infer_for_search(valid_queue, model):    
  top1 = utils.AvgrageMeter()  
  model.eval()    
  
  #with torch.no_grad():
  for step, (input, target) in enumerate(valid_queue):      
    input = Variable(input).cuda()
    target = Variable(target).cuda(async=True)
      
    #model.generate_random_alphas()
    logits = model(input)      
  
    prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
    n = input.size(0)    
    top1.update(prec1.data[0], n)            
      
      #if step >= 50:
        #break    

  return top1.avg

def best_arch_search(valid_queue, model):
    model.eval()
    steps = 4
    k = sum(1 for i in range(steps) for n in range(2 + i))
    num_ops = len(PRIMITIVES)

    avg_count_normal = np.zeros((k, num_ops))
    avg_count_reduce = np.zeros((k, num_ops))
    avg_weight_normal = np.zeros((k, num_ops))
    avg_weight_reduce = np.zeros((k, num_ops))
    result_df = pd.DataFrame(columns = ['Normal', 'Reduce', 'Val_acc'])
    for i in range(search_arch_num):
      input_search, target_search = next(iter(valid_queue))
      if input_search.size(0) != args.batch_size:
        input_search, target_search = next(iter(valid_queue))
      input_search = Variable(input_search, requires_grad=False).cuda()
      target_search = Variable(target_search, requires_grad=False).cuda(async=True)
      
      model.sample_new_architecture()
      logits = model(input_search)
    
      prec1, _ = utils.accuracy(logits, target_search, topk=(1, 5))



      tmp_row_normal = np.where(model.sampled_weight_normal.data.cpu().numpy() == 1)[0]
      tmp_col_normal = np.where(model.sampled_weight_normal.data.cpu().numpy() == 1)[1]
      tmp_row_reduce = np.where(model.sampled_weight_reduce.data.cpu().numpy() == 1)[0]
      tmp_col_reduce = np.where(model.sampled_weight_reduce.data.cpu().numpy() == 1)[1]
      for j in range(len(tmp_row_normal)):
        tmp_acc = prec1.data[0]
        avg_count_normal[tmp_row_normal[j], tmp_col_normal[j]] += 1
        avg_count_reduce[tmp_row_reduce[j], tmp_col_reduce[j]] += 1

        avg_weight_normal[tmp_row_normal[j],tmp_col_normal[j]] += tmp_acc
        avg_weight_reduce[tmp_row_reduce[j],tmp_col_reduce[j]] += tmp_acc

      #gene_normal = w_parse(model.sampled_weight_normal.data.cpu().numpy())
      #gene_reduce = w_parse(model.sampled_weight_reduce.data.cpu().numpy())
      #temp_df = pd.DataFrame([[gene_normal, gene_reduce, prec1.data[0]]], columns = ['Normal', 'Reduce', 'Val_acc'])
      #result_df = result_df.append(temp_df, ignore_index=True)
    avg_weight_normal = avg_weight_normal / avg_count_normal
    avg_weight_reduce = avg_weight_reduce / avg_count_reduce

    gene_normal = w_parse(avg_weight_normal)
    gene_reduce = w_parse(avg_weight_reduce)
    temp_df = pd.DataFrame([[gene_normal, gene_reduce, prec1.data[0]]], columns=['Normal', 'Reduce', 'Val_acc'])
    result_df = result_df.append(temp_df, ignore_index=True)

    result_df = result_df.sort_values(by='Val_acc', ascending=False)
    result_df.to_csv('search_result.csv')
    

if __name__ == '__main__':
  main(2)