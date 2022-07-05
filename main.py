import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import numpy as np
import os
import argparse
from preprocess_dataset import preprocess_fer
from torch.autograd import Variable
from model import VGG19
from tqdm import tqdm
import sys

parser = argparse.ArgumentParser(description='PyTorch Fer2013 CNN Training')
parser.add_argument('--dataset', type=str, default='FER2013', help='CNN architecture')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
opt = parser.parse_args()

best_PublicTest_acc = 0  
best_PublicTest_acc_epoch = 0
best_PrivateTest_acc = 0  
best_PrivateTest_acc_epoch = 0
start_epoch = 0  

learning_rate_decay_start = 80  # 50
learning_rate_decay_every = 5 # 5
learning_rate_decay_rate = 0.9 # 0.9
total_epoch = 250

#dataset
path = os.path.join(opt.dataset + '_' + 'VGG19')
trainloader = torch.utils.data.DataLoader(preprocess_fer(split='Training'), batch_size=128, shuffle=True, num_workers=2)
PublicTestloader = torch.utils.data.DataLoader(preprocess_fer(split = 'PublicTest'), batch_size=128, shuffle=False, num_workers=2)
PrivateTestloader = torch.utils.data.DataLoader(preprocess_fer(split = 'PrivateTest'), batch_size=128, shuffle=False, num_workers=2)
#model
net = VGG19().cuda()
#loss
criterion = nn.CrossEntropyLoss()
def set_lr(optimizer, lr):
    for group in optimizer.param_groups:
        group['lr'] = lr

def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        #print(group['params'])
        for param in group['params']:
            param.grad.data.clamp_(-grad_clip, grad_clip)
#train
def train(epoch):
    print('\nEpoch:', epoch + 1)
    optimizer = optim.SGD(net.parameters(), lr=opt.lr, momentum=0.9, weight_decay=5e-4)
    net.train()
    losses = []
    total = 0
    correct = 0
    if epoch > learning_rate_decay_start and learning_rate_decay_start >= 0:
        frac = (epoch - learning_rate_decay_start) // learning_rate_decay_every
        decay_factor = learning_rate_decay_rate**frac
        current_lr = opt.lr * decay_factor
        set_lr(optimizer, current_lr)
    else:
        current_lr = opt.lr
    print('Learning rate:', current_lr)
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        optimizer.zero_grad()
        inputs, targets = Variable(inputs.cuda()), Variable(targets.cuda())
        pred = net(inputs)
        loss = criterion(pred, targets)
        loss.backward()
        clip_gradient(optimizer, 0.1)
        optimizer.step()
        losses.append(loss.detach().cpu().item())
        _, predicted = torch.max(pred.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()
    print('Training loss:', np.round(np.mean(losses), 4))
    print('Training accuracy:', (100.0*correct/total).item())

def PublicTest(epoch):
    global PublicTest_acc
    global best_PublicTest_acc
    global best_PublicTest_acc_epoch
    #eval model on validation
    net.eval()
    losses = []
    correct = 0
    total = 0
    with torch.no_grad():
      for batch_idx, (inputs, targets) in enumerate(PublicTestloader):
          bs, ncrops, c, h, w = np.shape(inputs)
          inputs = inputs.view(-1, c, h, w)
          inputs, targets = Variable(inputs.cuda()), Variable(targets.cuda())
          pred = net(inputs)
          pred_avg = pred.view(bs, ncrops, -1).mean(1)  # avg over crops
          loss = criterion(pred_avg, targets)
          losses.append(loss.detach().cpu().item())
          _, predicted = torch.max(pred_avg.data, 1)
          total += targets.size(0)
          correct += predicted.eq(targets.data).cpu().sum()
    PublicTest_acc = 100.0*correct/total
    print('Validation loss:', np.round(np.mean(losses), 4))
    print('Validation accuracy:', PublicTest_acc.item())
    #save best validation model
    if PublicTest_acc > best_PublicTest_acc:
        print('Saving best validation model with accuracy:', PublicTest_acc.item())
        state = {'net': net.state_dict(), 'acc': PublicTest_acc, 'epoch': epoch}
        if not os.path.isdir(path):
            os.mkdir(path)
        torch.save(state, os.path.join(path, 'PublicTest_model.t7'))
        best_PublicTest_acc = PublicTest_acc
        best_PublicTest_acc_epoch = epoch

def PrivateTest(epoch):
    global PrivateTest_acc
    global best_PrivateTest_acc
    global best_PrivateTest_acc_epoch
    #eval model on test
    net.eval()
    losses = []
    correct = 0
    total = 0
    with torch.no_grad():
      for batch_idx, (inputs, targets) in enumerate(PrivateTestloader):
          bs, ncrops, c, h, w = np.shape(inputs)
          inputs = inputs.view(-1, c, h, w)
          inputs, targets = Variable(inputs.cuda()), Variable(targets.cuda())
          pred = net(inputs)
          pred_avg = pred.view(bs, ncrops, -1).mean(1)  # avg over crops
          loss = criterion(pred_avg, targets)
          losses.append(loss.detach().cpu().item())
          _, predicted = torch.max(pred_avg.data, 1)
          total += targets.size(0)
          correct += predicted.eq(targets.data).cpu().sum()
    PrivateTest_acc = 100.*correct/total
    print('Testing loss:', np.round(np.mean(losses), 4))
    print('Testing accuracy:', PrivateTest_acc.item())
    #save best test model
    if PrivateTest_acc > best_PrivateTest_acc:
        print('Saving best test model with accuracy:', PrivateTest_acc.item())
        state = {'net': net.state_dict(), 'best_PublicTest_acc': best_PublicTest_acc,
                 'best_PrivateTest_acc': PrivateTest_acc, 'best_PublicTest_acc_epoch': best_PublicTest_acc_epoch,
                 'best_PrivateTest_acc_epoch': epoch}
        if not os.path.isdir(path):
            os.mkdir(path)
        torch.save(state, os.path.join(path, 'PrivateTest_model.t7'))
        best_PrivateTest_acc = PrivateTest_acc
        best_PrivateTest_acc_epoch = epoch

for epoch in range(start_epoch, total_epoch):
    train(epoch)
    PublicTest(epoch)
    PrivateTest(epoch)

print("best_PublicTest_acc: %0.3f" % best_PublicTest_acc)
print("best_PublicTest_acc_epoch: %d" % best_PublicTest_acc_epoch)
print("best_PrivateTest_acc: %0.3f" % best_PrivateTest_acc)
print("best_PrivateTest_acc_epoch: %d" % best_PrivateTest_acc_epoch)