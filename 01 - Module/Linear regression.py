# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 12:08:48 2024

@author: Verkholomov
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.close('all')

import torch
import random

from torch.utils.data import TensorDataset, DataLoader

torch.manual_seed(100)

num_inputs: int = 2
num_examples: int = 1000
true_w = torch.Tensor([2, -3.4])
true_b = 4.2
features = torch.randn((num_examples, num_inputs))
labels = torch.matmul(features, true_w) + true_b + torch.randn(num_examples)

fig = plt.figure(1, figsize=(12,4))
ax = fig.add_subplot(131)
ax.scatter(features[:, 0], labels, 1)
ax = fig.add_subplot(132)
ax.scatter(features[:, 1], labels, 1)
ax = fig.add_subplot(133)
ax.scatter(features[:, 0], features[:, 1], 1)

def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        j = indices[i : min(i + batch_size, num_examples)]
        yield features[j, :], labels[j]

batch_size = 10
for X,y in data_iter(batch_size, features, labels):
    print(X,'\n',y)
    break


w = torch.randn((num_inputs))
b = torch.zeros((1,))

w.requires_grad_()
b.requires_grad_()

def linregr(X, w, b):
    return torch.matmul(X,w) + b

def squared_loss(y_hat, y):
    #print(y_hat.shape)
    return ((y_hat-y.reshape(y_hat.shape))**2).mean()

def sgd(params, lr):
    
    for param in params:
        param.data[:] = param - lr*param.grad


lr = 0.01
num_epochs = 20
for epoch in range(num_epochs):
    for X,y in data_iter(batch_size, features, labels):
        w = w.detach()
        b = b.detach()
        w.requires_grad_()
        b.requires_grad_()
        y_hat = linregr(X,w,b)
        loss = squared_loss(y_hat, y)
        loss.backward()
        sgd((w,b), lr)
        
    train_loss = squared_loss(linregr(features, w, b), labels)
    print('epoch %d, loss %f' % (epoch + 1, train_loss.mean().item()))
    
    
    
    
dataset = TensorDataset(features, labels)
data_iter = DataLoader(dataset,batch_size=batch_size, shuffle=True)

for X,y in data_iter:
    print(X,y)
    break
    
model=torch.nn.Sequential(torch.nn.Linear(2,1))
#model[0].weight.data = true_w.clone().detach().requires_grad_(True).reshape((1,2))
#model[0].bias.data = torch.Tensor([true_b]).detach().requires_grad_(True)
   
loss = torch.nn.MSELoss(reduction='mean')

trainer = torch.optim.SGD(model.parameters(), lr=0.001)

num_epochs = 1000
for epoch in range(num_epochs):
    for X,y in data_iter:
        trainer.zero_grad()
        l = loss(model(X).reshape(-1),y)
        l.backward()
        trainer.step()
    l = loss(model(features).reshape(-1),labels)
    if epoch%10 == 0:
        print("Epoch {} training loss = {} weights = {} bias = {}".format(epoch,l,model[0].weight.data, model[0].bias.data))
    
    
    
    
    
    
    