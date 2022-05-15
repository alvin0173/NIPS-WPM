import ray
import torch
import sys
import numpy as np
from torch import optim, nn

@ray.remote(num_cpus=1)
class Device(object):
    def __init__(self, device_index, args, model, train_loader, data_size):
        self.args = args
        self.data_size = data_size
        self.total_data_size = 0
        self.device_index = device_index
        if args.hardware_set == 'cpu':
            self.model = model
        elif args.hardware_set == 'cuda':
            self.model = model.cuda()
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.args.lr, momentum=0.9, weight_decay=5e-4)
        self.train_loader = train_loader
        self.data_iteration = iter(self.train_loader)
        if args.model == 'svm':
            self.criterion = nn.MultiMarginLoss()
        elif args.model == 'LR':
            self.criterion = nn.CrossEntropyLoss()
        elif args.model == 'LeNet':
            self.criterion = nn.CrossEntropyLoss()
        else:
            self.criterion = nn.MSELoss(reduction='mean')
        self.data_size_list = []
        self.weights_list = []
        #sys.stdout = open(f'{self.args.stdout}/{self.device_index:02}_stdout.log', 'a+', 1)
        #sys.stderr = open(f'{self.args.stdout}/{self.device_index:02}_stdout.log', 'a+', 1)

    def decentralized_train(self, now_device_index, epoch, current_weights):
        current_epoch = epoch
        self.model.set_weights(current_weights)
        self.model.train()
        try:
            inputs, targets = next(self.data_iteration)
        except StopIteration:
            self.data_iteration = iter(self.train_loader)
            inputs, targets = next(self.data_iteration)
        if self.args.hardware_set == 'cuda':
            inputs, targets = inputs.cuda(), targets.cuda()
        self.optimizer.zero_grad()
        
        #####################################
        if self.args.model == 'LeNet':
            outputs = self.model(inputs)
        elif self.args.model == 'LR':
            # reshape
            inputs = inputs.view(-1, 28 * 28)
            outputs = self.model(inputs)[0]
        ####################################

        loss = self.criterion(outputs, targets)
        loss.backward()
        self.optimizer.step()
        _, predicted = outputs.max(1)
        total = targets.size(0)
        correct = predicted.eq(targets).sum().item()
        print(
            f'DeviceIndex/SimulateIndex:{self.device_index}/{now_device_index} Training* loss:{loss.item()} | acc: {correct / total} | iter: {current_epoch}')
        return self.model.get_weights(), self.model.get_gradients()

    def decentralized_parallel_set_weights(self, weights):
        self.model.set_weights(weights)

    def decentralized_parallel_gradients_step(self, gradients):
        self.optimizer.zero_grad()
        self.model.set_gradients(gradients)
        self.optimizer.step()
        return self.model.get_weights()