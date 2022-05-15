import asyncio
import sys

import ray
import torch


@ray.remote(num_cpus=1)
class Server(object):
    def __init__(self, server_index, model, P_matrix, args, test_loader):
        self.device_connection_num = 0
        self.server_index = server_index
        self.model = model  # 初始化的 model 以及 agg_model
        self.P_matrix = P_matrix
        self.args = args
        self.total_data_size = 0
        self.test_loader = test_loader
        self.criterion = torch.nn.CrossEntropyLoss()
        self.sync_flag = False
        self.agg_flag = False
        self.mixed_flag = False
        self.data_size_list = []
        self.weights_list = []
        self.mixing_server_index = []
        self.mixing_server_weight = []
        self.peers = []
        #sys.stdout = open(f'{self.args.stdout}/ps{self.server_index:02}_stdout.log', 'a+', 1)
        #sys.stderr = open(f'{self.args.stdout}/ps{self.server_index:02}_stdout.log', 'a+', 1)

    def init_topology(self, *servers):
        self.peers = [peer for peer in servers]

    def reset(self):
        self.data_size_list.clear()
        self.weights_list.clear()
        self.mixing_server_index.clear()
        self.mixing_server_weight.clear()
        self.total_data_size = 0

    def transfer_device_to_server(self, weights, data_size):
        self.weights_list.append(weights)
        self.data_size_list.append(data_size)
        self.total_data_size += data_size

    def get_connection_num(self):
        return self.device_connection_num

    def transfer(self, weights, index):
        self.mixing_server_index.append(index)
        self.mixing_server_weight.append(weights)

    def wait_sync(self):
        return self.sync_flag

    def pull_weights(self):
        self.sync_flag = False
        return {k: v.cpu() for k, v in self.model.state_dict().items() if 'weight' in k or 'bias' in k}

    # Equation (2)
    def agg_ops(self, current_epoch, multi_server=True):
        w_new = []
        for index in range(len(self.data_size_list)):
            data_size = self.data_size_list[index]
            w = self.weights_list[index]
            for key in self.model.get_weights():
                w[key] = torch.mul(w[key], data_size / self.total_data_size)
            w_new.append(w)
        w_final = w_new[0]
        for w in w_new[1:]:
            for key in self.model.get_weights():
                w_final[key] = torch.add(w_final[key], w[key])
        for peer in self.peers:
            peer.transfer.remote(w_final, self.server_index)
        if not multi_server:
            self.model.set_weights(w_final)
            self.test(current_epoch)
            self.reset()
        return w_final, self.server_index

    # Equation (9)
    def mix_ops(self, local_agg_weights, server_index, current_epoch):
        for key in self.model.get_weights():
            local_agg_weights[key] = torch.mul(local_agg_weights[key],
                                               self.P_matrix[server_index][server_index])

        for i in range(len(self.mixing_server_weight)):
            w = self.mixing_server_weight[i]
            index = self.mixing_server_index[i]
            for key in self.model.get_weights():
                local_agg_weights[key] = torch.add(local_agg_weights[key],
                                                   torch.mul(w[key], self.P_matrix[server_index][index]))

        self.model.set_weights(local_agg_weights)
        test_loss = self.test(current_epoch)
        self.reset()
        return local_agg_weights, self.server_index, test_loss

    def ln_global_aggregation(self, weights, current_epoch):
        for key in self.model.get_weights():
            weights[key] = torch.add(torch.mul(weights[key], 0.5),
                                            torch.mul(self.model.get_weights()[key], 0.5))
        self.model.set_weights(weights)
        self.test(current_epoch)
        return weights

    def test(self, current_epoch):
        self.model.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(self.test_loader):
                if self.args.model == 'LR':
                    if self.args.dataset == 'mnist':
                        inputs = inputs.view(-1, 28 * 28)
                    elif self.args.dataset == 'cifar10':
                        inputs = inputs.view(-1, 32 * 32)
                    outputs = self.model(inputs)[0]
                else:
                    outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                test_loss += loss
                _, predicted = outputs.max(1)
                inner_total = targets.size(0)
                inner_correct = predicted.eq(targets).sum().item()
                total += inner_total
                correct += inner_correct
        print('Error Rate: %.4f%% (%d/%d) | epoch: %d | loss: %.2f' % (
            (1 - correct / total), correct, total, current_epoch, test_loss))
        return test_loss