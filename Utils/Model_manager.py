import types

from Models.CNN import CNN
from Models.LeNet import LeNet
from Models.ResNet import ResNet18
from Models.LeNet_5 import LENET
from Models.LogisticRegression import LogisticRegression, Cifar10LogisticRegression


def manager(obj):
    def get_weights(self):
        return {k: v.cpu() for k, v in self.state_dict().items() if 'weight' in k or 'bias' in k}

    def set_weights(self, weights):
        self.load_state_dict(weights, strict=False)
        
    def get_param(self):
        param = []
        for p in self.parameters():
            param.append(p.clone())
        return param
    
    def set_param(self, param):
        for p, m in zip(param, self.named_parameters()):
            m[1].data = p

    def get_gradients(self):
        grads = []
        for p in self.parameters():
            grad = None if p.grad is None else p.grad.data
            grads.append(grad)
        return grads

    def set_gradients(self, gradients):
        for g, p in zip(gradients, self.named_parameters()):
            if g is not None:
                p[1].grad = g

    obj.get_weights = types.MethodType(get_weights, obj)
    obj.set_weights = types.MethodType(set_weights, obj)
    
    obj.get_param = types.MethodType(get_param, obj)
    obj.set_param = types.MethodType(set_param, obj)
    
    obj.get_gradients = types.MethodType(get_gradients, obj)
    obj.set_gradients = types.MethodType(set_gradients, obj)
    return obj

def model_init(args):
    if args.model == 'LR' or args.model == 'svm':
        if args.dataset == 'mnist' or args.dataset == 'fmnist':
            return manager(LogisticRegression())
        elif args.dataset == 'cifar10':
            return manager(Cifar10LogisticRegression())
    elif args.model == 'CNN':
        return manager(CNN())
    elif args.model == 'resnet':
        return manager(ResNet18())
    elif args.model == 'LeNet':
        if args.dataset == 'mnist':
            return manager(LeNet())
        elif args.dataset == 'cifar10':
            return manager(LENET())

