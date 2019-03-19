import math
import torch
from torch.optim.optimizer import Optimizer
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
class Adam(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0):
        # Invalid input parameters raise error
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay)
        super(Adam, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(Adam, self).__setstate__(state)

    def step(self):
        loss = None
        for group in self.param_groups:
            # Actually, only one group
            # group = {'params': ...,
            #          'lr': 0.001,
            #          'betas': (0.9, 0.999),
            #          ...}
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p] # initially, state = {}
                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad.add_(group['weight_decay'], p.data)

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                denom = exp_avg_sq.sqrt().add_(group['eps'])

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

                p.data.addcdiv_(-step_size, exp_avg, denom)

        return loss
def num_flat_features(x):
    size = x.size()[1:]  # all dimensions except the batch dimension
    num_features = 1
    for s in size:
        num_features *= s
    return num_features
class FC(nn.Module):
    def __init__(self, activation="relu", drop_rate=0):
        super(FC, self).__init__()
        self.activation = activation
        self.drop_rate = drop_rate
        self.type = 'fc'
        self.fc1 = nn.Linear(784, 1000)
        self.fc2 = nn.Linear(1000, 1000)
        self.fc3 = nn.Linear(1000, 10)
        self.activations = nn.ModuleDict([
            ['relu', nn.ReLU()],
            ['sigmoid', nn.Sigmoid()]
        ])
        if self.drop_rate:
            self.input_dropout = nn.Dropout(0.2)
            self.dropout = nn.Dropout(self.drop_rate)

    def forward(self, x):
        x = x.view(-1, num_flat_features(x)) #flattened input 64 x 64 = 784
        if self.drop_rate:
            x = self.input_dropout(x)
        x = self.activations[self.activation](self.fc1(x))
        if self.drop_rate:
            x = self.dropout(x)
        x = self.activations[self.activation](self.fc2(x))
        if self.drop_rate:
            x = self.dropout(x)
        x = self.activations[self.activation](self.fc3(x))
        return F.log_softmax(x, dim=1)

if __name__ == '__main__':
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    dataset = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
                ])),
    batch_size=128, shuffle=True, **kwargs)
    model = FC().to(device)
    optimizer = Adam(model.parameters())
    for input, target in dataset:
        input, target = input.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(input)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        break
