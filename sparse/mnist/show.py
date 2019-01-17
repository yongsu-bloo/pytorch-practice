from __future__ import print_function
import argparse
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
import torchvision.utils as vutils
from tensorboardX import SummaryWriter
from neuron_mnist import Net

def num_flat_features(x):
    size = x.size()[1:]  # all dimensions except the batch dimension
    num_features = 1
    for s in size:
        num_features *= s
    return num_features

def main():
    # Training settings
    device = torch.device("cuda")
    # load and accuracy comparison
    # acc_dict = {}
    # time_dict = {}
    for activation in ['soft_plus', 'relu', 'prelu']:
        # acc_dict[activation] = {}
        # time_dict[activation] = {}
        for reg_type in ["base", "L1", "L2"]:
            # acc_list=[]
            # time_list=[]
            with SummaryWriter(comment= activation + reg_type) as w:
                # for manualSeed in [2]:
                for manualSeed in [1*(i+1) for i in range(9)]:
                    # print("\nRandom Seed\n: ", manualSeed)
                    # checkpoint loading
                    model = Net(activation=activation, reg_type=reg_type).to(device)
                    PATH = "./n_saved_models/{}_{}-{}_{}-{}".format(
                        35000,
                        model.fc2.weight.size()[1], model.fc2.weight.size()[0],
                        model.lambda1, model.lambda2)
                    PATH += "/{}_{}_{}.pt".format(activation, reg_type, manualSeed)
                    checkpoint = torch.load(PATH)
                    args = checkpoint['args']
                    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
                    model.load_state_dict(checkpoint['model_state_dict'])
                    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    # losses = checkpoint['losses']
                    accuracies = checkpoint['accuracies']
                    # times = checkpoint["times"]
                    # checkpoint load end

                    # acc_list.append(accuracies)
                    # time_list.append(times)
                    # acc_dict[activation][reg_type] = np.mean(acc_list, axis=0)
                    # time_dict[activation][reg_type] = np.mean(time_list, axis=0)
                    kwargs = {'num_workers': 1, 'pin_memory': True}
                    test_loader = torch.utils.data.DataLoader(
                        datasets.MNIST('../data', train=False, transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.1307,), (0.3081,))
                        ])),
                        batch_size=args.test_batch_size, shuffle=True, **kwargs)
                        # print(model)
                        # writing summary
                    model.eval()
                    num_zeros = torch.zeros(3)
                    test_loss = 0
                    correct = 0
                    for i, batch in enumerate(test_loader):
                        data, target = batch[0].to(device), batch[1].to(device)
                        # histogram
                        output = model(data)
                        test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
                        pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
                        correct += pred.eq(target.view_as(pred)).sum().item()
                        test_loss /= len(test_loader.dataset)
                        accuracy = 100. * correct / len(test_loader.dataset)
                        if i == 9:
                            print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
                                test_loss, correct, len(test_loader.dataset),
                                accuracy))

                        data = data.view(-1, num_flat_features(data)).to(device)
                        after_fc1 = model.activations[activation](model.fc1(data))
                        after_fc2 = model.activations[activation](model.fc2(after_fc1))
                        after_fc3 = model.activations[activation](model.fc3(after_fc2))
                        num_zeros[0] += torch.sum(after_fc1.eq(torch.zeros(after_fc1.size()).to(device)),dtype=torch.int)
                        num_zeros[1] += torch.sum(after_fc2.eq(torch.zeros(after_fc2.size()).to(device)),dtype=torch.int)
                        num_zeros[2] += torch.sum(after_fc3.eq(torch.zeros(after_fc3.size()).to(device)),dtype=torch.int)
                        # if (activation == "relu") and (reg_type=="elastic"):
                        #     print(after_fc1)
                        # num_zeros[0] += torch.sum(torch.abs(after_fc1).lt(0.001))
                        # num_zeros[1] += torch.sum(torch.abs(after_fc2).lt(0.001))
                        # num_zeros[2] += torch.sum(torch.abs(after_fc3).lt(0.001))
                        w.add_histogram("fc1", after_fc1.clone().cpu().data.numpy(), i)
                        w.add_histogram("fc2", after_fc2.clone().cpu().data.numpy(), i)
                        w.add_histogram("fc3", after_fc3.clone().cpu().data.numpy(), i)
                    num_zeros = num_zeros.div(10000).round()
                    print("#Dead activations of {}-{}: {:d}-{:d}-{:d}".format( \
                        activation, reg_type, \
                        int(num_zeros[0].data),int(num_zeros[1].data),int(num_zeros[2].data)))

if __name__ == '__main__':
    main()
