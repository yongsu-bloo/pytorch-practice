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
from mnist import Net

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
    for activation in ['relu', 'lrelu', 'prelu']:
        # acc_dict[activation] = {}
        # time_dict[activation] = {}
        for reg_type in ["base", "L1", "L2",'elastic']:
            # acc_list=[]
            # time_list=[]
            for manualSeed in [111*(i+1) for i in range(9)]:
                # print("\nRandom Seed\n: ", manualSeed)
                # checkpoint loading
                PATH = "./neu_saved_models/{}_{}_{}.pt".format(activation, reg_type, manualSeed)
                checkpoint = torch.load(PATH)
                args = checkpoint['args']
                model = Net(activation=activation, reg_type=reg_type).to(device)
                optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
                model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                losses = checkpoint['losses']
                accuracies = checkpoint['accuracies']
                # times = checkpoint["times"]
                # checkpoint load end
            # if (reg_type == "base") and (activation=="relu"):
            #     for param_tensor in model.state_dict():
            #         print(param_tensor, "\t", model.state_dict()[param_tensor])

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
                model.eval()
                # print(model)
                # writing summary
            with SummaryWriter(comment= activation + reg_type) as w:
                num_zeros = torch.zeros(3)
                for i, batch in enumerate(test_loader):
                    data, target = batch[0].to(device), batch[1].to(device)
                    # histogram
                    # for name, param in model.named_parameters():
                    #     w.add_histogram(name, param.clone().cpu().data.numpy(), i)
                    data = data.view(-1, num_flat_features(data)).to(device)
                    after_fc1 = model.activations[activation](model.fc1(data))
                    after_fc2 = model.activations[activation](model.fc2(after_fc1))
                    after_fc3 = model.activations[activation](model.fc3(after_fc2))
                    num_zeros[0] += torch.sum(after_fc1.eq(torch.zeros(after_fc1.size()).to(device)),dtype=torch.int)
                    num_zeros[1] += torch.sum(after_fc2.eq(torch.zeros(after_fc2.size()).to(device)),dtype=torch.int)
                    num_zeros[2] += torch.sum(after_fc3.eq(torch.zeros(after_fc3.size()).to(device)),dtype=torch.int)
                    w.add_histogram("fc1", after_fc1.clone().cpu().data.numpy(), i)
                    w.add_histogram("fc2", after_fc2.clone().cpu().data.numpy(), i)
                    w.add_histogram("fc3", after_fc3.clone().cpu().data.numpy(), i)
                num_zeros = num_zeros.div(10000).round()
                print("#Dead activations of {}-{}: {:d}-{:d}-{:d}".format( \
                    activation, reg_type, \
                    int(num_zeros[0].data),int(num_zeros[1].data),int(num_zeros[2].data)))

if __name__ == '__main__':
    main()
