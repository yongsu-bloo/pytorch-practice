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
    acc_dict = {}
    time_dict = {}
    for activation in ['relu', 'lrelu', 'prelu']:
        acc_dict[activation] = {}
        time_dict[activation] = {}
        for reg_type in ["base", "L1", "L2",'elastic']:
            acc_list=[]
            time_list=[]
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
                times = checkpoint["times"]
                # checkpoint load end
                acc_list.append(accuracies)
                time_list.append(times)
                acc_dict[activation][reg_type] = np.mean(acc_list, axis=0)
                time_dict[activation][reg_type] = np.mean(time_list, axis=0)

    # acc_mean = {}
    # for acc_dict in acc_list:
    #     for activation in acc_dict:
    #         acc_mean[activation] = {}
    #         for reg_type in acc_dict[activation]:
    #             acc_mean[activation][reg_type] = \
    #             np.mean(np.array([ acc_dict[activation][reg_type] for acc_dict in acc_list ]), axis=0)
    # print(acc_mean)
    # accuracy graph
    plt.figure(figsize=(18,12))
    for i, activation in enumerate(acc_dict):
        plt.subplot(1,len(acc_dict),i+1)
        plt.title(activation)
        for reg_type in acc_dict[activation]:
            plt.plot(acc_dict[activation][reg_type],label=reg_type)
        plt.xlabel("epochs")
        plt.ylabel("Accuracy")
    plt.legend()
    plt.show()
    # time graph
    # plt.figure(figsize=(18,12))
    # for i, activation in enumerate(time_dict):
    #     plt.subplot(1,len(time_dict),i+1)
    #     plt.title(activation)
    #     for reg_type in time_dict[activation]:
    #         plt.plot(time_dict[activation][reg_type],label=reg_type)
    #     plt.xlabel("epochs")
    #     plt.ylabel("time(sec)")
    # plt.legend()
    # plt.show()
    for activation in time_dict:
        for reg_type in time_dict[activation]:
            print("Time {}/{}:  {}".format(activation, reg_type, sum(time_dict[activation][reg_type])))

if __name__ == '__main__':
    main()
