from __future__ import print_function
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
import torchvision.utils as vutils
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
    acc_dict = {}
    time_dict = {}
    loss_dict = {}
    for activation in ['soft_plus', 'relu', 'prelu']:
        acc_dict[activation] = {}
        time_dict[activation] = {}
        loss_dict[activation] = {}
        for reg_type in ["base", "L1", "L2"]:
            acc_list=[]
            time_list=[]
            loss_list=[]
            for manualSeed in [1*(i+1) for i in range(9)]:
            # for manualSeed in [2]:
                model = Net(activation=activation, reg_type=reg_type).to(device)
                # print("\nRandom Seed\n: ", manualSeed)
                # checkpoint loading
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
                losses = checkpoint['losses']
                accuracies = checkpoint['accuracies']
                times = checkpoint["times"]
                # checkpoint load end
                acc_list.append(accuracies)
                time_list.append(times)
                loss_list.append(losses)
                acc_dict[activation][reg_type] = np.mean(acc_list, axis=0)
                time_dict[activation][reg_type] = np.mean(time_list, axis=0)
                loss_dict[activation][reg_type] = np.mean(loss_list, axis=0)

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
    plt.figure(figsize=(18,12))
    for i, activation in enumerate(time_dict):
        plt.subplot(1,len(time_dict),i+1)
        plt.title(activation)
        for reg_type in time_dict[activation]:
            plt.plot(time_dict[activation][reg_type],label=reg_type)
        plt.xlabel("epochs")
        plt.ylabel("time(sec)")
    plt.legend()
    plt.show()
    for activation in time_dict:
        for reg_type in time_dict[activation]:
            print("Time {}/{}:  {}".format(activation, reg_type, sum(time_dict[activation][reg_type])))
    # loss graph
    plt.figure(figsize=(18,12))
    for i, activation in enumerate(loss_dict):
        plt.subplot(1,len(loss_dict),i+1)
        plt.title(activation)
        for reg_type in loss_dict[activation]:
            plt.plot(loss_dict[activation][reg_type],label=reg_type)
        plt.xlabel("epochs")
        plt.ylabel("loss")
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()
