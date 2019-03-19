from __future__ import print_function
import torch
import numpy as np
import torch.optim as optim
from adam import Adam
import matplotlib.pyplot as plt
from mnist import LOGISTIC, FC

drop_rate = 0.5
save_dir = "./results/"
def num_flat_features(x):
    size = x.size()[1:]  # all dimensions except the batch dimension
    num_features = 1
    for s in size:
        num_features *= s
    return num_features

def main():
    # Training settings
    device = torch.device("cuda")
    model_types = ["fc", "logistic"] if not drop_rate else ["fc"]
    for model_type in model_types:
        activations = ['relu', 'sigmoid'] if model_type=="fc" else [None]
        fig, axs = plt.subplots(1, len(activations), figsize=(20, 5*len(activations)))
        for i, activation in enumerate(activations):
            if len(activations) == 1:
                ax = axs
            else:
                ax = axs[i]
            if model_type == "logistic":
                model = LOGISTIC().to(device)
            else:
                model = FC(activation=activation).to(device)
            weight_decay = 0.0001 if model_type=="fc" else 0.
            for opt_type in ['adam', 'adagrad', 'rmsprop', 'sgd']:
                # checkpoint loading
                PATH = "./saved_models/{}/{}_{}_{}.pt".format("drop" if drop_rate else "base", activation, model_type, opt_type)
                checkpoint = torch.load(PATH)
                args = checkpoint['args']
                model.load_state_dict(checkpoint['model_state_dict'])
                optimizers = {
                            'adam': Adam(model.parameters(), lr=args.lr, weight_decay = weight_decay, betas=args.betas),
                            'adagrad': optim.Adagrad(model.parameters(), lr=args.lr, weight_decay = weight_decay),
                            'rmsprop': optim.RMSprop(model.parameters(), lr=args.lr, weight_decay = weight_decay, alpha=args.betas[1], eps=1e-08, momentum=0),
                            'sgd': optim.SGD(model.parameters(), lr=args.lr, weight_decay=weight_decay, momentum=args.momentum)
                }
                optimizer = optimizers[opt_type]
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                train_losses = checkpoint['train_losses']
                test_losses = checkpoint['test_losses']
                accuracies = checkpoint['accuracies']
                times = checkpoint["times"]
                opt_type = checkpoint["opt_type"]
                # plot
                ax.plot(train_losses, label=opt_type)
            ax.set(title="MNIST " + activation, xlabel="epochs", ylabel="loss", ylim=(0., 2.5))
            ax.grid()
            ax.legend()
        fig.tight_layout()
        fig.savefig(save_dir + "/{}/{}.png".format("drop" if drop_rate else "base", model_type))

    # # accuracy graph
    # plt.figure(figsize=(18,12))
    # for i, activation in enumerate(acc_dict):
    #     plt.subplot(1,len(acc_dict),i+1)
    #     plt.title(activation)
    #     for reg_type in acc_dict[activation]:
    #         plt.plot(acc_dict[activation][reg_type],label=reg_type)
    #     plt.xlabel("epochs")
    #     plt.ylabel("Accuracy")
    # plt.legend()
    # plt.show()
    # # time graph
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
    # for activation in time_dict:
    #     for reg_type in time_dict[activation]:
    #         print("Time {}/{}:  {}".format(activation, reg_type, sum(time_dict[activation][reg_type])))
    # # loss graph
    # plt.figure(figsize=(18,12))
    # for i, activation in enumerate(loss_dict):
    #     plt.subplot(1,len(loss_dict),i+1)
    #     plt.title(activation)
    #     for reg_type in loss_dict[activation]:
    #         plt.plot(loss_dict[activation][reg_type],label=reg_type)
    #     plt.xlabel("epochs")
    #     plt.ylabel("loss")
    # plt.legend()
    # plt.show()

if __name__ == '__main__':
    main()
