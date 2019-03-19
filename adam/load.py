from __future__ import print_function
import torch
import numpy as np
import torch.optim as optim
from adam import Adam
import matplotlib.pyplot as plt
from mnist import LOGISTIC, FC
import os, argparse


def num_flat_features(x):
    size = x.size()[1:]  # all dimensions except the batch dimension
    num_features = 1
    for s in size:
        num_features *= s
    return num_features

def main(args):
    drop_rate = args.drop_rate
    data_dir = args.data_dir
    save_dir = args.save_dir
    # Training settings
    device = torch.device("cuda")
    model_types = ["fc", "logistic"] if not drop_rate else ["fc"]
    for model_type in model_types:
        activations = ['relu', 'sigmoid'] if model_type=="fc" else ["None"]
        fig, axs = plt.subplots(1, len(activations), figsize=(10*len(activations), 10))
        for i, activation in enumerate(activations):
            if len(activations) == 1:
                ax = axs
            else:
                ax = axs[i]
            if model_type == "logistic":
                model = LOGISTIC().to(device)
            else:
                model = FC(activation=activation).to(device)
            weight_decay = 1e-6 #args.weight_decay
            for opt_type in ['adam', 'adagrad', 'rmsprop', 'sgd']:
                # checkpoint loading
                PATH = os.path.join(data_dir,"{}/{}_{}_{}.pt".format("drop" if drop_rate else "base", activation, model_type, opt_type))
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
        fig.savefig(save_dir + "/{}_{}.png".format("drop" if drop_rate else "base", model_type))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Example ADAM LOAD')
    parser.add_argument('--data-dir', type=str, default="./saved_models/",
                        help='Model files Directory')
    parser.add_argument('--save-dir', type=str, default="./results/",
                        help='Directory to be saved')
    parser.add_argument('--drop-rate', type=float, default=0.,
                            help='Dropout rate to be zero')
    args = parser.parse_args()
    main(args)
