from __future__ import print_function
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="3"
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from adam import Adam
from torchvision import datasets, transforms
import time

def num_flat_features(x):
    size = x.size()[1:]  # all dimensions except the batch dimension
    num_features = 1
    for s in size:
        num_features *= s
    return num_features

class LOGISTIC(nn.Module):
    def __init__(self):
        super(LOGISTIC, self).__init__()
        self.type = "logistic"
        self.linear = nn.Linear(784, 10)
    def forward(self, x):
        x = x.view(-1, num_flat_features(x)) #flattened input 64 x 64 = 784
        return F.log_softmax(self.linear(x), dim=1)

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

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        l1_regularization, l2_regularization = torch.tensor(0.).to(device), torch.tensor(0.).to(device)
        optimizer.zero_grad()
        output = model(data)
        nll_loss = F.nll_loss(output, target)
        loss = nll_loss # + model.lambda1 * l1_regularization + model.lambda2 * l2_regularization
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('[{}]Train Epoch: {} \tLoss: {:.6f}'.format(
                model.type,
                epoch,
                loss.item()))
        return loss.item()
def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        accuracy))
    return test_loss, accuracy

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch Example')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=200, metavar='N',
                        help='number of epochs to train (default: 200)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed to use. Default=0')
    parser.add_argument('--weight-decay', '--wd', type=float, default=1e-6, metavar='W',
                        help='weight decay (default: 0)')
    parser.add_argument('--betas', nargs='+', type=float, default=(0.9, 0.999),
                        help='Beta values for Adam')
    parser.add_argument('--model', type=str, default="fc",
                        help='Model type: [logistic, fc, cnn]')
    parser.add_argument('--optimizer', type=str, default="adam",
                        help='Optimizer type: [adam, adagrad, rmsprop]')
    parser.add_argument('--drop-rate', type=float, default=0.,
                        help='Dropout rate to be zero')
    parser.add_argument('--save-dir', type=str, default="./saved_models/",
                        help='Directory to be saved')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    manual_seed = args.seed
    # print("Random Seed: ", manual_seed)
    # random.seed(manual_seed)
    save_dir = args.save_dir
    torch.manual_seed(manual_seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)
    model_types = ["fc", "logistic"] if not args.drop_rate else ["fc"]
    for model_type in model_types:
        activations = ['relu', 'sigmoid'] if model_type=="fc" else ["None"]
        for activation in activations:
            for opt_type in ['adam', 'adagrad', 'rmsprop', 'sgd']:
                if model_type == 'logistic':
                    model = LOGISTIC().to(device)
                else:
                    model = FC(activation=activation, drop_rate=args.drop_rate).to(device)
                weight_decay = 1e-6 if model_type=="fc" else 0.
                optimizers = {'adam': Adam(model.parameters(), lr=args.lr, weight_decay = args.weight_decay, betas=args.betas),
                              'adagrad': optim.Adagrad(model.parameters(), lr=args.lr, weight_decay = args.weight_decay),
                              'rmsprop': optim.RMSprop(model.parameters(), lr=args.lr, weight_decay = args.weight_decay, alpha=args.betas[1], eps=1e-08, momentum=0),
                              'sgd': optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)}
                optimizer = optimizers[opt_type]
                train_losses = []
                test_losses = []
                accuracies = []
                test_times = []
                for epoch in range(1, args.epochs + 1):
                    train_loss = train(args, model, device, train_loader, optimizer, epoch)
                    t1 = time.time()
                    test_loss, accuracy = test(args, model, device, test_loader)
                    test_time = time.time() - t1
                    train_losses.append(train_loss)
                    test_losses.append(test_loss)
                    accuracies.append(accuracy)
                    test_times.append(test_time)

                PATH = os.path.join(save_dir, "{}/{}_{}_{}.pt".format("drop" if args.drop_rate else "base",activation, model_type, opt_type))
                torch.save({
                        'args': args,
                        'opt_type': opt_type,
                        'train_losses': train_losses,
                        'test_losses': test_losses,
                        'accuracies': accuracies,
                        "times": test_times,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        }, PATH)
                print("\nModel saved in \n" + PATH)

if __name__ == '__main__':
    main()
