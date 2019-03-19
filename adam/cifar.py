from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from .adam import Adam
import time
from torchvision import datasets, transforms

class FC(nn.Module):
    def __init__(self, activation="relu", reg_type="base"):
        super(Net, self).__init__()
        self.activation = activation
        self.reg_type = reg_type
        self.fc1 = nn.Linear(784, 300)
        self.fc2 = nn.Linear(300, 100)
        self.fc3 = nn.Linear(100, 10)
        self.activations = nn.ModuleDict([
            ['relu', nn.ReLU()],
            ['lrelu', nn.LeakyReLU()]
        ])
        if reg_type == "l2":
            self.lambda1 = 0.
            self.lambda2 = 0.001
        elif reg_type == "l1":
            self.lambda1 = 0.001
            self.lambda2 = 0.
        elif reg_type == "elastic":
            self.lambda1 = 0.0009
            self.lambda2 = 0.0001
        else: # base case
            self.lambda1 = 0.
            self.lambda2 = 0.

    def forward(self, x):
        x = x.view(-1, self.num_flat_features(x)) #flattened input 64 x 64 = 784
        x = self.activations[self.activation](self.fc1(x))
        x = self.activations[self.activation](self.fc2(x))
        x = self.activations[self.activation](self.fc3(x))
        return F.log_softmax(x, dim=1)

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        l1_regularization, l2_regularization = torch.tensor(0.).to(device), torch.tensor(0.).to(device)
        optimizer.zero_grad()
        output = model(data)
        nll_loss = F.nll_loss(output, target)
        for param in model.parameters():
            l1_regularization += torch.norm(param, 1)
            l2_regularization += torch.norm(param, 2)
        loss = nll_loss + model.lambda1 * l1_regularization + model.lambda2 * l2_regularization
        loss.backward()
        optimizer.step()
        params = list(network.parameters())
        if batch_idx % args.log_interval == 0:
            print('[{}][{}]Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                model.activation, model.reg_type,
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            print(params)
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
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--betas', nargs='+', type=float, default=(0.9, 0.999), help='Beta values for Adam')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    # for manualSeed in [111*(i+1) for i in range(9)]:
    for manualSeed in [1]:
        # manualSeed = random.randint(1, 10000) # use if you want new results
        print("Random Seed: ", manualSeed)
        # random.seed(manualSeed)
        torch.manual_seed(manualSeed)

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
        print("Training/Test Data size: {}, {}".format(len(train_loader), len(test_loader)))
        for activation in ['relu', 'lrelu','prelu']:
            for reg_type in ["base", "L1", "L2", "elastic"]:
                model = FC(activation=activation, reg_type=reg_type).to(device)
                optimizer = Adam(model.parameters(), lr=args.lr, momentum=args.momentum)
                losses = []
                accuracies = []
                test_times = []
                for epoch in range(1, args.epochs + 1):
                    train(args, model, device, train_loader, optimizer, epoch)
                    t1 = time.time()
                    test_loss, accuracy = test(args, model, device, test_loader)
                    test_time = time.time() - t1
                    losses.append(test_loss)
                    accuracies.append(accuracy)
                    test_times.append(test_time)

                PATH = "./saved_models/{}_{}_{}.pt".format(activation, reg_type, manualSeed)
                torch.save({
                        'args': args,
                        'losses': losses,
                        'accuracies': accuracies,
                        "times": test_times,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        }, PATH)
                print("\nModel saved in \n" + PATH)
        # load and accuracy comparison
        print("\nRandom Seed\n: ", manualSeed)
        for activation in ['relu', 'lrelu', 'prelu']:
            for reg_type in ["base", "L1", "L2"]:
                model = Net(activation=activation, reg_type=reg_type).to(device)
                optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
                PATH = "./saved_models/{}_{}_{}.pt".format(activation, reg_type, manualSeed)
                checkpoint = torch.load(PATH)
                model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                args = checkpoint['args']
                losses = checkpoint['losses']
                accuracies = checkpoint['accuracies']
                print("Final loss and accuracy of {}_{} are {} and {}".format(
                    activation, reg_type, losses[-1], accuracies[-1]
                ))

if __name__ == '__main__':
    main()
