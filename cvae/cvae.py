# Todo changing from cvae_wtf
import torch
import torch.nn.functional as nn
import torch.autograd as autograd
import torch.optim as optim
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
from torch.autograd import Variable
from torchvision import datasets, transforms
# helper functions
def to_one_hot(y, n_dims=None):
    """ Take integer y (tensor or variable) with n dims and convert it to 1-hot representation with n+1 dims. """
    y_tensor = y.data if isinstance(y, Variable) else y
    y_tensor = y_tensor.type(torch.LongTensor).view(-1, 1)
    n_dims = n_dims if n_dims is not None else int(torch.max(y_tensor)) + 1
    y_one_hot = torch.zeros(y_tensor.size()[0], n_dims).scatter_(1, y_tensor, 1)
    y_one_hot = y_one_hot.view(*y.shape, -1)
    return Variable(y_one_hot) if isinstance(y, Variable) else y_one_hot

def num_flat_features(x):
    size = x.size()[1:]  # all dimensions except the batch dimension
    num_features = 1
    for s in size:
        num_features *= s
    return num_features
def flatten(x):
    return x.view(x.size()[0], num_flat_features(x))

# mnist = input_data.read_data_sets('../../MNIST_data', one_hot=True)
mnist = datasets.MNIST('../data', train=True, download=True,
               transform=transforms.Compose([
                   transforms.ToTensor(),
                   # transforms.Normalize((0.1307,), (0.3081,))
               ]))
mb_size = 1000
Z_dim = 100
num_classes = 10
X_dim = mnist.data.shape[1] # mnist.train.images.shape[1]
one_hot_target = to_one_hot(mnist.targets)
y_dim = one_hot_target.shape[1] # mnist.train.labels.shape[1]
h_dim = 128
cnt = 0
lr = 1e-3

class CVAE(nn.Module):
    def __init__(self):
        super(CVAE, self).__init__()
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        # For Encoder
        self.fcE = nn.Linear(X_dim + y_dim, h_dim)
        self.fcE_mu = nn.Linear(h_dim, Z_dim)
        self.fcE_var = nn.Linear(h_dim, Z_dim)
        # For Decoder
        self.fcD1 = nn.Linear(Z_dim + y_dim, h_dim)
        self.fcD2 = nn.Linear(h_dim, X_dim)

    def forward(self, x, c):
        x = x.view(-1, num_flat_features(x)) #flattened input 64 x 64 = 784
        # Encoder
        inputs = torch.cat([x, c], 1)
        out = self.fcE(inputs)
        z_mu = self.fc_mu(out)
        z_var = self.fc_var(out)
        z = sample_z(z_mu, z_var)
        # Decoder

        return F.log_softmax(x, dim=1)


def Q(X, c):
    inputs = torch.cat([X, c], 1)
    h = nn.relu(inputs @ Wxh + bxh.repeat(inputs.size(0), 1))
    z_mu = h @ Whz_mu + bhz_mu.repeat(h.size(0), 1)
    z_var = h @ Whz_var + bhz_var.repeat(h.size(0), 1)
    return z_mu, z_var


def sample_z(mu, log_var):
    eps = Variable(torch.randn(mb_size, Z_dim))
    return mu + torch.exp(log_var / 2) * eps

def P(z, c):
    inputs = torch.cat([z, c], 1)
    h = nn.relu(inputs @ Wzh + bzh.repeat(inputs.size(0), 1))
    X = nn.sigmoid(h @ Whx + bhx.repeat(h.size(0), 1))
    return X


# =============================== TRAINING ====================================

params = [Wxh, bxh, Whz_mu, bhz_mu, Whz_var, bhz_var,
          Wzh, bzh, Whx, bhx]

solver = optim.Adam(params, lr=lr)
use_cuda = torch.cuda.is_available()
kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
train_loader = iter(torch.utils.data.DataLoader(mnist, batch_size=mb_size, shuffle=True, **kwargs))
for it in range(100000):
    X, c = next(train_loader) # mnist.data.next_batch(mb_size)
    X = flatten(X)
    c = to_one_hot(c)

    X = Variable(X)
    c = Variable(c.float())

    # Forward
    z_mu, z_var = Q(X, c)
    z = sample_z(z_mu, z_var)
    X_sample = P(z, c)

    # Loss
    recon_loss = nn.binary_cross_entropy(X_sample, X, size_average=False) / mb_size
    kl_loss = torch.mean(0.5 * torch.sum(torch.exp(z_var) + z_mu**2 - 1. - z_var, 1))
    loss = recon_loss + kl_loss

    # Backward
    loss.backward()

    # Update
    solver.step()

    # Housekeeping
    for p in params:
        if p.grad is not None:
            data = p.grad.data
            p.grad = Variable(data.new().resize_as_(data).zero_())

    # Print and plot every now and then
    if it % 1000 == 0:
        print('Iter-{}; Loss: {:.4}'.format(it, loss.data[0]))

        c = np.zeros(shape=[mb_size, y_dim], dtype='float32')
        c[:, np.random.randint(0, 10)] = 1.
        c = Variable(torch.from_numpy(c))
        z = Variable(torch.randn(mb_size, Z_dim))
        samples = P(z, c).data.numpy()[:16]

        fig = plt.figure(figsize=(4, 4))
        gs = gridspec.GridSpec(4, 4)
        gs.update(wspace=0.05, hspace=0.05)

        for i, sample in enumerate(samples):
            ax = plt.subplot(gs[i])
            plt.axis('off')
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_aspect('equal')
            plt.imshow(sample.reshape(28, 28), cmap='Greys_r')

        if not os.path.exists('out/'):
            os.makedirs('out/')

        plt.savefig('out/{}.png'.format(str(cnt).zfill(3)), bbox_inches='tight')
        cnt += 1
        plt.close(fig)
